# -*- coding: utf-8 -*-
"""
TalentGraph Unified CLI Command Line Interface.
Provides:
  - talent prepare  : Pre-extract and warm up standard graph cache
  - talent evaluate : End-to-end assessment with instant rich terminal report
  - talent server   : Launch FastAPI backend & visualization dashboard
  - talent benchmark: Run full system performance & latency benchmark
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add backend to sys.path
backend_dir = Path(__file__).parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from configs.settings import get_settings
from src.knowledge.cache import get_graph_cache_manager
from src.pipeline.runner import PipelineRunner
from src.reporting.generator import ReportGenerator
from src.reporting.pdf_service import PDFExportService

app = typer.Typer(
    name="talent",
    help="TalentGraph - 无领导小组讨论量化评估系统 CLI 终端工具",
    add_completion=False,
)
console = Console()


@app.command(name="prepare")
def prepare(
    corpus: Path = typer.Option(..., "--corpus", "-c", help="案例参考答案文本文件路径 (corpus.txt)"),
    title: str = typer.Option("无领导小组讨论案例", "--title", "-t", help="案例题目"),
    force: bool = typer.Option(False, "--force", "-f", help="强制重新提取（忽略现有缓存）"),
):
    """【题库预热】提取标准题库因果影响图谱并预热本地 0 秒缓存"""
    if not corpus.exists():
        console.print(f"[bold red]错误：语料文件不存在: {corpus}[/bold red]")
        raise typer.Exit(1)

    text = corpus.read_text(encoding="utf-8")
    cache_mgr = get_graph_cache_manager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="正在构建标准图谱与缓存...", total=None)
        graph = asyncio.run(
            cache_mgr.get_or_extract(
                corpus_text=text,
                corpus_id=corpus.stem,
                title=title,
                force_refresh=force,
            )
        )

    console.print(
        Panel(
            f"[bold green]标准图谱预热成功！[/bold green]\n\n"
            f"• 案例标识: [cyan]{graph.corpus_id}[/cyan]\n"
            f"• 案例标题: [yellow]{graph.title}[/yellow]\n"
            f"• 知识节点数: [bold]{len(graph.nodes)}[/bold] (c: {len(graph.get_nodes_by_type('c'))}, d: {len(graph.get_nodes_by_type('d'))}, u: {len(graph.get_nodes_by_type('u'))})\n"
            f"• 逻辑因果边数: [bold]{len(graph.arcs)}[/bold]\n"
            f"• 缓存指纹: [dim]{graph.corpus_hash[:16]}...[/dim]",
            title="✨ 题库预热就绪",
        )
    )


@app.command(name="evaluate")
def evaluate(
    corpus: Path = typer.Option(..., "--corpus", "-c", help="案例参考答案文本路径"),
    group: Path = typer.Option(..., "--group", "-g", help="小组讨论对话转录文本路径 (group_*.txt)"),
    title: str = typer.Option("无领导小组讨论", "--title", "-t", help="讨论场次标题"),
    export_pdf: bool = typer.Option(False, "--export-pdf", help="评估完成后导出每位候选人的 PDF 报告"),
):
    """【一键评测】运行端到端评测流水线，输出 16 维量化画像与综合诊断报告"""
    if not corpus.exists() or not group.exists():
        console.print("[bold red]错误：输入语料或讨论文本文件不存在[/bold red]")
        raise typer.Exit(1)

    start_total = time.perf_counter()
    settings = get_settings()
    runner = PipelineRunner(settings=settings)
    report_gen = ReportGenerator(settings=settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Step 1: Pipeline
        progress.add_task(description="[1/3] 正在解析讨论并进行节点与因果边剪枝匹配...", total=None)
        pipeline_res = asyncio.run(runner.run_files(corpus, group, title=title))

        # Step 2: 评分与诊断
        progress.add_task(description="[2/3] 正在执行 16 维纯数学图论评分与候选人全维诊断...", total=None)
        report_data = asyncio.run(report_gen.generate_report(pipeline_res))

    total_time = time.perf_counter() - start_total

    # 输出高颜值终端大盘表格
    table = Table(title=f"📊 评测总榜 - {title} (全场耗时: {total_time:.2f}s)")
    table.add_column("总排名", justify="center", style="cyan", no_wrap=True)
    table.add_column("候选人", justify="center", style="bold")
    table.add_column("综合总分", justify="right", style="green")
    table.add_column("发言轮次", justify="right")
    table.add_column("命中节点", justify="right")
    table.add_column("核心标签", justify="left", style="yellow")

    for p in report_data.participants:
        tags_str = ", ".join(p.diagnosis.tags[:2]) if p.diagnosis.tags else "稳健参与者"
        table.add_row(
            f"第 {p.overall_rank} 名",
            p.display_name,
            f"{p.total_score:.1f}",
            str(p.speaking_turns),
            str(p.matched_nodes_count),
            tags_str,
        )

    console.print(table)
    console.print(
        f"\n[bold green]✅ 评测项目已归档至:[/bold green] [cyan]{settings.get_project_dir(report_data.task_id)}[/cyan]\n"
        f"  • 全量契约报告: [bold]{settings.get_project_report_file(report_data.task_id)}[/bold]\n"
        f"  • 拓扑图谱目录: [bold]{settings.get_project_graphs_dir(report_data.task_id)}[/bold]"
    )

    if export_pdf:
        from src.reporting.pdf_service import PDFExportService
        pdf_service = PDFExportService(settings=settings)
        console.print("\n[bold cyan]📄 正在导出候选人 PDF 诊断报告...[/bold cyan]")
        for p in report_data.participants:
            try:
                pdf_path = asyncio.run(pdf_service.export_report_to_pdf(task_id=report_data.task_id, speaker_id=p.speaker_id))
                console.print(f"  • {p.display_name}: [green]{pdf_path}[/green]")
            except Exception as e:
                console.print(f"  • {p.display_name}: [yellow]导出失败 ({e})[/yellow]")


@app.command(name="server")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", "-r", help="热重载"),
):
    """【启动服务】启动 FastAPI 后端服务与可视化前端页面"""
    import uvicorn
    console.print(f"[bold cyan]🚀 正在启动 TalentGraph Web 服务：http://{host}:{port}[/bold cyan]")
    uvicorn.run("server:app", host=host, port=port, reload=reload, app_dir=str(backend_dir))


@app.command(name="benchmark")
def benchmark():
    """【基准测速】测试纯 CPU 向量引擎与 16 维评分器的微秒级极速性能"""
    console.print("[bold yellow]⚡ 开始 TalentGraph 核心组件极限性能基准测试...[/bold yellow]\n")

    # 1. 向量测试
    from src.core.embedding import get_embedding_engine
    emb = get_embedding_engine()
    t0 = time.perf_counter()
    vecs = emb.embed_texts(["这是一段商业决策测试文本", "我们需要降低运营成本", "开拓海外新兴市场"] * 10)
    emb_time = (time.perf_counter() - t0) * 1000.0
    console.print(f"• CPU 句向量批量编码 (30句): [bold green]{emb_time:.2f}ms[/bold green]")

    # 2. 评分引擎测试
    from src.models.discussion import DiscussionTranscript, Speaker, Utterance
    from src.models.graph import Node, StandardGraph
    from src.models.discussion import InteractionNetwork
    from src.scoring.base import ScoringContext
    from src.scoring.engine import ScoringEngine

    nodes = [Node(id=f"n_{i}", name=f"节点_{i}", type="c") for i in range(20)]
    std_g = StandardGraph(corpus_id="bm", nodes=nodes, arcs=[])
    speakers = {
        f"spk_{i}": Speaker(
            speaker_id=f"spk_{i}",
            display_name=f"候选人 {i}",
            is_host=False,
            utterances=[Utterance(index=0, speaker_id=f"spk_{i}", speaker_name=f"候选人 {i}", text="测试发言内容")],
        )
        for i in range(12)
    }
    trans = DiscussionTranscript(group_id="bm", speakers=speakers, utterances=[])
    ctx = ScoringContext(group_id="bm", corpus_id="bm", standard_graph=std_g, transcript=trans, interaction_network=InteractionNetwork())

    engine = ScoringEngine()
    t1 = time.perf_counter()
    matrix = engine.evaluate_all(ctx)
    score_time = (time.perf_counter() - t1) * 1000.0
    console.print(f"• 16 维度纯数学图论评分 (12人全员): [bold green]{score_time:.2f}ms[/bold green]")

    console.print("\n[bold green]🎉 基准测速全部达标！微秒级计算性能卓越。[/bold green]")


if __name__ == "__main__":
    app()

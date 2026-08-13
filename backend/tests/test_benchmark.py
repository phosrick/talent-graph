# -*- coding: utf-8 -*-
"""
Performance Benchmark & System Verification Suite.
Validates latency targets (<60s end-to-end, <100ms pure scoring) and 16-dimension score consistency.
"""

import time
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.core.embedding import get_embedding_engine
from src.core.llm_client import LLMClient
from src.knowledge.cache import GraphCacheManager
from src.models.discussion import DiscussionTranscript, InteractionNetwork, Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, StandardGraph
from src.models.report import ComprehensiveDiagnosis
from src.pipeline.runner import PipelineRunner
from src.reporting.generator import ReportGenerator
from src.reporting.summarizer import CandidateSummarizer
from src.scoring.base import ScoringContext
from src.scoring.engine import ScoringEngine


def test_pure_scoring_latency_under_50ms():
    """验证 16 个评分器在 12 位候选人全量场景下的纯内存计算时间 <= 50ms (老系统需数百秒)"""
    nodes = [Node(id=f"n_{i}", name=f"知识节点_{i}", type="c" if i < 10 else ("d" if i < 16 else "u")) for i in range(20)]
    arcs = [Arc(source=f"n_{i}", target=f"n_{i+1}", type="causal") for i in range(15)]
    std_graph = StandardGraph(corpus_id="bm_case", title="基准测试题库", nodes=nodes, arcs=arcs)

    speakers = {}
    cand_graphs = {}
    for spk_idx in range(1, 13):
        spk_id = f"speaker_{spk_idx}"
        u = Utterance(index=spk_idx - 1, speaker_id=spk_id, speaker_name=f"候选人 {spk_idx}", text="这是基准性能测试发言文本。", start_time_str="01:00", start_seconds=60.0)
        speakers[spk_id] = Speaker(speaker_id=spk_id, display_name=f"候选人 {spk_idx}", utterances=[u], is_host=False, turn_count=2, total_duration_seconds=90.0)
        cand_graphs[spk_id] = CandidateGraph(speaker_id=spk_id, matched_nodes=nodes[:5], verified_arcs=arcs[:3])

    transcript = DiscussionTranscript(group_id="bm_group", title="基准讨论", speakers=speakers, utterances=list(speakers.values())[0].utterances, total_duration_seconds=1800.0)
    interaction = InteractionNetwork(speakers=list(speakers.keys()), edges=[])

    context = ScoringContext(
        group_id="bm_group",
        corpus_id="bm_case",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs=cand_graphs,
        interaction_network=interaction,
    )

    engine = ScoringEngine()
    start_time = time.perf_counter()
    score_matrix = engine.evaluate_all(context)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[BENCHMARK] 16 Dimensions across 12 candidates evaluated in {elapsed_ms:.2f}ms")
    assert elapsed_ms < 50.0, f"Scoring took {elapsed_ms:.2f}ms, expected < 50ms"
    assert len(score_matrix.participants) == 12
    assert len(score_matrix.dimension_averages) == 16


@pytest.mark.asyncio
async def test_end_to_end_pipeline_and_reporting_speed():
    """验证包含剪枝匹配、16维评分与诊断聚合的全流程调度极速闭环"""
    nodes = [
        Node(id="c1", name="市场背景", type="c"),
        Node(id="d1", name="智能化转型", type="d"),
        Node(id="u1", name="市场份额", type="u"),
    ]
    arcs = [Arc(source="c1", target="d1", type="info"), Arc(source="d1", target="u1", type="causal")]
    std_graph = StandardGraph(corpus_id="c_test", title="标准案例", nodes=nodes, arcs=arcs)

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock()

    mock_cache = MagicMock(spec=GraphCacheManager)
    mock_cache.get_or_extract = AsyncMock(return_value=std_graph)

    # 1. Pipeline
    runner = PipelineRunner(llm_client=mock_llm, cache_manager=mock_cache)
    mock_summarizer = MagicMock(spec=CandidateSummarizer)
    mock_summarizer.summarize_all_participants = AsyncMock(
        return_value={
            f"speaker_{i}": ComprehensiveDiagnosis(summary="表现优异", strengths=["逻辑严密"], weaknesses=[], tags=["领航者"])
            for i in range(2, 6)
        }
    )
    generator = ReportGenerator(summarizer=mock_summarizer)

    corpus_sample = "这是专家标准答案。包含核心市场背景、智能化转型决策以及提升市场份额效用。"
    discussion_sample = """
    说话人 1 00:00
    欢迎大家参加无领导小组讨论。
    说话人 2 00:30
    我们要以提升市场份额为核心目标，推动智能化转型。
    说话人 3 01:15
    我完全赞同2号的战略方向，在预算层面我们要精细化核算。
    说话人 4 02:00
    关于执行路径，我建议设立专门的项目攻坚小组。
    说话人 5 02:45
    我补充一点，同时要做好各部门之间的协同机制。
    """

    t0 = time.perf_counter()
    pipeline_res = await runner.run(
        corpus_text=corpus_sample,
        discussion_text=discussion_sample,
        group_id="test_bm_group",
    )
    report_data = await generator.generate_report(pipeline_res, task_id="task_bm_test")
    total_elapsed = time.perf_counter() - t0

    print(f"\n[BENCHMARK] End-to-end evaluation & reporting finished in {total_elapsed:.2f}s")
    assert total_elapsed < 5.0  # 远低于 60 秒上限
    assert report_data.team_overview.participant_count == 4
    assert len(report_data.participants) == 4
    assert report_data.participants[0].total_score > 0

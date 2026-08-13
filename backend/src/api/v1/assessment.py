# -*- coding: utf-8 -*-
"""
Assessment API Routes.
Endpoints for submitting evaluation tasks, checking progress, retrieving ReportData, and exporting PDF.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from configs.settings import get_settings
from src.api.websocket import get_progress_manager
from src.core.logger import get_logger
from src.models.report import ReportData
from src.pipeline.runner import PipelineRunner
from src.reporting.generator import ReportGenerator
from src.reporting.pdf_service import PDFExportService

logger = get_logger("assessment_api")
router = APIRouter(prefix="/assessment", tags=["Assessment"])

# 内存任务状态存储 (生产环境中可放 Redis)
TASK_STORE: dict[str, dict] = {}


class AssessmentSubmitRequest(BaseModel):
    corpus_text: str = Field(..., description="案例标准参考语料文本")
    discussion_text: str = Field(..., description="小组讨论转录文本 (包含发言人与时间戳)")
    group_id: Optional[str] = Field(default=None, description="讨论小组标识 (如 271_1)")
    corpus_id: Optional[str] = Field(default=None, description="题库语料标识")
    title: Optional[str] = Field(default="无领导小组讨论", description="案例标题")
    project_id: Optional[str] = Field(default="default_project", description="所属项目标识")


class AssessmentSubmitResponse(BaseModel):
    task_id: str = Field(..., description="评测任务唯一ID")
    status: str = Field(default="processing", description="任务状态: processing / completed / failed")
    message: str = Field(default="任务已成功提交", description="状态提示")


async def run_assessment_task(
    task_id: str,
    req: AssessmentSubmitRequest,
):
    """异步后台运行完整评测流水线并实时推送进度"""
    settings = get_settings()
    pm = get_progress_manager()
    runner = PipelineRunner(settings=settings)
    report_gen = ReportGenerator(settings=settings)

    try:
        TASK_STORE[task_id] = {"status": "processing", "progress": 5, "stage": "INIT"}
        await pm.broadcast_progress(task_id, 5, "INIT", "正在初始化任务环境...")

        # Stage 0: 图谱加载/提取
        await pm.broadcast_progress(task_id, 20, "STANDARD_GRAPH", "正在检索或提取题库标准图谱...")

        # Stage 1 & 2: Pipeline 执行
        await pm.broadcast_progress(task_id, 40, "PIPELINE", "正在解析发言并执行全场互动与节点边剪枝匹配...")
        pipeline_res = await runner.run(
            corpus_text=req.corpus_text,
            discussion_text=req.discussion_text,
            group_id=req.group_id or task_id,
            corpus_id=req.corpus_id or "corpus_default",
            title=req.title or "无领导小组讨论",
        )

        # Stage 3 & 4: 评分与综合诊断
        await pm.broadcast_progress(task_id, 75, "SCORING_AND_DIAGNOSIS", "正在进行 16 维量化图论评分与候选人全维诊断聚合...")
        report_data = await report_gen.generate_report(
            pipeline_result=pipeline_res,
            task_id=task_id,
            project_id=req.project_id or "default_project",
        )

        # 完成
        TASK_STORE[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "COMPLETED",
            "report_data": report_data,
        }
        await pm.broadcast_progress(task_id, 100, "COMPLETED", "评测已全部完成！", data={"group_id": report_data.group_id})
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
        TASK_STORE[task_id] = {"status": "failed", "progress": 0, "stage": "FAILED", "error": str(exc)}
        await pm.broadcast_progress(task_id, 0, "FAILED", f"评测失败: {exc}")


@router.post("/submit", response_model=AssessmentSubmitResponse)
async def submit_assessment(
    req: AssessmentSubmitRequest,
    background_tasks: BackgroundTasks,
):
    """提交无领导小组讨论评测任务"""
    if not req.corpus_text.strip():
        raise HTTPException(status_code=400, detail="Corpus text cannot be empty.")
    if not req.discussion_text.strip():
        raise HTTPException(status_code=400, detail="Discussion text cannot be empty.")

    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(req.discussion_text[:50])) % 10000}"
    background_tasks.add_task(run_assessment_task, task_id, req)

    return AssessmentSubmitResponse(
        task_id=task_id,
        status="processing",
        message="评测任务已提交并启动后台计算",
    )


@router.get("/{task_id}")
async def get_assessment_status(task_id: str):
    """查询评测任务当前进度状态"""
    if task_id not in TASK_STORE:
        settings = get_settings()
        report_file = settings.get_project_report_file(task_id)
        if report_file.exists():
            return {"task_id": task_id, "status": "completed", "progress": 100, "stage": "COMPLETED"}
        raise HTTPException(status_code=404, detail="Task not found.")

    t_info = TASK_STORE[task_id]
    return {
        "task_id": task_id,
        "status": t_info.get("status", "unknown"),
        "progress": t_info.get("progress", 0),
        "stage": t_info.get("stage", "UNKNOWN"),
        "error": t_info.get("error"),
    }


@router.get("/{task_id}/data", response_model=ReportData)
async def get_assessment_data(task_id: str):
    """获取供前端可视化大屏与报告渲染的完整 ReportData JSON"""
    # 优先从内存获取
    if task_id in TASK_STORE and "report_data" in TASK_STORE[task_id]:
        return TASK_STORE[task_id]["report_data"]

    # 尝试从磁盘加载
    settings = get_settings()
    report_file = settings.get_project_report_file(task_id)
    if report_file.exists():
        try:
            return ReportData.model_validate_json(report_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Corrupted report file: {e}")

    raise HTTPException(status_code=404, detail="Report data not ready or task not found.")


@router.get("/{task_id}/pdf")
async def get_assessment_pdf(
    task_id: str,
    speaker_id: Optional[str] = Query(default=None, description="候选人 ID (如 speaker_2)，默认导出第一名候选人"),
):
    """导出并下载候选人的高清矢量 A4 PDF 报告"""
    settings = get_settings()
    report_file = settings.get_project_report_file(task_id)

    # 确认评测报告数据存在
    if not report_file.exists() and (task_id not in TASK_STORE or "report_data" not in TASK_STORE[task_id]):
        raise HTTPException(status_code=404, detail="Report data not ready or task not found.")

    target_speaker_id = speaker_id
    if not target_speaker_id:
        if task_id in TASK_STORE and "report_data" in TASK_STORE[task_id]:
            report_data: ReportData = TASK_STORE[task_id]["report_data"]
            if report_data.participants:
                target_speaker_id = report_data.participants[0].speaker_id
        elif report_file.exists():
            try:
                report_data = ReportData.model_validate_json(report_file.read_text(encoding="utf-8"))
                if report_data.participants:
                    target_speaker_id = report_data.participants[0].speaker_id
            except Exception:
                pass

    target_speaker_id = target_speaker_id or "speaker_2"

    pdf_service = PDFExportService(settings=settings)
    pdf_path = await pdf_service.export_report_to_pdf(task_id=task_id, speaker_id=target_speaker_id)

    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="Failed to generate PDF export.")

    media_type = "application/pdf" if pdf_path.suffix == ".pdf" else "text/html"
    return FileResponse(
        path=str(pdf_path),
        filename=pdf_path.name,
        media_type=media_type,
    )

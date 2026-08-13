# -*- coding: utf-8 -*-
"""
TalentGraph FastAPI Server Entrypoint.
Provides REST APIs, WebSocket real-time progress, and static frontend hosting.
"""

import argparse
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from configs.settings import get_settings
from src.api.v1 import api_v1_router
from src.api.websocket import get_progress_manager
from src.core.logger import get_logger

logger = get_logger("server")
settings = get_settings()

app = FastAPI(
    title="TalentGraph - 无领导小组讨论量化评估系统",
    description="基于因果影响图谱与纯数学图论的高并发人才评估引擎",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 API 路由
app.include_router(api_v1_router, prefix="/api")


@app.get("/health", tags=["System"])
async def health_check():
    """系统健康检查探针"""
    return {
        "status": "healthy",
        "service": "talent_graph",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }


@app.get("/report-print/{task_id}/{speaker_id}", tags=["Reporting"])
async def get_report_print_page(task_id: str, speaker_id: str):
    """供 Playwright 无头浏览器与用户打印使用的 A4 报告母版页面"""
    print_html_file = Path(__file__).parent.parent / "frontend" / "dist" / "report-print.html"
    if print_html_file.exists():
        return FileResponse(print_html_file, media_type="text/html")
    return HTMLResponse("<h3>Report print template not found.</h3>", status_code=404)


@app.websocket("/ws/assessment/{task_id}")
async def websocket_assessment_progress(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送评测进度 (0% ~ 100%)"""
    pm = get_progress_manager()
    await pm.connect(websocket, task_id)
    try:
        while True:
            # 保持长连接心跳
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pm.disconnect(websocket, task_id)
    except Exception as e:
        logger.warning(f"WebSocket error on task {task_id}: {e}")
        pm.disconnect(websocket, task_id)


# 挂载前端静态页面 (若构建产物存在)
frontend_dist_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist_dir), html=True), name="frontend")
    logger.info(f"Mounted frontend static directory from {frontend_dist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Start TalentGraph API Server")
    parser.add_argument("--host", default=settings.app_host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.app_port, help="Port to listen")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logger.info(f"Starting TalentGraph Server on http://{args.host}:{args.port} (Docs: http://{args.host}:{args.port}/docs)")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

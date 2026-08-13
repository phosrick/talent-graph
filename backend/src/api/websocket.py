# -*- coding: utf-8 -*-
"""
WebSocket Connection & Real-Time Assessment Progress Manager.
"""

import asyncio
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
from src.core.logger import get_logger

logger = get_logger("websocket_manager")


class ProgressManager:
    """实时任务进度广播管理器"""

    def __init__(self):
        # task_id -> set of active WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # task_id -> latest progress payload
        self.task_progress: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections.setdefault(task_id, set()).add(websocket)
        logger.info(f"WebSocket client connected for task {task_id}")

        # 发送当前最新进度（若已有）
        if task_id in self.task_progress:
            await websocket.send_json(self.task_progress[task_id])

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket client disconnected for task {task_id}")

    async def broadcast_progress(
        self,
        task_id: str,
        progress: int,
        stage: str,
        message: str = "",
        data: dict = None,
    ):
        """向所有监听该任务的客户端推送进度 (0 ~ 100)"""
        payload = {
            "task_id": task_id,
            "progress": progress,
            "stage": stage,
            "message": message,
            "data": data or {},
        }
        self.task_progress[task_id] = payload

        if task_id in self.active_connections:
            websockets = list(self.active_connections[task_id])
            for ws in websockets:
                try:
                    await ws.send_json(payload)
                except Exception:
                    self.disconnect(ws, task_id)


_global_progress_manager = ProgressManager()


def get_progress_manager() -> ProgressManager:
    return _global_progress_manager

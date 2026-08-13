# -*- coding: utf-8 -*-
"""
Domain models for End-to-End Report Contract and Data Transfer Object (DTO).
Unified schema used for both Backend APIs, Frontend Visualizations, and PDF Print Motherboard.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.models.score import DimensionScore


class RadarPoint(BaseModel):
    """雷达图单点数据结构"""
    dimension_id: str = Field(..., description="维度 ID (如 1_1)")
    dimension_name: str = Field(..., description="维度短名 (如 逻辑思维)")
    category: str = Field(default="", description="能力大类")
    score: float = Field(..., description="个人得分")
    team_avg: float = Field(..., description="团队平均分")


class ComprehensiveDiagnosis(BaseModel):
    """个人全维综合诊断评价 (LLM 一次性生成)"""
    summary: str = Field(default="", description="全场整体综合表现评价")
    strengths: List[str] = Field(default_factory=list, description="核心优势与高光表现 (2~3条)")
    weaknesses: List[str] = Field(default_factory=list, description="主要薄弱环节与待改进项 (1~2条)")
    growth_suggestions: List[str] = Field(default_factory=list, description="针对性成长与能力进阶建议 (2~3条)")
    tags: List[str] = Field(default_factory=list, description="人才个性化核心特征标签 (如 破局引领者, 严谨推演者)")


class ParticipantReport(BaseModel):
    """单个候选人的完整可视化报告契约数据"""
    speaker_id: str = Field(..., description="候选人编号 (如 speaker_2)")
    display_name: str = Field(default="", description="展示姓名 (如 候选人2 / 张三)")
    total_score: float = Field(default=0.0, description="综合总分 (0~100)")
    overall_rank: int = Field(default=1, description="全场综合排名")
    participant_total: int = Field(default=12, description="全场总参评人数")
    radar_data: List[RadarPoint] = Field(default_factory=list, description="16 维雷达图序列")
    dimension_details: Dict[str, DimensionScore] = Field(default_factory=dict, description="16 维细分得分与佐证")
    diagnosis: ComprehensiveDiagnosis = Field(default_factory=ComprehensiveDiagnosis, description="综合诊断建议")
    matched_nodes_count: int = Field(default=0, description="命中专家图谱核心节点数")
    speaking_word_count: int = Field(default=0, description="全场发言总字数")
    speaking_turns: int = Field(default=0, description="发言总轮次数")


class CollaborationNetworkDTO(BaseModel):
    """供前端 ECharts 拓扑图直接渲染的网络数据"""
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="节点列表 [{id, name, symbolSize, ...}]")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="连线列表 [{source, target, value, lineStyle, ...}]")


class TeamOverview(BaseModel):
    """讨论全组宏观大盘数据"""
    participant_count: int = Field(default=0, description="参评候选人数量")
    average_total_score: float = Field(default=0.0, description="全场平均总分")
    average_scores: Dict[str, float] = Field(default_factory=dict, description="各维度团队均分")
    collaboration_network: CollaborationNetworkDTO = Field(default_factory=CollaborationNetworkDTO, description="团队互动拓扑网络")


class ReportData(BaseModel):
    """前后端统一的完整报告契约 (自包含闭环 JSON)"""
    task_id: str = Field(..., description="评测任务唯一 ID")
    project_id: str = Field(default="default_project", description="所属项目 ID")
    group_id: str = Field(..., description="讨论小组 ID")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="报告生成时间")
    corpus_title: str = Field(default="无领导小组讨论案例", description="案例题目")
    team_overview: TeamOverview = Field(default_factory=TeamOverview, description="团队宏观大盘数据")
    participants: List[ParticipantReport] = Field(default_factory=list, description="所有候选人报告列表")

    def get_participant(self, speaker_id: str) -> Optional[ParticipantReport]:
        for p in self.participants:
            if p.speaker_id == speaker_id:
                return p
        return None

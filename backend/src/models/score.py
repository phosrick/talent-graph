# -*- coding: utf-8 -*-
"""
Domain models for 16-Dimension Scoring System, Matrices, and Quant Evaluation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DimensionDefinition(BaseModel):
    """维度元数据定义"""
    id: str = Field(..., description="维度唯一编号 (如 1_1, 5_3, 4)")
    name: str = Field(..., description="维度全名")
    category: str = Field(..., description="所属能力大类 (如 思维能力, 决策能力)")
    description: str = Field(default="", description="维度能力释义")
    default_weight: float = Field(default=1.0, description="综合总分计算权重")


# 系统标准 16 个能力维度定义（与老系统 100% 对齐）
ALL_DIMENSIONS: List[DimensionDefinition] = [
    DimensionDefinition(id="1_1", name="思维能力 - 逻辑思维", category="思维能力", description="演绎/归纳/溯因推理链条完整性"),
    DimensionDefinition(id="1_2", name="思维能力 - 抽象归纳", category="思维能力", description="概念事实归纳提炼与金字塔层级"),
    DimensionDefinition(id="1_3", name="思维能力 - 系统思维", category="思维能力", description="多变量关联子网与环路连通度"),
    DimensionDefinition(id="2_1", name="决策能力 - 目标识别与拆解", category="决策能力", description="效用目标到决策方案的层级拆解"),
    DimensionDefinition(id="2_2", name="决策能力 - 信息收集与整合", category="决策能力", description="事实信息的引用跨度与整合度"),
    DimensionDefinition(id="2_3", name="决策能力 - 方案影响预判", category="决策能力", description="方案推演对下游影响的广度与深度"),
    DimensionDefinition(id="3_1", name="创新能力 - 创造性", category="创新能力", description="提出超出标准答案的高价值新颖视角"),
    DimensionDefinition(id="3_2", name="创新能力 - 联想能力", category="创新能力", description="跨领域远距离知识节点之间的关联"),
    DimensionDefinition(id="3_3", name="创新能力 - 方案可行性评估", category="创新能力", description="方案在现实资源与环境约束下的可行性"),
    DimensionDefinition(id="4",   name="思维激发能力", category="思维激发", description="发言对后续讨论节点的启发带动时序效应"),
    DimensionDefinition(id="5_1", name="冲突协调能力 - 开放性", category="冲突协调", description="对他人观点的倾听、接纳与正面承接"),
    DimensionDefinition(id="5_2", name="冲突协调能力 - 冲突消解", category="冲突协调", description="应对分歧、推动共识形成的状态转移能力"),
    DimensionDefinition(id="5_3", name="冲突协调能力 - 协作中心性", category="冲突协调", description="在互动网络中的入度中心性与协同枢纽地位"),
    DimensionDefinition(id="6_1", name="推进能力 - 聚焦性", category="推进能力", description="发言与讨论核心主题方向的贴合度"),
    DimensionDefinition(id="6_2", name="推进能力 - 时间把握能力", category="推进能力", description="讨论全流程的时间节奏分配与效率"),
    DimensionDefinition(id="6_3", name="推进能力 - 收敛性", category="推进能力", description="讨论后半程促进方案闭环与收敛的推动力"),
]

DIMENSION_MAP: Dict[str, DimensionDefinition] = {d.id: d for d in ALL_DIMENSIONS}


class DimensionScore(BaseModel):
    """候选人单维度得分详情"""
    dimension_id: str = Field(..., description="维度 ID (如 1_1)")
    dimension_name: str = Field(..., description="维度名称")
    score: float = Field(..., description="标准百分制得分 (0.0 ~ 100.0)")
    raw_score: float = Field(default=0.0, description="原始数学图论计算指标值")
    rank: int = Field(default=1, description="全场该维度排名 (1 为最高)")
    key_points: List[str] = Field(default_factory=list, description="得分佐证要点或高光表现")
    advantages: List[str] = Field(default_factory=list, description="该子维度优势表现 (对齐 V5 refined_summary advantage)")
    to_improve: List[str] = Field(default_factory=list, description="该子维度待提升项 (对齐 V5 refined_summary toImprove)")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="量化图论细分特征指标")


class ParticipantScoreProfile(BaseModel):
    """单个候选人 16 维全量评分画像"""
    speaker_id: str = Field(..., description="候选人 ID (如 speaker_2)")
    display_name: str = Field(default="", description="候选人姓名或代号")
    total_score: float = Field(default=0.0, description="综合加权总分 (0.0 ~ 100.0)")
    overall_rank: int = Field(default=1, description="全场综合总分排名")
    dimension_scores: Dict[str, DimensionScore] = Field(default_factory=dict, description="16 维度分数字典")

    def get_score(self, dim_id: str) -> float:
        return self.dimension_scores[dim_id].score if dim_id in self.dimension_scores else 0.0


class ScoreMatrix(BaseModel):
    """全场所有候选人的评分矩阵与团队宏观基准"""
    group_id: str = Field(default="", description="讨论小组 ID")
    participants: Dict[str, ParticipantScoreProfile] = Field(default_factory=dict, description="候选人画像字典")
    dimension_averages: Dict[str, float] = Field(default_factory=dict, description="各维度团队平均分")
    dimension_max_scores: Dict[str, float] = Field(default_factory=dict, description="各维度全场最高分")
    dimension_min_scores: Dict[str, float] = Field(default_factory=dict, description="各维度全场最低分")
    team_average_total_score: float = Field(default=0.0, description="团队综合平均总分")

    def get_speaker_score(self, speaker_id: str, dim_id: str) -> float:
        if speaker_id in self.participants:
            return self.participants[speaker_id].get_score(dim_id)
        return 0.0

    def compute_aggregations(self) -> None:
        """重新计算全场均分、极值与排名"""
        if not self.participants:
            return

        dim_ids = [d.id for d in ALL_DIMENSIONS]
        for dim_id in dim_ids:
            scores = [p.get_score(dim_id) for p in self.participants.values()]
            if scores:
                self.dimension_averages[dim_id] = round(sum(scores) / len(scores), 1)
                self.dimension_max_scores[dim_id] = max(scores)
                self.dimension_min_scores[dim_id] = min(scores)

                # 计算单项排名 (降序，同分并列)
                sorted_spk = sorted(self.participants.keys(), key=lambda s: self.participants[s].get_score(dim_id), reverse=True)
                for rank_idx, spk in enumerate(sorted_spk, 1):
                    if dim_id in self.participants[spk].dimension_scores:
                        self.participants[spk].dimension_scores[dim_id].rank = rank_idx

        # 计算综合总分排名
        total_scores = [p.total_score for p in self.participants.values()]
        if total_scores:
            self.team_average_total_score = round(sum(total_scores) / len(total_scores), 1)
            sorted_total = sorted(self.participants.keys(), key=lambda s: self.participants[s].total_score, reverse=True)
            for rank_idx, spk in enumerate(sorted_total, 1):
                self.participants[spk].overall_rank = rank_idx

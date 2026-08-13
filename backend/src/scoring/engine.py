# -*- coding: utf-8 -*-
"""
16-Dimension Quant Scoring Engine.
Orchestrates all 16 pure-mathematical & graph-theoretical scorers in memory.
Ultra-fast execution (<10ms per candidate, <50ms for the entire team).
"""

import time
from typing import Dict, List, Optional
from configs.settings import Settings, get_settings
from src.core.logger import get_logger
from src.models.score import ALL_DIMENSIONS, DimensionScore, ParticipantScoreProfile, ScoreMatrix
from src.scoring.base import BaseDimensionScorer, ScoringContext
from src.scoring.dimensions.d1_thinking import AbstractionScorer, LogicalReasoningScorer, SystematicThinkingScorer
from src.scoring.dimensions.d2_execution import GoalDecompositionScorer, ImpactPredictionScorer, InfoCollectionScorer
from src.scoring.dimensions.d3_creativity import AssociationScorer, CreativityScorer, FeasibilityScorer
from src.scoring.dimensions.d4_influence import InfluenceScorer
from src.scoring.dimensions.d5_teamwork import CollaborationCentralityScorer, ConflictResolutionScorer, OpennessScorer
from src.scoring.dimensions.d6_efficiency import ConvergenceScorer, FocusScorer, TimingControlScorer

logger = get_logger("scoring_engine")


class ScoringEngine:
    """16 维纯量化图论评分引擎"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.scorers: List[BaseDimensionScorer] = [
            # 维度一：思维能力
            LogicalReasoningScorer(),
            AbstractionScorer(),
            SystematicThinkingScorer(),
            # 维度二：决策能力
            GoalDecompositionScorer(),
            InfoCollectionScorer(),
            ImpactPredictionScorer(),
            # 维度三：创新能力
            CreativityScorer(),
            AssociationScorer(),
            FeasibilityScorer(),
            # 维度四：思维激发
            InfluenceScorer(),
            # 维度五：冲突协调
            OpennessScorer(),
            ConflictResolutionScorer(),
            CollaborationCentralityScorer(),
            # 维度六：推进能力
            FocusScorer(),
            TimingControlScorer(),
            ConvergenceScorer(),
        ]
        self.scorer_map = {s.SCORER_ID: s for s in self.scorers}

    def evaluate_all(self, context: ScoringContext) -> ScoreMatrix:
        """
        在纯内存中执行全量 16 个评分器，生成标准评分矩阵与团队大盘
        """
        start_time = time.perf_counter()
        logger.info(f"📊 [SCORING ENGINE] Starting 16-dimension evaluation for {len(context.candidate_ids)} candidates")

        candidate_ids = context.candidate_ids
        matrix = ScoreMatrix(group_id=context.group_id)

        # 初始化候选人画像
        for spk_id in candidate_ids:
            spk = context.get_speaker(spk_id)
            display_name = spk.display_name if spk else spk_id
            matrix.participants[spk_id] = ParticipantScoreProfile(
                speaker_id=spk_id,
                display_name=display_name,
                dimension_scores={},
            )

        # 逐个维度运行打分器 (每个维度内部对全场候选人计算并归一化)
        for scorer in self.scorers:
            dim_scores: Dict[str, DimensionScore] = scorer.evaluate_all(context)
            for spk_id, d_score in dim_scores.items():
                if spk_id in matrix.participants:
                    matrix.participants[spk_id].dimension_scores[scorer.SCORER_ID] = d_score

        # 计算每个候选人的综合加权总分
        for spk_id, profile in matrix.participants.items():
            if profile.dimension_scores:
                scores_list = [ds.score for ds in profile.dimension_scores.values()]
                profile.total_score = round(sum(scores_list) / len(scores_list), 1)

        # 聚合计算全场均分、极值与排名
        matrix.compute_aggregations()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"✅ [SCORING ENGINE COMPLETE] 16 dimensions calculated in {elapsed_ms:.2f}ms in-memory!")

        return matrix

# -*- coding: utf-8 -*-
"""
Abstract Scoring Base Class & Context.
Enforces pure-memory, pure-mathematical & graph-theoretical evaluation (<10ms per scorer).
Zero LLM calls, zero disk I/O, zero matplotlib drawing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import numpy as np
from pydantic import BaseModel, Field
from src.core.embedding import EmbeddingEngine
from src.models.discussion import DiscussionTranscript, InteractionNetwork, Speaker
from src.models.graph import CandidateGraph, StandardGraph
from src.models.score import DimensionScore


class ScoringContext(BaseModel):
    """评分上下文容器 (纯内存共享)"""
    model_config = {"arbitrary_types_allowed": True}

    group_id: str
    corpus_id: str
    standard_graph: StandardGraph
    transcript: DiscussionTranscript
    candidate_graphs: Dict[str, CandidateGraph] = Field(default_factory=dict)
    interaction_network: InteractionNetwork
    embedding_engine: Optional[Any] = None

    def get_speaker(self, speaker_id: str) -> Optional[Speaker]:
        return self.transcript.speakers.get(speaker_id)

    def get_candidate_graph(self, speaker_id: str) -> Optional[CandidateGraph]:
        return self.candidate_graphs.get(speaker_id)

    @property
    def candidate_ids(self) -> List[str]:
        return [s.speaker_id for s in self.transcript.candidate_speakers]


def normalize_to_standard_score(
    raw_val: float,
    all_raw_vals: List[float],
    min_scale: float = 60.0,
    max_scale: float = 95.0,
    default_empty: float = 65.0,
) -> float:
    """
    将原始图论/量化指标归一化为百分制标准分 (60 ~ 95 分区间，符合心理学测验分布)
    """
    if not all_raw_vals:
        return default_empty

    min_v = min(all_raw_vals)
    max_v = max(all_raw_vals)

    if max_v <= min_v:
        # 全员相同原始分
        return 75.0 if raw_val > 0 else default_empty

    ratio = (raw_val - min_v) / (max_v - min_v)
    score = min_scale + ratio * (max_scale - min_scale)
    return round(float(score), 1)


class BaseDimensionScorer(ABC):
    """16 维纯数学图论评分器基类"""

    SCORER_ID: str = ""
    SCORER_NAME: str = ""

    @abstractmethod
    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        """
        计算单人的纯数学量化原始分、细分图论指标与高光得分点
        Returns:
            (raw_score, metrics_dict, key_points_list)
        """
        pass

    def evaluate_all(
        self,
        context: ScoringContext,
    ) -> Dict[str, DimensionScore]:
        """
        为全场所有候选人计算并归一化该维度的得分与排名
        """
        candidate_ids = context.candidate_ids
        raw_results: Dict[str, tuple[float, Dict[str, Any], List[str]]] = {}

        for spk_id in candidate_ids:
            raw_val, metrics, key_points = self.calculate_raw_metrics(spk_id, context)
            raw_results[spk_id] = (raw_val, metrics, key_points)

        all_raw_vals = [res[0] for res in raw_results.values()]

        scores_dict: Dict[str, DimensionScore] = {}
        for spk_id in candidate_ids:
            raw_val, metrics, key_points = raw_results[spk_id]
            norm_score = normalize_to_standard_score(raw_val, all_raw_vals)

            scores_dict[spk_id] = DimensionScore(
                dimension_id=self.SCORER_ID,
                dimension_name=self.SCORER_NAME,
                score=norm_score,
                raw_score=round(float(raw_val), 3),
                key_points=key_points,
                metrics=metrics,
            )

        return scores_dict

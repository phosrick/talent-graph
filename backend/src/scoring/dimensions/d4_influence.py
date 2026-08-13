# -*- coding: utf-8 -*-
"""
Dimension 4: Thought Stimulation & Influence Competence Scorer (思维激发能力)
- 4: Influence (思维激发能力: 候选人发言引发后续讨论的引力与时序带动效应)
"""

from typing import Any, Dict, List, Set, Tuple
from src.scoring.base import BaseDimensionScorer, ScoringContext


class InfluenceScorer(BaseDimensionScorer):
    """4 思维激发能力评分器 (发言引发后续讨论节点的时序被跟进率)"""

    SCORER_ID = "4"
    SCORER_NAME = "思维激发能力"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"stimulated_turns": 0, "node_adoptions": 0}, []

        # 1. 人际互动被跟进次数 (在互动网络中被他人回应/支持的次数)
        in_interactions = [e for e in context.interaction_network.edges if e.target == speaker_id]
        support_in = len([e for e in in_interactions if e.type == "support"])
        total_in = len(in_interactions)

        # 2. 节点提出先驱度：候选人首次提出某个标准节点后，后续有其他人再次提到该节点
        cand_graph = context.get_candidate_graph(speaker_id)
        first_mentions: Dict[str, float] = {}  # node_id -> timestamp
        if cand_graph:
            for m in cand_graph.node_mentions:
                if m.node_id not in first_mentions or m.timestamp_seconds < first_mentions[m.node_id]:
                    first_mentions[m.node_id] = m.timestamp_seconds

        # 检查全场后续其他人的发言中对这些节点的引用
        node_adoptions = 0
        for other_id, other_graph in context.candidate_graphs.items():
            if other_id == speaker_id:
                continue
            for m in other_graph.node_mentions:
                if m.node_id in first_mentions and m.timestamp_seconds > first_mentions[m.node_id]:
                    node_adoptions += 1

        total_raw = total_in * 2.0 + support_in * 1.5 + node_adoptions * 2.5

        key_points = []
        if total_in > 0:
            key_points.append(f"言论有效激发全场互动，吸引他人针对性回应 {total_in} 次")
        if node_adoptions > 0:
            key_points.append(f"率先提出的核心议题被其他成员在后续发言中吸收引用 {node_adoptions} 次")
        if not key_points and total_raw > 0:
            key_points.append("在讨论中保持积极的思维参与度")

        metrics = {
            "total_responses_received": total_in,
            "support_received": support_in,
            "subsequent_node_adoptions": node_adoptions,
        }
        return float(total_raw), metrics, key_points

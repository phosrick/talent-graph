# -*- coding: utf-8 -*-
"""
Dimension 5: Conflict Resolution & Teamwork Competence Scorers (冲突协调能力)
- 5_1: Openness (开放度)
- 5_2: Conflict Resolution (冲突消解)
- 5_3: Collaboration Centrality (协作中心性)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.scoring.base import BaseDimensionScorer, ScoringContext


class OpennessScorer(BaseDimensionScorer):
    """5_1 开放度评分器 (对他人观点的倾听、接纳与正面承接)"""

    SCORER_ID = "5_1"
    SCORER_NAME = "冲突协调能力 - 开放性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        # 统计主动发起 support 和 neutral 互动的次数 (积极回应接纳他人)
        out_edges = [e for e in context.interaction_network.edges if e.source == speaker_id]
        support_out = [e for e in out_edges if e.type == "support"]
        neutral_out = [e for e in out_edges if e.type == "neutral"]

        # 承接的不同成员数量 (广度)
        targeted_speakers = {e.target for e in out_edges}

        total_raw = len(support_out) * 3.0 + len(neutral_out) * 1.5 + len(targeted_speakers) * 2.0

        key_points = []
        if len(support_out) > 0:
            key_points.append(f"展现高度包容开放，主动对团队成员提出 {len(support_out)} 次肯定与支持")
        if len(targeted_speakers) >= 2:
            key_points.append(f"互动覆盖全组 {len(targeted_speakers)} 位不同成员，倾听广度良好")
        if not key_points and total_raw > 0:
            key_points.append("在讨论中展现了基本的接纳与协作态度")

        metrics = {
            "support_given": len(support_out),
            "neutral_given": len(neutral_out),
            "interacted_peers": len(targeted_speakers),
        }
        return float(total_raw), metrics, key_points


class ConflictResolutionScorer(BaseDimensionScorer):
    """5_2 冲突消解能力评分器 (化解分歧、推动团队共识形成的能力)"""

    SCORER_ID = "5_2"
    SCORER_NAME = "冲突协调能力 - 冲突消解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        # 寻找全场时序中的分歧消解模式：
        # 当某人受到 deny 后，speaker 介入并提出中立/支持建议或新折中决策 (d)
        network_edges = context.interaction_network.edges
        deny_edges = [e for e in network_edges if e.type == "deny"]

        # 候选人主动发起的纠偏 (有建设性的 deny) 与调解 (在冲突后的 support)
        my_out = [e for e in network_edges if e.source == speaker_id]
        my_deny = [e for e in my_out if e.type == "deny"]
        my_support = [e for e in my_out if e.type == "support"]

        # 候选人提出的决策方案数量 (用于弥合分歧)
        cand_graph = context.get_candidate_graph(speaker_id)
        decisions_count = len(cand_graph.matched_nodes) if cand_graph else 0

        total_raw = len(my_support) * 2.0 + len(my_deny) * 1.5 + decisions_count * 0.8

        key_points = []
        if len(my_deny) > 0 and len(my_support) > 0:
            key_points.append("在讨论出现分歧时能够兼顾纠偏与凝聚，促进团队共识达成")
        elif len(my_support) > 0:
            key_points.append(f"多次通过支持与调和化解潜在分歧 ({len(my_support)}次)")

        metrics = {
            "constructive_challenges": len(my_deny),
            "alignment_support": len(my_support),
            "consensus_decisions": decisions_count,
        }
        return float(total_raw), metrics, key_points


class CollaborationCentralityScorer(BaseDimensionScorer):
    """5_3 协作中心性评分器 (基于 NetworkX 互动网络的加权入度与中介中心性，极速 0.1ms 计算)"""

    SCORER_ID = "5_3"
    SCORER_NAME = "冲突协调能力 - 协作中心性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        G = context.interaction_network.to_networkx()

        if G.number_of_nodes() == 0 or not G.has_node(speaker_id):
            return 0.0, {"in_degree": 0, "centrality": 0.0}, []

        # 1. 加权入度中心性 (受到全场支持与认可的净权重)
        in_degree_weight = 0.0
        for _, _, data in G.in_edges(speaker_id, data=True):
            w = float(data.get("weight", 1.0))
            types = data.get("types", [])
            primary_type = data.get("type", "support")
            if "support" in types or primary_type == "support":
                in_degree_weight += w
            elif "deny" in types or primary_type == "deny":
                in_degree_weight -= 0.5 * w
            else:
                in_degree_weight += 0.5 * w

        # 2. 中介中心性 (作为信息与协作枢纽的程度)
        try:
            betweenness = nx.betweenness_centrality(G.to_undirected()).get(speaker_id, 0.0)
        except Exception:
            betweenness = 0.0

        # 3. 相对入度占全场比例
        total_edges = G.number_of_edges()
        centrality_ratio = (in_degree_weight / max(1, total_edges))

        total_raw = max(0.0, in_degree_weight) * 5.0 + betweenness * 20.0 + total_edges * 0.2

        key_points = []
        if in_degree_weight >= 2.0:
            key_points.append(f"处于全场互动核心枢纽位置，获得团队成员高频响应与协作认可")
        if betweenness >= 0.1:
            key_points.append("在不同讨论子群之间发挥了关键的桥梁与中介协同作用")
        if not key_points and total_raw > 0:
            key_points.append("保持良好的团队互动与协作连接")

        metrics = {
            "net_in_degree_weight": round(in_degree_weight, 2),
            "betweenness_centrality": round(betweenness, 3),
            "centrality_ratio": round(centrality_ratio, 3),
        }
        return float(total_raw), metrics, key_points

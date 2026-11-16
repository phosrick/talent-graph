# -*- coding: utf-8 -*-
"""
Dimension 5: Conflict Resolution & Teamwork Competence Scorers (冲突协调能力 - 对齐 V5 团队动力模型)
- 5_1: Openness (开放性: 积极倾听+认可他人+分析分歧+推动共识 四维计数与互动广度)
- 5_2: Conflict Resolution (冲突消解: 分歧发生后的时序介入调解、折中方案供给与观点收敛)
- 5_3: Collaboration Centrality (协作中心性: 对齐 V5 CollaborationNetwork 有向带符号中心性 + 中介中心性)
"""

from typing import Any, Dict, List
import networkx as nx
from src.scoring.base import BaseDimensionScorer, ScoringContext


class OpennessScorer(BaseDimensionScorer):
    """5_1 开放度评分器 (融合 V5 四维开放行为统计 + 互动网络正向承接广度)"""

    SCORER_ID = "5_1"
    SCORER_NAME = "冲突协调能力 - 开放性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        out_edges = [e for e in context.interaction_network.edges if e.source == speaker_id]
        support_out = [e for e in out_edges if e.type == "support"]
        neutral_out = [e for e in out_edges if e.type == "neutral"]
        targeted_speakers = {e.target for e in out_edges}

        cand_graph = context.get_candidate_graph(speaker_id)
        oc = (cand_graph.openness_counts if cand_graph else None) or {}

        listening = oc.get("listening", len(neutral_out))
        supporting = oc.get("supporting", len(support_out))
        analysis = oc.get("analysis", 0)
        pushing = oc.get("pushing", 0)
        v5_openness_sum = listening + supporting + analysis + pushing

        total_raw = (
            v5_openness_sum * 1.8
            + len(support_out) * 2.2
            + len(neutral_out) * 1.0
            + len(targeted_speakers) * 1.8
        )

        key_points = []
        if supporting > 0 or len(support_out) > 0:
            cnt = max(supporting, len(support_out))
            key_points.append(f"尊重并正向接纳他人合理见解 {cnt} 次，展现出良好的开放协作姿态")
        if len(targeted_speakers) >= 2:
            key_points.append(f"主动回应覆盖 {len(targeted_speakers)} 位不同成员，倾听与互动面广")
        elif v5_openness_sum > 0:
            key_points.append(f"累计展现 {v5_openness_sum} 次倾听反馈与共识推进动作")

        metrics = {
            "listening_count": listening,
            "supporting_count": supporting,
            "conflict_analysis_count": analysis,
            "consensus_pushing_count": pushing,
            "interacted_peers": len(targeted_speakers),
        }
        return float(total_raw), metrics, key_points


class ConflictResolutionScorer(BaseDimensionScorer):
    """5_2 冲突消解能力评分器 (检测分歧事件后的时序介入调和、折中方案输出与共识恢复)"""

    SCORER_ID = "5_2"
    SCORER_NAME = "冲突协调能力 - 冲突消解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        network_edges = context.interaction_network.edges
        deny_edges = [e for e in network_edges if e.type == "deny"]

        # 统计在场上发生分歧(deny)后的 1~3 个话轮内，该候选人主动介入调和或提出折中决策(d)的次数
        conflict_turn_indices = [e.utterance_index for e in deny_edges if e.utterance_index is not None]
        cand_graph = context.get_candidate_graph(speaker_id)
        speaker = context.get_speaker(speaker_id)

        post_conflict_mediations = 0
        if speaker and conflict_turn_indices:
            for u in speaker.utterances:
                if any(0 < (u.index - c_idx) <= 3 for c_idx in conflict_turn_indices):
                    post_conflict_mediations += 1

        my_out = [e for e in network_edges if e.source == speaker_id]
        my_deny = [e for e in my_out if e.type == "deny"]
        my_support = [e for e in my_out if e.type == "support"]

        decision_nodes = (
            [n for n in cand_graph.matched_nodes if n.type == "d"] + [n for n in cand_graph.novel_nodes if n.type == "d"]
            if cand_graph
            else []
        )
        conflict_analysis_cnt = (cand_graph.openness_counts.get("analysis", 0) if cand_graph and cand_graph.openness_counts else 0)

        total_raw = (
            post_conflict_mediations * 3.0
            + conflict_analysis_cnt * 2.0
            + len(my_support) * 1.8
            + min(2, len(my_deny)) * 1.2
            + len(decision_nodes) * 1.0
        )

        key_points = []
        if post_conflict_mediations > 0:
            key_points.append(f"在团队出现观点交锋后及时介入调和 {post_conflict_mediations} 次，缓和场上对立")
        if len(my_support) > 0 and len(decision_nodes) > 0:
            key_points.append(f"结合 {len(decision_nodes)} 项建设性折中方案凝聚成员共识")
        elif len(my_support) > 0:
            key_points.append(f"通过 {len(my_support)} 次正向整合回应化解潜在分歧")

        metrics = {
            "post_conflict_mediations": post_conflict_mediations,
            "conflict_analysis_count": conflict_analysis_cnt,
            "alignment_support": len(my_support),
            "constructive_challenges": len(my_deny),
            "consensus_decisions": len(decision_nodes),
        }
        return float(total_raw), metrics, key_points


class CollaborationCentralityScorer(BaseDimensionScorer):
    """5_3 协作中心性评分器 (100% 对齐 V5 CollaborationNetwork 带符号入度中心性 + 中介中心性)"""

    SCORER_ID = "5_3"
    SCORER_NAME = "冲突协调能力 - 协作中心性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        G = context.interaction_network.to_networkx()

        if G.number_of_nodes() == 0 or not G.has_node(speaker_id):
            return 0.0, {"net_in_degree_weight": 0.0, "v5_collab_centrality": 0.0, "betweenness_centrality": 0.0}, []

        # 1. 对齐 V5 CollaborationNetwork.calculate_collab_centrality():
        # support 为 +weight，deny 为 -weight，neutral 为 +0.5*weight，除以总边数
        in_signed_weight = 0.0
        for _, _, data in G.in_edges(speaker_id, data=True):
            in_signed_weight += float(data.get("signed_weight", data.get("weight", 1.0)))

        total_edges = G.number_of_edges()
        v5_centrality = in_signed_weight / total_edges if total_edges > 0 else 0.0

        # 2. 中介中心性 (衡量跨子群协同枢纽地位)
        try:
            betweenness = nx.betweenness_centrality(G.to_undirected()).get(speaker_id, 0.0)
        except Exception:
            betweenness = 0.0

        # 保持对负向净中心性的区分度（避免全被截断成 0）
        total_raw = v5_centrality * 15.0 + in_signed_weight * 3.0 + betweenness * 15.0

        key_points = []
        if in_signed_weight >= 1.5:
            key_points.append(f"处于团队协作网络核心位置（协作中心度 {v5_centrality:.2f}），获高频正向响应")
        if betweenness >= 0.1:
            key_points.append("在多位成员间起到关键的沟通桥梁与意见串联作用")
        if not key_points and in_signed_weight > 0:
            key_points.append("在团队协作网络中获得正向支持与认可")

        metrics = {
            "v5_collab_centrality": round(v5_centrality, 3),
            "net_in_degree_weight": round(in_signed_weight, 2),
            "betweenness_centrality": round(betweenness, 3),
        }
        return float(total_raw), metrics, key_points

# -*- coding: utf-8 -*-
"""
Dimension 2: Execution & Decision-Making Competence Scorers (决策能力)
- 2_1: Goal Identification & Decomposition (目标识别与拆解)
- 2_2: Information Collection & Integration (信息收集与整合)
- 2_3: Impact Prediction (方案影响预判)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.models.graph import Node
from src.scoring.base import BaseDimensionScorer, ScoringContext


class GoalDecompositionScorer(BaseDimensionScorer):
    """2_1 目标识别与拆解能力评分器 (效用目标u到决策方案d的路径与覆盖)"""

    SCORER_ID = "2_1"
    SCORER_NAME = "决策能力 - 目标识别与拆解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"utility_coverage": 0.0, "decision_alignment": 0}, []

        matched_u = [n for n in cand_graph.matched_nodes if n.type == "u"]
        matched_d = [n for n in cand_graph.matched_nodes if n.type == "d"] + cand_graph.novel_nodes
        total_std_u = len([n for n in context.standard_graph.nodes if n.type == "u"])

        u_coverage = len(matched_u) / max(1, total_std_u)

        # 检查候选人个人图谱中 d -> u 的连通边
        G = cand_graph.to_networkx()
        d_to_u_paths = 0
        u_ids = {n.id for n in matched_u}
        d_ids = {n.id for n in matched_d}

        for d_id in d_ids:
            for u_id in u_ids:
                if d_id != u_id and G.has_node(d_id) and G.has_node(u_id):
                    if nx.has_path(G, d_id, u_id):
                        d_to_u_paths += 1

        total_raw = len(matched_u) * 3.0 + d_to_u_paths * 2.0 + len(matched_d) * 1.0

        key_points = []
        if len(matched_u) > 0:
            key_points.append(f"精准识别定位了 {len(matched_u)} 项核心商业与效用目标")
        if d_to_u_paths > 0:
            key_points.append(f"成功将目标拆解映射至 {d_to_u_paths} 条落地方案路径")

        metrics = {
            "utility_count": len(matched_u),
            "utility_coverage": round(u_coverage, 2),
            "decision_count": len(matched_d),
            "goal_to_action_paths": d_to_u_paths,
        }
        return float(total_raw), metrics, key_points


class InfoCollectionScorer(BaseDimensionScorer):
    """2_2 信息收集与整合能力评分器 (事实信息节点c的引用跨度与频次)"""

    SCORER_ID = "2_2"
    SCORER_NAME = "决策能力 - 信息收集与整合"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"concept_count": 0, "time_span": 0.0}, []

        c_nodes = [n for n in cand_graph.matched_nodes if n.type == "c"]
        c_mentions = [m for m in cand_graph.node_mentions if context.standard_graph.get_node(m.node_id) and context.standard_graph.get_node(m.node_id).type == "c"]

        # 时间跨度 (信息引用的持续度)
        if c_mentions:
            timestamps = [m.timestamp_seconds for m in c_mentions]
            time_span = max(timestamps) - min(timestamps)
        else:
            time_span = 0.0

        total_raw = len(c_nodes) * 2.5 + len(c_mentions) * 1.0 + (time_span / 60.0) * 0.5

        key_points = []
        if len(c_nodes) >= 2:
            key_points.append(f"全场详实引用了 {len(c_nodes)} 项客观背景与数据事实")
        if len(c_mentions) >= 4:
            key_points.append(f"信息引用频率达 {len(c_mentions)} 次，论证依据扎实")

        metrics = {
            "concept_nodes": len(c_nodes),
            "total_mentions": len(c_mentions),
            "time_span_seconds": round(time_span, 1),
        }
        return float(total_raw), metrics, key_points


class ImpactPredictionScorer(BaseDimensionScorer):
    """2_3 方案影响预判能力评分器 (决策节点下游影响与出度加权)"""

    SCORER_ID = "2_3"
    SCORER_NAME = "决策能力 - 方案影响预判"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"impact_edges": 0, "downstream_outdegree": 0}, []

        G = cand_graph.to_networkx()
        decision_nodes = [n.id for n in cand_graph.matched_nodes if n.type == "d"] + [n.id for n in cand_graph.novel_nodes]

        # 计算决策节点的出度加权总和 (引出的下游影响数量)
        total_outdegree = 0
        for d_id in decision_nodes:
            if G.has_node(d_id):
                total_outdegree += G.out_degree(d_id)

        # 候选人是否提及决策对概念或效用的反馈边
        impact_edges = len(cand_graph.verified_arcs)
        total_raw = total_outdegree * 2.0 + impact_edges * 1.2 + len(decision_nodes) * 1.0

        key_points = []
        if total_outdegree > 0:
            key_points.append(f"针对提出的行动举措，有效推演了 {total_outdegree} 项下游连锁影响")
        if not key_points and total_raw > 0:
            key_points.append("具备基础的方案影响预判意识")

        metrics = {
            "decision_count": len(decision_nodes),
            "outdegree_sum": total_outdegree,
            "impact_edges": impact_edges,
        }
        return float(total_raw), metrics, key_points

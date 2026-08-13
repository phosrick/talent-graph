# -*- coding: utf-8 -*-
"""
Dimension 3: Creativity & Innovation Competence Scorers (创新能力)
- 3_1: Creativity (创造性)
- 3_2: Association & Remote Thinking (联想能力)
- 3_3: Feasibility Evaluation (方案可行性)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.scoring.base import BaseDimensionScorer, ScoringContext


class CreativityScorer(BaseDimensionScorer):
    """3_1 创造性评分器 (提出标准库之外的高价值发散创新节点)"""

    SCORER_ID = "3_1"
    SCORER_NAME = "创新能力 - 创造性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"novel_nodes": 0}, []

        novel_count = len(cand_graph.novel_nodes)
        unique_matched = len(cand_graph.matched_nodes)

        total_raw = novel_count * 5.0 + unique_matched * 0.5
        key_points = []
        if novel_count > 0:
            names = [n.name for n in cand_graph.novel_nodes[:2]]
            key_points.append(f"提出了 {novel_count} 个超出标准参考库的创新视角 (如: {', '.join(names)})")
        if not key_points and unique_matched > 0:
            key_points.append("在既有标准框架下稳定输出观点")

        metrics = {
            "novel_nodes_count": novel_count,
            "novel_nodes_list": [n.name for n in cand_graph.novel_nodes],
        }
        return float(total_raw), metrics, key_points


class AssociationScorer(BaseDimensionScorer):
    """3_2 联想能力评分器 (跨领域/远距离节点的最短跳数与跨度)"""

    SCORER_ID = "3_2"
    SCORER_NAME = "创新能力 - 联想能力"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"remote_paths": 0, "avg_hop": 0.0}, []

        std_G = context.standard_graph.to_networkx()
        matched_ids = list(cand_graph.matched_node_ids)

        # 计算候选人同时提及的节点在标准图谱中的最短距离
        hop_distances = []
        for i in range(len(matched_ids)):
            for j in range(i + 1, len(matched_ids)):
                u, v = matched_ids[i], matched_ids[j]
                if std_G.has_node(u) and std_G.has_node(v):
                    try:
                        # 计算无向或有向距离
                        length = nx.shortest_path_length(std_G.to_undirected(), u, v)
                        if length >= 2:
                            hop_distances.append(length)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

        remote_connections = len([d for d in hop_distances if d >= 3])
        avg_hop = sum(hop_distances) / max(1, len(hop_distances))
        total_raw = sum(hop_distances) * 1.5 + remote_connections * 3.0

        key_points = []
        if remote_connections > 0:
            key_points.append(f"实现了 {remote_connections} 次跨越 3 跳以上远距离知识节点的联想碰撞")
        elif hop_distances:
            key_points.append(f"平均联想跨度达 {avg_hop:.1f} 跳拓扑距离")

        metrics = {
            "total_associations": len(hop_distances),
            "remote_connections_ge_3": remote_connections,
            "avg_hop_distance": round(avg_hop, 2),
        }
        return float(total_raw), metrics, key_points


class FeasibilityScorer(BaseDimensionScorer):
    """3_3 方案可行性评估能力评分器 (资源限制、风险控制、落地性约束拓扑满足)"""

    SCORER_ID = "3_3"
    SCORER_NAME = "创新能力 - 方案可行性评估"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"feasible_decisions": 0, "constraint_ratio": 0.0}, []

        # 获取候选人提出的决策节点
        decisions = [n for n in cand_graph.matched_nodes if n.type == "d"] + cand_graph.novel_nodes
        concepts = [n for n in cand_graph.matched_nodes if n.type == "c"]

        # 可行性量化：决策节点是否具有概念事实支撑 (info: c -> d)
        G = cand_graph.to_networkx()
        grounded_decisions = 0

        for d in decisions:
            if G.has_node(d.id):
                in_degree = G.in_degree(d.id)
                if in_degree > 0:
                    grounded_decisions += 1

        feasibility_ratio = grounded_decisions / max(1, len(decisions))
        total_raw = grounded_decisions * 3.0 + len(concepts) * 1.5 + len(decisions) * 1.0

        key_points = []
        if grounded_decisions > 0:
            key_points.append(f"提出的方案中，有 {grounded_decisions} 项具备清晰的现实依据与支撑条件")
        if feasibility_ratio >= 0.5:
            key_points.append("方案可行性结构占比良好，兼顾理想与现实落地")

        metrics = {
            "decision_count": len(decisions),
            "grounded_decisions": grounded_decisions,
            "feasibility_ratio": round(feasibility_ratio, 2),
        }
        return float(total_raw), metrics, key_points

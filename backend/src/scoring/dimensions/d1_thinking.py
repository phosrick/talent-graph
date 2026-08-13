# -*- coding: utf-8 -*-
"""
Dimension 1: Thinking Competence Scorers (思维能力)
- 1_1: Logical Reasoning (逻辑思维)
- 1_2: Abstraction & Induction (抽象归纳)
- 1_3: Systematic Thinking (系统思维)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.models.graph import Node
from src.scoring.base import BaseDimensionScorer, ScoringContext


class LogicalReasoningScorer(BaseDimensionScorer):
    """1_1 逻辑思维能力评分器 (演绎/归纳/溯因推理链)"""

    SCORER_ID = "1_1"
    SCORER_NAME = "思维能力 - 逻辑思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"deduction": 0, "induction": 0, "abduction": 0, "total_chains": 0}, []

        # 获取候选人提及的节点类型字典
        node_types: Dict[str, str] = {}
        for n in cand_graph.matched_nodes + cand_graph.novel_nodes:
            node_types[n.id] = n.type

        verified_edges = cand_graph.verified_arcs
        G = cand_graph.to_networkx()

        # 计算演绎推理 (c -> d, d -> u), 归纳推理 (c -> c, d -> d), 溯因推理 (u <- d, d <- c 反向)
        deduction_count = 0
        induction_count = 0
        abduction_count = 0

        for arc in verified_edges:
            src_type = node_types.get(arc.source, "c")
            tgt_type = node_types.get(arc.target, "c")

            if (src_type == "c" and tgt_type == "d") or (src_type == "d" and tgt_type == "u"):
                deduction_count += 1
            elif src_type == tgt_type:
                induction_count += 1
            else:
                abduction_count += 1

        # 查找长因果链 (路径长度 >= 2)
        chain_count = 0
        for src in G.nodes():
            for tgt in G.nodes():
                if src != tgt and nx.has_path(G, src, tgt):
                    try:
                        paths = list(nx.all_simple_paths(G, src, tgt, cutoff=3))
                        for p in paths:
                            if len(p) >= 3:
                                chain_count += 1
                    except Exception:
                        pass

        total_raw = deduction_count * 1.5 + induction_count * 1.2 + abduction_count * 1.0 + chain_count * 0.8
        key_points = []
        if deduction_count > 0:
            key_points.append(f"成功完成 {deduction_count} 次由事实到决策/效用的演绎逻辑推导")
        if chain_count > 0:
            key_points.append(f"构建出 {chain_count} 条多跳深度因果推理链")
        if not key_points and total_raw > 0:
            key_points.append("展现了一定的逻辑论证与因果关联能力")

        metrics = {
            "deduction_count": deduction_count,
            "induction_count": induction_count,
            "abduction_count": abduction_count,
            "chain_count": chain_count,
            "verified_arcs": len(verified_edges),
        }
        return float(total_raw), metrics, key_points


class AbstractionScorer(BaseDimensionScorer):
    """1_2 抽象归纳能力评分器 (跨事实抽象归纳与概念提炼)"""

    SCORER_ID = "1_2"
    SCORER_NAME = "思维能力 - 抽象归纳"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"concept_coverage": 0, "abstraction_pairs": 0}, []

        matched_nodes = cand_graph.matched_nodes
        std_nodes = context.standard_graph.nodes

        concept_nodes = [n for n in matched_nodes if n.type == "c"]
        decision_nodes = [n for n in matched_nodes if n.type == "d"]
        utility_nodes = [n for n in matched_nodes if n.type == "u"]

        # 概念覆盖率
        total_std_concepts = len([n for n in std_nodes if n.type == "c"])
        concept_coverage = len(concept_nodes) / max(1, total_std_concepts)

        # 检查同一发言段中提及多个不同具体事实的抽象整合次数
        u_map: Dict[int, List[Node]] = {}
        for m in cand_graph.node_mentions:
            node = context.standard_graph.get_node(m.node_id)
            if node:
                u_map.setdefault(m.utterance_index, []).append(node)

        abstraction_pairs = 0
        for u_idx, nodes_in_u in u_map.items():
            if len(nodes_in_u) >= 2:
                # 出现不同节点
                unique_ids = {n.id for n in nodes_in_u}
                if len(unique_ids) >= 2:
                    abstraction_pairs += len(unique_ids) - 1

        total_raw = len(concept_nodes) * 2.0 + abstraction_pairs * 1.5 + len(utility_nodes) * 1.0
        key_points = []
        if len(concept_nodes) >= 3:
            key_points.append(f"提炼整合了 {len(concept_nodes)} 个核心概念事实与背景维度")
        if abstraction_pairs > 0:
            key_points.append(f"在单次发言中实现 {abstraction_pairs} 次多源信息的抽象综合")

        metrics = {
            "concept_nodes_count": len(concept_nodes),
            "concept_coverage": round(concept_coverage, 2),
            "abstraction_pairs": abstraction_pairs,
        }
        return float(total_raw), metrics, key_points


class SystematicThinkingScorer(BaseDimensionScorer):
    """1_3 系统思维能力评分器 (多变量弱连通子网与金字塔层级)"""

    SCORER_ID = "1_3"
    SCORER_NAME = "思维能力 - 系统思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"level1": 0, "level2": 0, "level3": 0, "graph_density": 0.0}, []

        # 金字塔三层划分：
        # 一级：效用/宏观目标 (u)
        # 二级：决策/中观方案 (d)
        # 三级：概念/微观事实 (c)
        level1_u = len([n for n in cand_graph.matched_nodes if n.type == "u"])
        level2_d = len([n for n in cand_graph.matched_nodes if n.type == "d"] + cand_graph.novel_nodes)
        level3_c = len([n for n in cand_graph.matched_nodes if n.type == "c"])

        # 核心系统思维公式：(1 + 一级宏观) × 二级中观 × 三级微观
        pyramid_score = (1 + level1_u) * max(1, level2_d) * max(1, level3_c)

        # 拓扑连通度与子图密度
        G = cand_graph.to_networkx()
        density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        wcc = nx.number_weakly_connected_components(G) if G.number_of_nodes() > 0 else 0

        total_raw = float(pyramid_score) + density * 10.0 + len(cand_graph.verified_arcs) * 1.5

        key_points = []
        if level1_u > 0 and level2_d > 0 and level3_c > 0:
            key_points.append("兼备宏观目标(U)、中观方案(D)与微观事实(C)的完整金字塔系统架构")
        if len(cand_graph.verified_arcs) >= 3:
            key_points.append(f"形成了包含 {len(cand_graph.verified_arcs)} 条因果链路的复杂系统网络")

        metrics = {
            "macro_utility_u": level1_u,
            "meso_decision_d": level2_d,
            "micro_concept_c": level3_c,
            "pyramid_product": pyramid_score,
            "graph_density": round(density, 3),
            "connected_components": wcc,
        }
        return float(total_raw), metrics, key_points

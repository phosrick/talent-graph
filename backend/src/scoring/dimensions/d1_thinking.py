# -*- coding: utf-8 -*-
"""
Dimension 1: Thinking Competence Scorers (思维能力 - 100% 对齐 V5 影响图推理与金字塔理论)
- 1_1: Logical Reasoning (逻辑思维: 概念↔事实 演绎/归纳/溯因推理连通簇 + 决策链)
- 1_2: Abstraction & Induction (抽象归纳: 同属上位概念且在因果图中 d-分离的跨事实归纳对)
- 1_3: Systematic Thinking (系统思维: 宏观主题[一级] × 抽象概念[二级] × 具体事实[三级] 金字塔乘积)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.scoring.base import BaseDimensionScorer, ScoringContext


def _count_connected_components(edges: List[Tuple[str, str]]) -> int:
    """无向连通分量计数（对齐 V5 find_connected_components_from_edges，识别独立推理簇数量）"""
    if not edges:
        return 0
    adj: Dict[str, Set[str]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited: Set[str] = set()
    comp = 0
    for node in adj:
        if node in visited:
            continue
        comp += 1
        stack = [node]
        visited.add(node)
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
    return comp


class LogicalReasoningScorer(BaseDimensionScorer):
    """1_1 逻辑思维能力评分器 (还原 V5 概念↔事实 演绎/归纳/溯因三模式连通簇算法)"""

    SCORER_ID = "1_1"
    SCORER_NAME = "思维能力 - 逻辑思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"deduction_count": 0, "induction_count": 0, "abduction_count": 0, "chain_count": 0}, []

        std_graph = context.standard_graph
        has_levels = std_graph.has_explicit_levels

        if has_levels:
            concept_ids = std_graph.concept_node_ids
            fact_ids = std_graph.fact_node_ids
        else:
            # 兼容未标注层级的极简测试图谱：以 u/d 作为上位概念层、c 作为基础事实层
            concept_ids = {n.id for n in std_graph.nodes if n.type in ("u", "d")}
            fact_ids = {n.id for n in std_graph.nodes if n.type == "c"}

        # 候选人经验证的个人边集合（同时参考标准图谱边用于时序发言段内的隐含推理模式识别）
        verified_pairs = {(a.source, a.target) for a in cand_graph.verified_arcs}
        std_pairs = {(a.source, a.target) for a in std_graph.arcs}
        active_edges = verified_pairs if verified_pairs else (
            {(u, v) for (u, v) in std_pairs if u in cand_graph.matched_node_ids and v in cand_graph.matched_node_ids}
        )

        # 按候选人发言顺序组织的节点序列（对齐 V5 load_speaker_reasoning_data）
        if cand_graph.node_mentions:
            seq = [m.node_id for m in sorted(cand_graph.node_mentions, key=lambda x: (x.utterance_index, x.timestamp_seconds))]
        else:
            seq = [n.id for n in cand_graph.matched_nodes]

        ded_edges: List[Tuple[str, str]] = []
        ind_edges: List[Tuple[str, str]] = []
        abd_edges: List[Tuple[str, str]] = []

        # 1. 基于发言时序的 V5 三大推理模式检测
        for i in range(len(seq)):
            u_id = seq[i]
            for j in range(i + 1, len(seq)):
                v_id = seq[j]
                if u_id == v_id:
                    continue
                # Pattern 1 演绎推理: 先提概念 C，后推导事实 F (或事实链 F->F)
                if (u_id, v_id) in active_edges:
                    if (u_id in concept_ids and v_id in fact_ids) or (u_id in fact_ids and v_id in fact_ids):
                        ded_edges.append((u_id, v_id))
                    # Pattern 3 归纳推理: 先提事实 F，后归纳概念 C (或概念递进 C->C)
                    elif (u_id in fact_ids and v_id in concept_ids) or (u_id in concept_ids and v_id in concept_ids):
                        ind_edges.append((u_id, v_id))
                # Pattern 4 溯因推理: 先提结果事实 v_id (出现在前即 u_id)，后追溯原因 (v_id -> u_id)
                if (v_id, u_id) in active_edges and u_id in fact_ids:
                    abd_edges.append((v_id, u_id))

        # 补充已验证决策推导边 (c -> d, d -> u)
        node_types = {n.id: n.type for n in cand_graph.matched_nodes + cand_graph.novel_nodes}
        for arc in cand_graph.verified_arcs:
            st = node_types.get(arc.source, "c")
            tt = node_types.get(arc.target, "c")
            pair = (arc.source, arc.target)
            if (st == "c" and tt == "d") or (st == "d" and tt == "u"):
                if pair not in ded_edges:
                    ded_edges.append(pair)
            elif st == tt and pair not in ind_edges:
                ind_edges.append(pair)

        deduction_count = _count_connected_components(ded_edges)
        induction_count = _count_connected_components(ind_edges)
        abduction_count = _count_connected_components(abd_edges)

        # 多跳深度因果链计数 (路径长度 >= 2 跳，即 >= 3 个节点)
        G = cand_graph.to_networkx()
        chain_count = 0
        for src in G.nodes():
            for tgt in G.nodes():
                if src != tgt and nx.has_path(G, src, tgt):
                    try:
                        for p in nx.all_simple_paths(G, src, tgt, cutoff=3):
                            if len(p) >= 3:
                                chain_count += 1
                    except Exception:
                        pass

        # 对齐 V5 主公式 (count_deduction + count_induction + count_abduction) 并融合多跳链区分度
        total_raw = (
            deduction_count * 1.5
            + induction_count * 1.5
            + abduction_count * 1.5
            + chain_count * 0.8
            + len(cand_graph.verified_arcs) * 0.3
        )

        key_points = []
        if deduction_count > 0:
            key_points.append(f"构建了 {deduction_count} 个演绎推理簇（由概念/依据推演具体结论）")
        if induction_count > 0:
            key_points.append(f"完成了 {induction_count} 个归纳推理簇（由具体事实提炼上位判断）")
        if abduction_count > 0:
            key_points.append(f"体现了 {abduction_count} 次由果溯因的逆向溯因分析")
        if chain_count > 0 and len(key_points) < 2:
            key_points.append(f"形成 {chain_count} 条多跳深度因果链条")
        if not key_points and total_raw > 0:
            key_points.append("展现了一定的逻辑论证与因果关联能力")

        metrics = {
            "deduction_count": deduction_count,
            "induction_count": induction_count,
            "abduction_count": abduction_count,
            "chain_count": chain_count,
            "verified_arcs": len(cand_graph.verified_arcs),
        }
        return float(total_raw), metrics, key_points


class AbstractionScorer(BaseDimensionScorer):
    """1_2 抽象归纳能力评分器 (还原 V5: 同上位概念且在因果图中不直接相连的具体节点对归纳次数)"""

    SCORER_ID = "1_2"
    SCORER_NAME = "思维能力 - 抽象归纳"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"abstraction_count": 0, "concept_nodes_count": 0, "concept_coverage": 0.0}, []

        std_graph = context.standard_graph
        causal_undirected: Set[Tuple[str, str]] = set()
        for a in std_graph.arcs:
            if a.type == "causal":
                causal_undirected.add((a.source, a.target))
                causal_undirected.add((a.target, a.source))

        # 1. V5 核心指标：先后提及两个同属一个上位 parent_concept、彼此不同且无直接因果边的具体节点对
        matched_nodes = cand_graph.matched_nodes
        v5_abstraction_pairs: List[Tuple[str, str]] = []
        n_len = len(matched_nodes)
        for i in range(n_len):
            ni = matched_nodes[i]
            ci = ni.parent_concept
            if not ci or ni.name == ci:
                continue
            for j in range(i + 1, n_len):
                nj = matched_nodes[j]
                cj = nj.parent_concept
                if ci != cj or ni.id == nj.id or nj.name == cj:
                    continue
                # 空条件集下的简化 d-分离（无直接因果边）
                if (ni.id, nj.id) not in causal_undirected:
                    v5_abstraction_pairs.append((ni.name, nj.name))

        # 2. 单次发言内跨多个非相邻事实节点的综合归纳（当极简图谱未填 parent_concept 时提供无缝信号）
        u_map: Dict[int, List[str]] = {}
        for m in cand_graph.node_mentions:
            u_map.setdefault(m.utterance_index, []).append(m.node_id)

        co_utterance_pairs = 0
        for _, nids in u_map.items():
            uniq = list(dict.fromkeys(nids))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    if (uniq[i], uniq[j]) not in causal_undirected:
                        co_utterance_pairs += 1

        if std_graph.has_explicit_levels:
            concept_nodes = [n for n in matched_nodes if n.level in ("一级", "二级")]
            total_std_concepts = max(1, len(std_graph.concept_node_ids))
        else:
            concept_nodes = [n for n in matched_nodes if n.type == "c"]
            total_std_concepts = max(1, len([n for n in std_graph.nodes if n.type == "c"]))

        concept_coverage = len(concept_nodes) / total_std_concepts
        abstraction_count = len(v5_abstraction_pairs) + co_utterance_pairs

        total_raw = len(v5_abstraction_pairs) * 3.0 + co_utterance_pairs * 1.5 + len(concept_nodes) * 2.0

        key_points = []
        if v5_abstraction_pairs:
            p0 = v5_abstraction_pairs[0]
            key_points.append(
                f"完成 {len(v5_abstraction_pairs)} 组跨独立事实的同概念抽象提炼（如「{p0[0][:10]}」与「{p0[1][:10]}」）"
            )
        if len(concept_nodes) >= 2:
            key_points.append(f"准确把握 {len(concept_nodes)} 项核心抽象概念，概念覆盖率 {concept_coverage:.0%}")
        elif co_utterance_pairs > 0:
            key_points.append(f"在单轮发言中完成 {co_utterance_pairs} 次多源独立信息的综合归纳")

        metrics = {
            "abstraction_count": abstraction_count,
            "v5_concept_dsep_pairs": len(v5_abstraction_pairs),
            "concept_nodes_count": len(concept_nodes),
            "concept_coverage": round(concept_coverage, 2),
        }
        return float(total_raw), metrics, key_points


class SystematicThinkingScorer(BaseDimensionScorer):
    """1_3 系统思维能力评分器 (还原 V5 公式: (1 + 一级宏观主题) × 二级抽象概念 × 三级具体事实)"""

    SCORER_ID = "1_3"
    SCORER_NAME = "思维能力 - 系统思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"level1_count": 0, "level2_count": 0, "level3_count": 0, "pyramid_product": 0}, []

        std_graph = context.standard_graph
        if std_graph.has_explicit_levels:
            level1 = len([n for n in cand_graph.matched_nodes if n.level == "一级"])
            level2 = len([n for n in cand_graph.matched_nodes if n.level == "二级"])
            level3 = len([n for n in cand_graph.matched_nodes if n.level == "三级"]) + len(cand_graph.novel_nodes)
        else:
            # 极简无层级测试夹具回退：按 u(宏观目标) / d(中观举措) / c(微观事实) 统计
            level1 = len([n for n in cand_graph.matched_nodes if n.type == "u"])
            level2 = len([n for n in cand_graph.matched_nodes if n.type == "d"]) + len(cand_graph.novel_nodes)
            level3 = len([n for n in cand_graph.matched_nodes if n.type == "c"])

        # V5 原始公式：(1 + 一级) × 二级 × 三级；辅以节点覆盖与边连通平滑项区分缺层但仍有贡献的候选人
        v5_pyramid = (1 + level1) * level2 * level3
        G = cand_graph.to_networkx()
        density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        base_coverage = (level1 * 1.5 + level2 * 1.2 + level3 * 0.8) + len(cand_graph.verified_arcs) * 1.2

        total_raw = float(v5_pyramid) * 2.0 + base_coverage + density * 5.0

        key_points = []
        if level1 > 0 and level2 > 0 and level3 > 0:
            key_points.append(
                f"兼备一级主题({level1})、二级概念({level2})与三级事实({level3})的完整三层金字塔思维结构"
            )
        elif level2 > 0 and level3 > 0:
            key_points.append(f"较好串联了中观抽象概念({level2})与微观具体事实({level3})")
        if len(cand_graph.verified_arcs) >= 3:
            key_points.append(f"构建了含 {len(cand_graph.verified_arcs)} 条验证关系的系统关联子网")

        metrics = {
            "level1_count": level1,
            "level2_count": level2,
            "level3_count": level3,
            "pyramid_product": v5_pyramid,
            "graph_density": round(density, 3),
        }
        return float(total_raw), metrics, key_points

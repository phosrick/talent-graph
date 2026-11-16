# -*- coding: utf-8 -*-
"""
Dimension 2: Execution & Decision-Making Competence Scorers (决策能力 - 100% 对齐 V5 决策分析理论)
- 2_1: Goal Identification & Decomposition (目标识别与拆解: (手段目标+最终目标)×(1+d->手段->最终拆解路径))
- 2_2: Information Collection & Integration (信息收集与整合: 已知/未知信息广度 + 向量空间覆盖分散度)
- 2_3: Impact Prediction (方案影响预判: 信息支撑决策 c->d 与决策反作用于环境 d->c 的双向影响预判)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import numpy as np
from src.scoring.base import BaseDimensionScorer, ScoringContext


class GoalDecompositionScorer(BaseDimensionScorer):
    """2_1 目标识别与拆解能力评分器 (还原 V5: 区分最终目标/手段目标及 d -> 手段目标 -> 最终目标 拆解子图)"""

    SCORER_ID = "2_1"
    SCORER_NAME = "决策能力 - 目标识别与拆解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"final_count": 0, "instrumental_count": 0, "decompose_count": 0}, []

        std_graph = context.standard_graph
        final_ids = set(std_graph.final_target_ids)
        inst_ids = set(std_graph.instrumental_target_ids)

        # 若测试图谱未标注 final/instrumental，按 u->u 拓扑自动推断（有出边到其他 u 的为手段目标，其余为最终目标）
        std_u_ids = {n.id for n in std_graph.nodes if n.type == "u"}
        if std_u_ids and not (final_ids and inst_ids):
            u_with_out_u = {
                a.source for a in std_graph.arcs if a.source in std_u_ids and a.target in std_u_ids
            }
            if u_with_out_u:
                inst_ids = u_with_out_u
                final_ids = std_u_ids - u_with_out_u
            else:
                final_ids = std_u_ids

        # 构建 V5 目标拆解三元组: d -> u_inst -> u_final (且不存在 d -> u_final 直接跳边)
        d_ids_std = {n.id for n in std_graph.nodes if n.type == "d"}
        d_u_edges = {(a.source, a.target) for a in std_graph.arcs if a.source in d_ids_std and a.target in std_u_ids}
        u_u_edges = {(a.source, a.target) for a in std_graph.arcs if a.source in std_u_ids and a.target in std_u_ids}

        v5_subgraphs: List[Tuple[str, str, str]] = []
        for d_id, u_mid in d_u_edges:
            if u_mid in inst_ids:
                for u1, u_fin in u_u_edges:
                    if u1 == u_mid and u_fin in final_ids and (d_id, u_fin) not in d_u_edges:
                        v5_subgraphs.append((d_id, u_mid, u_fin))

        matched_set = cand_graph.matched_node_ids
        final_matched = [nid for nid in matched_set if nid in final_ids]
        inst_matched = [nid for nid in matched_set if nid in inst_ids]

        # V5 子图命中次数 + 个人已验证图谱中的 d -> u 拆解路径
        decompose_count = sum(1 for (d_id, _, u_fin) in v5_subgraphs if d_id in matched_set and u_fin in matched_set)

        cand_G = cand_graph.to_networkx()
        cand_d_ids = {n.id for n in cand_graph.matched_nodes if n.type == "d"} | {n.id for n in cand_graph.novel_nodes if n.type == "d"}
        cand_u_ids = {n.id for n in cand_graph.matched_nodes if n.type == "u"}
        verified_d_u_paths = sum(
            1
            for d_id in cand_d_ids
            for u_id in cand_u_ids
            if cand_G.has_node(d_id) and cand_G.has_node(u_id) and nx.has_path(cand_G, d_id, u_id)
        )

        effective_decompose = decompose_count + verified_d_u_paths
        # 对齐 V5 核心公式：(手段目标数 + 最终目标数) × (1 + 拆解次数)
        v5_score = (len(inst_matched) + len(final_matched)) * (1 + effective_decompose)
        total_raw = float(v5_score) * 2.0 + len(cand_d_ids) * 0.8

        key_points = []
        if final_matched or inst_matched:
            key_points.append(
                f"识别了 {len(final_matched)} 项最终目标与 {len(inst_matched)} 项阶段性手段目标"
            )
        if effective_decompose > 0:
            key_points.append(f"完成了 {effective_decompose} 条「方案举措 → 手段目标 → 最终目标」拆解链路")

        metrics = {
            "final_count": len(final_matched),
            "instrumental_count": len(inst_matched),
            "decompose_count": effective_decompose,
            "decision_count": len(cand_d_ids),
        }
        return float(total_raw), metrics, key_points


class InfoCollectionScorer(BaseDimensionScorer):
    """2_2 信息收集与整合能力评分器 (结合已知/未知信息挖掘深度与向量语义覆盖广度)"""

    SCORER_ID = "2_2"
    SCORER_NAME = "决策能力 - 信息收集与整合"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"concept_nodes": 0, "unknown_explored": 0, "semantic_spread": 0.0}, []

        c_nodes = [n for n in cand_graph.matched_nodes if n.type == "c"]
        unknown_c = [n for n in c_nodes if n.sub_type == "unknown"]
        c_ids = {n.id for n in c_nodes}
        c_mentions = [m for m in cand_graph.node_mentions if m.node_id in c_ids]

        # 计算所收集信息节点在语义向量空间中的分散广度（对应 V5 信息节点 Embedding/PCA 覆盖度思想）
        semantic_spread = 0.0
        emb = context.embedding_engine
        if emb is not None and len(c_nodes) >= 2:
            try:
                vecs = emb.embed_texts([n.name for n in c_nodes])
                sim_mat = emb.batch_cosine_similarity(vecs, vecs)
                n_c = len(c_nodes)
                upper_dists = [1.0 - float(sim_mat[i, j]) for i in range(n_c) for j in range(i + 1, n_c)]
                if upper_dists:
                    semantic_spread = sum(upper_dists) / len(upper_dists)
            except Exception:
                semantic_spread = 0.0

        time_span = 0.0
        if c_mentions:
            ts_list = [m.timestamp_seconds for m in c_mentions]
            time_span = max(ts_list) - min(ts_list)

        total_raw = (
            len(c_nodes) * 2.5
            + len(unknown_c) * 1.5
            + len(c_mentions) * 0.8
            + semantic_spread * 4.0
            + min(3.0, (time_span / 120.0))
        )

        key_points = []
        if len(c_nodes) >= 2:
            key_points.append(f"整合引用了 {len(c_nodes)} 项客观事实与背景依据（含 {len(unknown_c)} 项深层待查证信息）")
        if semantic_spread >= 0.35 and len(c_nodes) >= 3:
            key_points.append(f"信息采集覆盖多维视角，语义广度指数达 {semantic_spread:.2f}")
        elif len(c_mentions) >= 3:
            key_points.append(f"全场高频调用事实论据 {len(c_mentions)} 次支撑观点")

        metrics = {
            "concept_nodes": len(c_nodes),
            "unknown_explored": len(unknown_c),
            "total_mentions": len(c_mentions),
            "semantic_spread": round(semantic_spread, 3),
            "time_span_seconds": round(time_span, 1),
        }
        return float(total_raw), metrics, key_points


class ImpactPredictionScorer(BaseDimensionScorer):
    """2_3 方案影响预判能力评分器 (还原 V5: 信息支撑决策 info_arcs 与决策改变环境状态 d->c 耦合子图)"""

    SCORER_ID = "2_3"
    SCORER_NAME = "决策能力 - 方案影响预判"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"impact_count": 0, "feedback_loops": 0, "outdegree_sum": 0}, []

        std_graph = context.standard_graph
        node_type_map = {n.id: n.type for n in std_graph.nodes}

        # V5 方案影响预判子图:
        # 1) info_arcs 中的 (c -> d)
        # 2) causal_arcs 中的反向环境影响 (d -> c)
        info_cd_pairs: Set[Tuple[str, str]] = set()
        causal_dc_pairs: Set[Tuple[str, str]] = set()
        for a in std_graph.arcs:
            st = node_type_map.get(a.source, "")
            tt = node_type_map.get(a.target, "")
            if a.type == "info" or (st == "c" and tt == "d"):
                info_cd_pairs.add((a.source, a.target))
            if st == "d" and tt == "c":
                causal_dc_pairs.add((a.source, a.target))

        matched_ids = cand_graph.matched_node_ids
        v5_impact_hits = sum(1 for (c_id, d_id) in info_cd_pairs if c_id in matched_ids and d_id in matched_ids)
        feedback_hits = sum(1 for (d_id, c_id) in causal_dc_pairs if d_id in matched_ids and c_id in matched_ids)

        # 结合个人已验证边中决策节点的下游推演数 (d -> c/d/u)
        G = cand_graph.to_networkx()
        decision_ids = [n.id for n in cand_graph.matched_nodes if n.type == "d"] + [
            n.id for n in cand_graph.novel_nodes if n.type == "d"
        ]
        outdegree_sum = sum(G.out_degree(d_id) for d_id in decision_ids if G.has_node(d_id))

        total_raw = v5_impact_hits * 2.0 + feedback_hits * 3.0 + outdegree_sum * 2.0 + len(decision_ids) * 0.5

        key_points = []
        if outdegree_sum > 0 or feedback_hits > 0:
            key_points.append(
                f"针对决策方案推演了 {outdegree_sum + feedback_hits} 项下游连锁效应与环境反馈"
            )
        if v5_impact_hits > 0:
            key_points.append(f"准确建立 {v5_impact_hits} 组「客观条件 ↔ 方案选择」双向影响预判")
        if not key_points and total_raw > 0:
            key_points.append("具备基础的方案落地影响评估意识")

        metrics = {
            "impact_count": v5_impact_hits,
            "feedback_loops": feedback_hits,
            "outdegree_sum": outdegree_sum,
            "decision_count": len(decision_ids),
        }
        return float(total_raw), metrics, key_points

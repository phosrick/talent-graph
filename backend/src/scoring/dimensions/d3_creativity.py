# -*- coding: utf-8 -*-
"""
Dimension 3: Creativity & Innovation Competence Scorers (创新能力 - 100% 对齐 V5 创新与可行性理论)
- 3_1: Creativity (创造性: 决策节点数 × 节点序列向量跳动总距离 + 标准库外新颖观点)
- 3_2: Association (联想能力: 信息发散联想 V 型子图 + 多方案收敛联想子图 + 远距跳数)
- 3_3: Feasibility Evaluation (方案可行性: 资源限制/风险识别/执行门槛/动态性 四维评估 + 事实依据闭环)
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
from src.models.score import DimensionScore
from src.scoring.base import BaseDimensionScorer, ScoringContext, normalize_to_standard_score
from src.core.embedding import get_embedding_engine


class CreativityScorer(BaseDimensionScorer):
    """3_1 创造性评分器 (还原 V5 核心公式: 决策方案数 × 发言节点序列语义跳跃总距离 + 创新发散节点)"""

    SCORER_ID = "3_1"
    SCORER_NAME = "创新能力 - 创造性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"decision_count": 0, "semantic_distance": 0.0, "novel_nodes_count": 0}, []

        std_decisions = [n for n in cand_graph.matched_nodes if n.type == "d"]
        novel_nodes = cand_graph.novel_nodes
        decision_count = len(std_decisions) + len([n for n in novel_nodes if n.type == "d"])

        # 按发言时序还原候选人的节点名称序列（对齐 V5 load_speaker_node_sequence）
        seq_names: List[str] = []
        if cand_graph.node_mentions:
            for m in sorted(cand_graph.node_mentions, key=lambda x: (x.timestamp_seconds, x.utterance_index)):
                nd = context.standard_graph.get_node(m.node_id)
                if nd:
                    seq_names.append(nd.name)
        else:
            seq_names = [n.name for n in cand_graph.matched_nodes]
        seq_names.extend([n.name for n in novel_nodes])

        emb = context.embedding_engine or get_embedding_engine()
        semantic_distance = 0.0
        if emb is not None and len(seq_names) >= 2:
            try:
                semantic_distance = emb.compute_sequence_jump_distance(seq_names)
            except Exception:
                semantic_distance = 0.0

        # 对齐 V5 公式: decision_count * semantic_distance (若仅单点则保留 decision_count)，并叠加 novel_nodes 贡献
        v5_core = (decision_count * semantic_distance) if semantic_distance > 0 else float(decision_count)
        total_raw = v5_core + len(novel_nodes) * 3.5

        key_points = []
        if len(novel_nodes) > 0:
            names = [n.name for n in novel_nodes[:2]]
            key_points.append(f"突破既有参考框架，提出 {len(novel_nodes)} 项新颖方案（如：{', '.join(names)}）")
        if decision_count > 0 and semantic_distance > 0:
            key_points.append(
                f"贡献 {decision_count} 项决策举措，思维跨度活跃（语义跳跃距离 {semantic_distance:.2f}）"
            )
        elif not key_points and total_raw > 0:
            key_points.append("围绕核心问题提出了具有建设性的行动方案")

        metrics = {
            "decision_count": decision_count,
            "semantic_distance": round(semantic_distance, 3),
            "novel_nodes_count": len(novel_nodes),
            "novel_nodes_list": [n.name for n in novel_nodes],
        }
        return float(total_raw), metrics, key_points


class AssociationScorer(BaseDimensionScorer):
    """3_2 联想能力评分器 (还原 V5: 信息发散联想子图 + 多方案平行联想子图 + 全图预计算远距跳数)"""

    SCORER_ID = "3_2"
    SCORER_NAME = "创新能力 - 联想能力"

    def _precompute_graph_structures(self, context: ScoringContext):
        std_graph = context.standard_graph
        type_map = {n.id: n.type for n in std_graph.nodes}

        c_c_set: Set[Tuple[str, str]] = set()
        d_c_list: List[Tuple[str, str]] = []
        d_d_set: Set[Tuple[str, str]] = set()
        d_u_list: List[Tuple[str, str]] = []
        c_c_list: List[Tuple[str, str]] = []

        for a in std_graph.arcs:
            st, tt = type_map.get(a.source, ""), type_map.get(a.target, "")
            if st == "c" and tt == "c":
                c_c_set.add((a.source, a.target))
                c_c_set.add((a.target, a.source))
                c_c_list.append((a.source, a.target))
            elif st == "d" and tt == "c":
                d_c_list.append((a.source, a.target))
            elif st == "d" and tt == "d":
                d_d_set.add((a.source, a.target))
                d_d_set.add((a.target, a.source))
            elif st == "d" and tt == "u":
                d_u_list.append((a.source, a.target))

        # 1. V5 信息联想子图: 同一源头(d 或 root_c) -> (c1, c2) 且 c1, c2 无直接边
        info_by_src: Dict[str, List[str]] = {}
        for src, c_id in d_c_list + c_c_list:
            info_by_src.setdefault(src, []).append(c_id)

        info_triplets: Set[Tuple[str, str, str]] = set()
        for src, c_targets in info_by_src.items():
            uniq_c = sorted(set(c_targets))
            for i in range(len(uniq_c)):
                for j in range(i + 1, len(uniq_c)):
                    if (uniq_c[i], uniq_c[j]) not in c_c_set:
                        info_triplets.add((src, uniq_c[i], uniq_c[j]))

        # 2. V5 方案联想子图: 多个互不直接相连的决策 (d1, d2) 共同指向同一目标/机会
        sol_by_tgt: Dict[str, List[str]] = {}
        for d_id, tgt in d_c_list + d_u_list:
            sol_by_tgt.setdefault(tgt, []).append(d_id)

        sol_triplets: Set[Tuple[str, str, str]] = set()
        for tgt, d_sources in sol_by_tgt.items():
            uniq_d = sorted(set(d_sources))
            for i in range(len(uniq_d)):
                for j in range(i + 1, len(uniq_d)):
                    if (uniq_d[i], uniq_d[j]) not in d_d_set:
                        sol_triplets.add((tgt, uniq_d[i], uniq_d[j]))

        # 3. 一次性预计算无向图全对最短路径（避免对 12 人重复建图和 BFS）
        undir_G = std_graph.to_networkx().to_undirected()
        all_spl = dict(nx.all_pairs_shortest_path_length(undir_G))
        return info_triplets, sol_triplets, all_spl

    def evaluate_all(self, context: ScoringContext) -> Dict[str, DimensionScore]:
        info_triplets, sol_triplets, all_spl = self._precompute_graph_structures(context)
        candidate_ids = context.candidate_ids
        raw_results: Dict[str, tuple[float, Dict[str, Any], List[str]]] = {}

        for spk_id in candidate_ids:
            raw_results[spk_id] = self._calc_with_cache(
                spk_id, context, info_triplets, sol_triplets, all_spl
            )

        all_raw_vals = [r[0] for r in raw_results.values()]
        out: Dict[str, DimensionScore] = {}
        for spk_id in candidate_ids:
            raw_val, metrics, key_points = raw_results[spk_id]
            out[spk_id] = DimensionScore(
                dimension_id=self.SCORER_ID,
                dimension_name=self.SCORER_NAME,
                score=normalize_to_standard_score(raw_val, all_raw_vals),
                raw_score=round(float(raw_val), 3),
                key_points=key_points,
                metrics=metrics,
            )
        return out

    def _calc_with_cache(
        self,
        speaker_id: str,
        context: ScoringContext,
        info_triplets: Set[Tuple[str, str, str]],
        sol_triplets: Set[Tuple[str, str, str]],
        all_spl: Dict[str, Dict[str, int]],
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"info_associations": 0, "solution_associations": 0, "remote_connections_ge_3": 0}, []

        m_ids = cand_graph.matched_node_ids
        info_hits = sum(1 for (s, c1, c2) in info_triplets if s in m_ids and c1 in m_ids and c2 in m_ids)
        sol_hits = sum(1 for (t, d1, d2) in sol_triplets if t in m_ids and d1 in m_ids and d2 in m_ids)

        m_list = list(m_ids)
        hop_distances: List[int] = []
        for i in range(len(m_list)):
            dist_map = all_spl.get(m_list[i], {})
            for j in range(i + 1, len(m_list)):
                d = dist_map.get(m_list[j])
                if d is not None and d >= 2:
                    hop_distances.append(d)

        remote_ge_3 = sum(1 for d in hop_distances if d >= 3)
        avg_hop = sum(hop_distances) / max(1, len(hop_distances))
        total_raw = info_hits * 3.0 + sol_hits * 3.0 + sum(hop_distances) * 0.8 + remote_ge_3 * 2.0

        key_points = []
        if info_hits > 0 or sol_hits > 0:
            key_points.append(f"触发 {info_hits} 次跨信息发散联想与 {sol_hits} 次多方案替代联想")
        if remote_ge_3 > 0:
            key_points.append(f"实现 {remote_ge_3} 次跨越 3 跳以上远距知识节点的联想迁移")
        elif hop_distances:
            key_points.append(f"平均知识关联跨度达 {avg_hop:.1f} 跳")

        metrics = {
            "info_associations": info_hits,
            "solution_associations": sol_hits,
            "remote_connections_ge_3": remote_ge_3,
            "avg_hop_distance": round(avg_hop, 2),
        }
        return float(total_raw), metrics, key_points

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        info_t, sol_t, spl = self._precompute_graph_structures(context)
        return self._calc_with_cache(speaker_id, context, info_t, sol_t, spl)


class FeasibilityScorer(BaseDimensionScorer):
    """3_3 方案可行性评估能力评分器 (融合 V5 四维可行性评分 + 客观信息支撑闭环率)"""

    SCORER_ID = "3_3"
    SCORER_NAME = "创新能力 - 方案可行性评估"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"resource_awareness": 0.0, "risk_sensitivity": 0.0, "execution_feasibility": 0.0, "adaptability": 0.0}, []

        decisions = [n for n in cand_graph.matched_nodes if n.type == "d"] + [
            n for n in cand_graph.novel_nodes if n.type == "d"
        ]
        concepts = [n for n in cand_graph.matched_nodes if n.type == "c"]

        G = cand_graph.to_networkx()
        grounded_decisions = sum(1 for d in decisions if G.has_node(d.id) and G.in_degree(d.id) > 0)
        feasibility_ratio = grounded_decisions / max(1, len(decisions)) if decisions else 0.0

        # 读取 NodeMatcher 并发阶段同步评出的 V5 四维可行性分 (0~10)，若离线测试未生成则基于图结构闭环映射
        fr = cand_graph.feasibility_ratings or {}
        if fr:
            res_aw = float(fr.get("resource_awareness", 0.0))
            risk_s = float(fr.get("risk_sensitivity", 0.0))
            exec_f = float(fr.get("execution_feasibility", 0.0))
            adapt = float(fr.get("adaptability", 0.0))
            v5_four_dim_avg = (res_aw + risk_s + exec_f + adapt) / 4.0
        else:
            res_aw = min(10.0, len(concepts) * 2.5)
            risk_s = min(10.0, grounded_decisions * 3.5)
            exec_f = min(10.0, feasibility_ratio * 8.0 + (2.0 if decisions else 0.0))
            adapt = min(10.0, len(cand_graph.verified_arcs) * 2.0)
            v5_four_dim_avg = (res_aw + risk_s + exec_f + adapt) / 4.0 if (decisions or concepts) else 0.0

        total_raw = v5_four_dim_avg * 2.0 + grounded_decisions * 2.0 + feasibility_ratio * 3.0

        key_points = []
        if fr and v5_four_dim_avg >= 6.0:
            key_points.append(
                f"四维可行性均分 {v5_four_dim_avg:.1f}/10（资源意识 {res_aw:.0f} / 风险管控 {risk_s:.0f} / 落地门槛 {exec_f:.0f} / 动态性 {adapt:.0f}）"
            )
        if grounded_decisions > 0:
            key_points.append(f"所提方案中有 {grounded_decisions} 项具备明确的客观事实依据支撑")
        elif decisions:
            key_points.append("提出了具体行动方案，建议进一步补充资源与风险边界论证")

        metrics = {
            "resource_awareness": round(res_aw, 2),
            "risk_sensitivity": round(risk_s, 2),
            "execution_feasibility": round(exec_f, 2),
            "adaptability": round(adapt, 2),
            "grounded_decisions": grounded_decisions,
            "feasibility_ratio": round(feasibility_ratio, 2),
        }
        return float(total_raw), metrics, key_points

# -*- coding: utf-8 -*-
"""
Dimension 6: Process Driving & Efficiency Competence Scorers (讨论控场与推进能力 - 100% 对齐 V5 算法)
- 6_1: Focus (聚焦性: 逐轮发言句向量与讨论核心主题向量的平均贴合度 1 - mean_distance)
- 6_2: Timing Control (时间把握能力: 显式控时/议程推进表达计数 + 前中后三阶段参与均衡度)
- 6_3: Convergence (收敛性: 还原 V5 发言前后团队距离最终决议向量的缩减量 Δ = pre_dist - post_dist + 冲刺期共识贡献)
"""

import re
from typing import Any, Dict, List
import numpy as np
from src.core.embedding import get_embedding_engine
from src.models.score import DimensionScore
from src.scoring.base import BaseDimensionScorer, ScoringContext, normalize_to_standard_score

_TIMING_REGEX = re.compile(
    r"(\d+\s*分钟|还有.{0,4}时间|注意.{0,3}时间|控制.{0,3}时间|把控.{0,3}时间|时间有限|抓紧时间|"
    r"进入下一|下一阶段|先讨论.{1,8}再|总结一下|梳理一下框架|按顺序发言)"
)


class FocusScorer(BaseDimensionScorer):
    """6_1 聚焦性评分器 (还原 V5 s6_1_focus: 逐次发言与核心议题向量的平均余弦贴合度)"""

    SCORER_ID = "6_1"
    SCORER_NAME = "推进能力 - 聚焦性"

    def _resolve_topic_text(self, context: ScoringContext) -> str:
        if context.transcript.core_topic and context.transcript.core_topic.strip():
            return context.transcript.core_topic.strip()
        top_nodes = [
            n.name for n in context.standard_graph.nodes if n.level in ("一级", "二级") or n.type == "u"
        ][:6] or [n.name for n in context.standard_graph.nodes[:6]]
        return f"{context.standard_graph.title} {' '.join(top_nodes)}".strip()

    def evaluate_all(self, context: ScoringContext) -> Dict[str, DimensionScore]:
        emb = context.embedding_engine or get_embedding_engine()
        topic_text = self._resolve_topic_text(context)

        # 一次性批量编码核心主题与全员所有发言（极速矩阵批处理）
        all_texts = [topic_text]
        for cid in context.candidate_ids:
            spk = context.get_speaker(cid)
            if spk:
                all_texts.extend(u.text[:500] for u in spk.utterances if u.text.strip())
        if emb is not None and all_texts:
            emb.embed_texts(list(dict.fromkeys(all_texts)))

        return super().evaluate_all(context)

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"topic_cosine_similarity": 0.0, "mean_topic_distance": 1.0, "matched_ratio": 0.0}, []

        emb = context.embedding_engine or get_embedding_engine()
        topic_text = self._resolve_topic_text(context)
        u_texts = [u.text[:500] for u in speaker.utterances if u.text.strip()]

        mean_sim = 0.5
        if emb is not None and u_texts and topic_text:
            try:
                topic_vec = emb.embed_text(topic_text)
                u_vecs = emb.embed_texts(u_texts)
                sims = emb.compute_similarity(topic_vec, u_vecs)
                if len(sims) > 0:
                    mean_sim = float(np.mean(sims))
            except Exception:
                mean_sim = 0.5

        mean_dist = 1.0 - mean_sim
        cand_graph = context.get_candidate_graph(speaker_id)
        matched_ratio = (
            len(cand_graph.matched_nodes) / max(1, len(context.standard_graph.nodes))
            if cand_graph
            else 0.0
        )

        # 对齐 V5: 距主题平均距离越小(mean_sim越高)，聚焦度越高
        total_raw = mean_sim * 10.0 + matched_ratio * 5.0

        key_points = []
        if mean_sim >= 0.45:
            key_points.append(f"各轮发言紧密围绕核心议题展开，平均语义聚焦度达 {mean_sim:.2f}")
        if matched_ratio >= 0.25:
            key_points.append(f"覆盖题库 {matched_ratio:.0%} 核心节点，未出现偏题发散")
        elif not key_points and total_raw > 0:
            key_points.append("发言基本贴合案例讨论主线")

        metrics = {
            "topic_cosine_similarity": round(mean_sim, 3),
            "mean_topic_distance": round(mean_dist, 3),
            "matched_ratio": round(matched_ratio, 2),
        }
        return float(total_raw), metrics, key_points


class TimingControlScorer(BaseDimensionScorer):
    """6_2 时间把握能力评分器 (融合 V5 显式控时/议程引导次数 + 前中后三阶段均衡度)"""

    SCORER_ID = "6_2"
    SCORER_NAME = "推进能力 - 时间把握能力"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"timing_control_count": 0, "phases_covered": 0, "turns_count": 0}, []

        # 1. V5 核心信号：显式提醒时间、阶段切分与议程控制次数（融合 LLM 嗅探与关键词检出）
        regex_hits = sum(len(_TIMING_REGEX.findall(u.text)) for u in speaker.utterances)
        cand_graph = context.get_candidate_graph(speaker_id)
        llm_timing = cand_graph.timing_control_count if (cand_graph and cand_graph.timing_control_count is not None) else None
        timing_count = max(regex_hits, llm_timing) if llm_timing is not None else regex_hits

        # 2. 全场前、中、后三阶段参与分布与单轮发言适度性
        total_duration = context.transcript.total_duration_seconds or 1.0
        spk_duration = speaker.total_duration_seconds
        turns = speaker.turn_count

        t_third = total_duration / 3.0
        phase1 = sum(1 for u in speaker.utterances if u.start_seconds < t_third)
        phase2 = sum(1 for u in speaker.utterances if t_third <= u.start_seconds < 2 * t_third)
        phase3 = sum(1 for u in speaker.utterances if u.start_seconds >= 2 * t_third)
        phases_covered = sum(1 for p in (phase1, phase2, phase3) if p > 0)

        avg_turn_sec = spk_duration / max(1, turns)
        turn_quality = 1.3 if 15.0 <= avg_turn_sec <= 100.0 else (0.6 if avg_turn_sec > 180.0 else 1.0)

        total_raw = timing_count * 4.0 + phases_covered * 2.5 + min(8, turns) * 0.6 * turn_quality

        key_points = []
        if timing_count > 0:
            key_points.append(f"主动发起 {timing_count} 次时间提醒或阶段议程推进，控场意识鲜明")
        if phases_covered == 3:
            key_points.append("在开篇立论、中期深化与收尾决议三阶段均保持均衡参与")
        elif 15.0 <= avg_turn_sec <= 95.0:
            key_points.append(f"单次发言精炼适度（平均 {avg_turn_sec:.0f} 秒），不拖沓不霸麦")

        metrics = {
            "timing_control_count": timing_count,
            "phases_covered": phases_covered,
            "turns_count": turns,
            "avg_turn_duration_sec": round(avg_turn_sec, 1),
        }
        return float(total_raw), metrics, key_points


class ConvergenceScorer(BaseDimensionScorer):
    """
    6_3 收敛性评分器 (完整还原 V5 s6_3_convergence 向量距离差算法):
    测量每次该成员发言前后，场上言论距最终共识向量 (convergence_vec) 的距离缩减量:
        Δ_i = dist(u_{i-1}, conv) - dist(u_{i+1}, conv),  其中 dist(a, b) = 1 - a·b
    并修复了原 TalentGraph 中后半程 support 未校验时间戳的统计缺陷。
    """

    SCORER_ID = "6_3"
    SCORER_NAME = "推进能力 - 收敛性"

    def _resolve_convergence_text(self, context: ScoringContext) -> str:
        if context.transcript.convergence_decision and context.transcript.convergence_decision.strip():
            return context.transcript.convergence_decision.strip()
        # 兜底：取最终目标/核心决策节点 + 最后两轮候选人总结发言
        target_names = [
            n.name for n in context.standard_graph.nodes if n.type in ("u", "d")
        ][:8]
        cand_set = set(context.candidate_ids)
        tail_utts = [u.text[:200] for u in context.transcript.utterances if u.speaker_id in cand_set][-2:]
        return " ".join(target_names + tail_utts).strip() or context.standard_graph.title

    def _precompute_convergence_deltas(self, context: ScoringContext) -> Dict[str, tuple[float, float]]:
        """一次性批量计算全场每位发言人促成的收敛距离缩减总和 total_delta 与平均收敛率 ratio"""
        deltas_map = {cid: (0.0, 0.0) for cid in context.candidate_ids}
        cand_set = set(context.candidate_ids)
        utts = [u for u in context.transcript.utterances if u.speaker_id in cand_set]
        if len(utts) < 2:
            return deltas_map

        emb = context.embedding_engine or get_embedding_engine()
        if emb is None:
            return deltas_map

        conv_text = self._resolve_convergence_text(context)
        if not conv_text:
            return deltas_map

        unique_texts = list(dict.fromkeys([conv_text] + [u.text[:500] for u in utts if u.text.strip()]))
        mat = emb.embed_texts(unique_texts)
        txt_to_vec = {txt: mat[idx] for idx, txt in enumerate(unique_texts)}
        conv_vec = txt_to_vec[conv_text]
        zero_vec = np.zeros_like(conv_vec)

        def _dist_to_conv(t_str: str) -> float:
            s = t_str[:500] if t_str else ""
            v = txt_to_vec.get(s, zero_vec) if s.strip() else zero_vec
            return float(1.0 - np.dot(v, conv_vec))

        utt_dists = [_dist_to_conv(u.text) for u in utts]
        zero_dist = 1.0  # 1 - dot(0, conv_vec) = 1.0，与 V5 calculate_distances_batch 完全一致

        spk_deltas: Dict[str, List[tuple[float, float]]] = {cid: [] for cid in context.candidate_ids}
        n_u = len(utts)
        for i, u in enumerate(utts):
            pre_d = utt_dists[i - 1] if i > 0 else zero_dist
            post_d = utt_dists[i + 1] if i < n_u - 1 else utt_dists[i]
            delta = pre_d - post_d
            spk_deltas[u.speaker_id].append((pre_d, delta))

        for cid, pairs in spk_deltas.items():
            if not pairs:
                continue
            total_delta = sum(d for _, d in pairs)
            avg_ratio = sum(d / max(pre, 1e-6) for pre, d in pairs) / len(pairs)
            deltas_map[cid] = (total_delta, avg_ratio)

        return deltas_map

    def evaluate_all(self, context: ScoringContext) -> Dict[str, DimensionScore]:
        deltas_map = self._precompute_convergence_deltas(context)
        candidate_ids = context.candidate_ids
        raw_results: Dict[str, tuple[float, Dict[str, Any], List[str]]] = {}

        for spk_id in candidate_ids:
            raw_results[spk_id] = self._score_with_deltas(spk_id, context, deltas_map)

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

    def _score_with_deltas(
        self,
        speaker_id: str,
        context: ScoringContext,
        deltas_map: Dict[str, tuple[float, float]],
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"v5_convergence_delta": 0.0, "convergence_ratio": 0.0, "late_phase_node_mentions": 0}, []

        total_delta, avg_ratio = deltas_map.get(speaker_id, (0.0, 0.0))

        total_duration = context.transcript.total_duration_seconds or 1.0
        cutoff_time = total_duration * 0.60

        cand_graph = context.get_candidate_graph(speaker_id)
        late_mentions = (
            sum(1 for m in cand_graph.node_mentions if m.timestamp_seconds >= cutoff_time)
            if cand_graph
            else 0
        )

        # 修复原版未过滤时间戳的缺陷：严格限定在后 40% 收敛冲刺阶段发起的共识支持
        late_supports = sum(
            1
            for e in context.interaction_network.edges
            if e.source == speaker_id and e.type == "support" and e.timestamp_seconds >= cutoff_time
        )

        # 以 V5 向量收敛推进量 total_delta 为主导，辅以冲刺期决策收敛动作
        total_raw = total_delta * 8.0 + late_mentions * 1.5 + late_supports * 1.5

        key_points = []
        if total_delta > 0.05:
            key_points.append(f"发言有效拉近全场讨论与最终决议的语义距离（累计收敛推进量 +{total_delta:.2f}）")
        if late_mentions > 0 or late_supports > 0:
            key_points.append(f"在后半程收尾期贡献 {late_mentions} 项关键要点并促成 {late_supports} 次共识对齐")
        if not key_points and speaker.turn_count > 0:
            key_points.append("参与了团队结论的归纳与收束过程")

        metrics = {
            "v5_convergence_delta": round(total_delta, 4),
            "convergence_ratio": round(avg_ratio, 4),
            "late_phase_node_mentions": late_mentions,
            "late_consensus_pushes": late_supports,
        }
        return float(total_raw), metrics, key_points

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        deltas_map = self._precompute_convergence_deltas(context)
        return self._score_with_deltas(speaker_id, context, deltas_map)

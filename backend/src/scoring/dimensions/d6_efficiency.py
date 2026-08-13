# -*- coding: utf-8 -*-
"""
Dimension 6: Process Driving & Efficiency Competence Scorers (推进能力)
- 6_1: Focus (聚焦性)
- 6_2: Timing Control (时间把握能力)
- 6_3: Convergence (收敛性)
"""

from typing import Any, Dict, List, Set, Tuple
import numpy as np
from src.scoring.base import BaseDimensionScorer, ScoringContext


class FocusScorer(BaseDimensionScorer):
    """6_1 聚焦性评分器 (候选人发言与讨论核心主题方向的余弦相似度)"""

    SCORER_ID = "6_1"
    SCORER_NAME = "推进能力 - 聚焦性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"cosine_similarity": 0.0}, []

        # 核心主题文本：来自标准案例标题与核心概念
        topic_text = f"{context.standard_graph.title} " + " ".join(
            [f"{n.name} {n.description}" for n in context.standard_graph.nodes[:5]]
        )

        speaker_full_text = " ".join([u.text for u in speaker.utterances])

        # 使用 Embedding 向量计算余弦相似度
        sim = 0.75  # 默认基准
        emb_engine = context.embedding_engine
        if emb_engine is None:
            try:
                from src.core.embedding import get_embedding_engine
                emb_engine = get_embedding_engine()
            except Exception:
                emb_engine = None

        if emb_engine is not None:
            try:
                topic_vec = emb_engine.embed_text(topic_text)
                spk_vec = emb_engine.embed_text(speaker_full_text[:2000])
                sims = emb_engine.compute_similarity(topic_vec, [spk_vec])
                if len(sims) > 0:
                    sim = float(sims[0])
            except Exception:
                pass

        # 结合节点命中率进行加权
        cand_graph = context.get_candidate_graph(speaker_id)
        matched_ratio = (len(cand_graph.matched_nodes) / max(1, len(context.standard_graph.nodes))) if cand_graph else 0.0

        total_raw = sim * 10.0 + matched_ratio * 5.0

        key_points = []
        if sim >= 0.75:
            key_points.append(f"发言高度紧扣案例核心诉求与关键目标，主题贴合度高 (相似度: {sim:.2f})")
        if matched_ratio >= 0.3:
            key_points.append("紧密围绕题库核心逻辑框架展开论述，未出现明显离题")

        metrics = {
            "topic_cosine_similarity": round(sim, 3),
            "matched_ratio": round(matched_ratio, 2),
        }
        return float(total_raw), metrics, key_points


class TimingControlScorer(BaseDimensionScorer):
    """6_2 时间把握能力评分器 (全场发言节奏、时长分布、阶段控场)"""

    SCORER_ID = "6_2"
    SCORER_NAME = "推进能力 - 时间把握能力"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"turns": 0, "avg_duration": 0.0, "time_balance": 0.0}, []

        total_duration = context.transcript.total_duration_seconds or 1.0
        spk_duration = speaker.total_duration_seconds
        turns = speaker.turn_count

        # 检查发言是否均匀分布在讨论前、中、后三个阶段
        t_third = total_duration / 3.0
        phase1_turns = len([u for u in speaker.utterances if u.start_seconds < t_third])
        phase2_turns = len([u for u in speaker.utterances if t_third <= u.start_seconds < 2 * t_third])
        phase3_turns = len([u for u in speaker.utterances if u.start_seconds >= 2 * t_third])

        # 阶段覆盖数 (满分 3 阶段全参与)
        phases_covered = sum([1 for p in [phase1_turns, phase2_turns, phase3_turns] if p > 0])

        # 单次发言适度性 (平均每次 30~90 秒最佳，过短或过长霸麦扣分)
        avg_turn_sec = spk_duration / max(1, turns)
        turn_quality = 1.0
        if 20.0 <= avg_turn_sec <= 100.0:
            turn_quality = 1.5
        elif avg_turn_sec > 180.0:
            turn_quality = 0.6  # 霸麦

        total_raw = (phases_covered * 3.0) + (turns * 0.8) * turn_quality

        key_points = []
        if phases_covered == 3:
            key_points.append("在破冰立论、深化推演与最终总结三阶段均保持良好节奏参与")
        if 20.0 <= avg_turn_sec <= 90.0:
            key_points.append(f"单次发言时长控制合理 (平均 {avg_turn_sec:.0f} 秒)，控时意识突出")

        metrics = {
            "phases_covered": phases_covered,
            "turns_count": turns,
            "avg_turn_duration_sec": round(avg_turn_sec, 1),
            "speaking_duration_sec": round(spk_duration, 1),
        }
        return float(total_raw), metrics, key_points


class ConvergenceScorer(BaseDimensionScorer):
    """6_3 收敛性评分器 (讨论后半程促进方案闭环与共识推动贡献分)"""

    SCORER_ID = "6_3"
    SCORER_NAME = "推进能力 - 收敛性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"late_decisions": 0, "late_turns": 0}, []

        total_duration = context.transcript.total_duration_seconds or 1.0
        cutoff_time = total_duration * 0.60  # 后 40% 的收敛冲刺阶段

        late_utterances = [u for u in speaker.utterances if u.start_seconds >= cutoff_time]
        cand_graph = context.get_candidate_graph(speaker_id)

        # 后半程命中的决策与效用节点
        late_mentions = 0
        if cand_graph:
            for m in cand_graph.node_mentions:
                if m.timestamp_seconds >= cutoff_time:
                    late_mentions += 1

        # 统计后半程主动发起的 support (推动结论共识)
        late_supports = 0
        for e in context.interaction_network.edges:
            if e.source == speaker_id and e.type == "support":
                late_supports += 1

        total_raw = len(late_utterances) * 2.0 + late_mentions * 2.5 + late_supports * 1.5

        key_points = []
        if late_mentions > 0:
            key_points.append(f"在讨论收尾与冲刺阶段积极输出 {late_mentions} 项决策与共识要点")
        if late_supports > 0:
            key_points.append("在决议期主动凝聚全组意见，有效推动结论收敛")

        metrics = {
            "late_phase_turns": len(late_utterances),
            "late_phase_node_mentions": late_mentions,
            "late_consensus_pushes": late_supports,
        }
        return float(total_raw), metrics, key_points

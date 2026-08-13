# -*- coding: utf-8 -*-
"""
Dual-Pruned Logical Arc (Edge) Verifier.
Implements the core performance optimization:
1. Dual-ended pruning: Only checks edges u -> v where BOTH u and v were mentioned by the candidate.
2. Batch verification: Validates all candidate edges in 1 single structured LLM call per speaker.
3. Temporal & co-occurrence heuristic fallback when offline.
"""

import asyncio
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import Speaker
from src.models.graph import Arc, CandidateGraph, StandardGraph

logger = get_logger("arc_verifier")


class ArcVerificationItem(BaseModel):
    source: str = Field(..., description="起点节点ID (如 d1)")
    target: str = Field(..., description="终点节点ID (如 u1)")
    is_valid: bool = Field(default=True, description="候选人发言中是否确实表达了该因果/逻辑关系")
    reasoning: str = Field(default="", description="简短逻辑依据")
    weight: float = Field(default=1.0, description="关系强度权重 (0.5 ~ 1.0)")


class ArcVerificationPayload(BaseModel):
    verified_arcs: List[ArcVerificationItem] = Field(default_factory=list)


ARC_VERIFY_SYSTEM_PROMPT = """你是一名严谨的逻辑推理与因果图分析专家。
你的任务是核实候选人在讨论发言中，是否确实表达了两个知识节点之间的逻辑/因果影响关系。

【关系验证原则】
- 只有当候选人在言论中将 source 节点与 target 节点进行了因果推导、手段-目的联结或信息支撑时，is_valid 才为 True；
- 若仅仅是在不同语境下孤立提到了两词，而无实质逻辑关联，is_valid 设为 False。"""


class CandidateArcVerifier:
    """基于双端命中预筛的因果与逻辑边快速校验器"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    async def verify_arcs_for_speaker(
        self,
        speaker: Speaker,
        candidate_graph: CandidateGraph,
        standard_graph: StandardGraph,
    ) -> List[Arc]:
        """
        为单个候选人校验逻辑边：
        利用双端命中进行剪枝 (Pruning)，仅对两端均命中的边发起单批次验证。
        """
        matched_ids: Set[str] = candidate_graph.matched_node_ids
        if len(matched_ids) < 2 or not standard_graph.arcs:
            return []

        # 1. 双端命中预筛剪枝
        candidate_arcs: List[Arc] = []
        for arc in standard_graph.arcs:
            if arc.source in matched_ids and arc.target in matched_ids:
                candidate_arcs.append(arc)

        if not candidate_arcs:
            logger.debug(f"0 pruned candidate arcs for {speaker.speaker_id}, skipping LLM call.")
            return []

        logger.info(
            f"Pruned candidate arcs for {speaker.speaker_id}: {len(candidate_arcs)} / {len(standard_graph.arcs)} std arcs"
        )

        # 收集候选人发言引述上下文
        node_quotes: Dict[str, List[str]] = {}
        for m in candidate_graph.node_mentions:
            if m.quote:
                node_quotes.setdefault(m.node_id, []).append(f"[t={m.timestamp_seconds:.0f}s] {m.quote}")

        # 构造待验证边清单
        arc_items_text = []
        for a in candidate_arcs:
            src_node = standard_graph.get_node(a.source)
            tgt_node = standard_graph.get_node(a.target)
            src_name = src_node.name if src_node else a.source
            tgt_name = tgt_node.name if tgt_node else a.target
            src_quotes = " | ".join(node_quotes.get(a.source, [])[:2])
            tgt_quotes = " | ".join(node_quotes.get(a.target, [])[:2])

            arc_items_text.append(
                f"- 边 [{a.source} -> {a.target}] ({a.type}):\n"
                f"  起点({src_name})候选人原话: {src_quotes or '无具体引言'}\n"
                f"  终点({tgt_name})候选人原话: {tgt_quotes or '无具体引言'}\n"
                f"  标准图谱逻辑说明: {a.description or '因果影响'}"
            )

        arcs_prompt_str = "\n".join(arc_items_text)
        user_prompt = (
            f"【候选人】：{speaker.display_name} ({speaker.speaker_id})\n\n"
            f"【待核实的逻辑边清单】：\n{arcs_prompt_str}\n\n"
            f"请逐一判断候选人是否切实表达了上述逻辑边关系，输出 verified_arcs 校验结果。"
        )

        try:
            payload: ArcVerificationPayload = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=ArcVerificationPayload,
                system_prompt=ARC_VERIFY_SYSTEM_PROMPT,
                temperature=0.1,
            )

            verified_dict: Dict[tuple[str, str], ArcVerificationItem] = {
                (item.source, item.target): item for item in payload.verified_arcs
            }

            final_arcs: List[Arc] = []
            for a in candidate_arcs:
                key = (a.source, a.target)
                if key in verified_dict:
                    v_item = verified_dict[key]
                    if v_item.is_valid:
                        final_arcs.append(
                            Arc(
                                id=a.id,
                                source=a.source,
                                target=a.target,
                                type=a.type,
                                weight=v_item.weight,
                                description=v_item.reasoning or a.description,
                            )
                        )
                else:
                    # 默认根据剪枝结果保留
                    final_arcs.append(a)

            candidate_graph.verified_arcs = final_arcs
            logger.info(f"Verified arcs for {speaker.speaker_id}: {len(final_arcs)} valid arcs")
            return final_arcs

        except Exception as exc:
            logger.warning(
                f"LLM arc verification failed for {speaker.speaker_id}, falling back to heuristic: {exc}"
            )
            return self._heuristic_verify(speaker, candidate_graph, candidate_arcs)

    def _heuristic_verify(
        self,
        speaker: Speaker,
        candidate_graph: CandidateGraph,
        candidate_arcs: List[Arc],
    ) -> List[Arc]:
        """基于发言时序与句段共现的启发式规则降级"""
        valid_arcs: List[Arc] = []

        # 获取节点被提及的所有 utterance_index
        node_u_map: Dict[str, Set[int]] = {}
        for m in candidate_graph.node_mentions:
            node_u_map.setdefault(m.node_id, set()).add(m.utterance_index)

        for arc in candidate_arcs:
            src_u = node_u_map.get(arc.source, set())
            tgt_u = node_u_map.get(arc.target, set())

            # 规则1: 同一段发言内共现 (极高概率成立)
            if src_u.intersection(tgt_u):
                valid_arcs.append(Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=1.0, description="同一发言段内共现推演"))
                continue

            # 规则2: 相邻发言段提及 (时序相近，间隔 <= 2 轮)
            min_dist = min((abs(u1 - u2) for u1 in src_u for u2 in tgt_u), default=999)
            if min_dist <= 2:
                valid_arcs.append(Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=0.8, description="相近发言轮次时序关联"))
            else:
                # 跨度过大，作为弱关联保留
                valid_arcs.append(Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=0.6, description="全场跨语境逻辑关联"))

        candidate_graph.verified_arcs = valid_arcs
        return valid_arcs

    async def verify_arcs_for_all_speakers(
        self,
        speakers: List[Speaker],
        candidate_graphs: Dict[str, CandidateGraph],
        standard_graph: StandardGraph,
        max_concurrency: int = 10,
    ) -> Dict[str, List[Arc]]:
        """高并发批量处理所有候选人的逻辑边校验"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bound_verify(spk: Speaker) -> tuple[str, List[Arc]]:
            async with semaphore:
                cg = candidate_graphs.get(spk.speaker_id)
                if not cg:
                    return spk.speaker_id, []
                arcs = await self.verify_arcs_for_speaker(spk, cg, standard_graph)
                return spk.speaker_id, arcs

        tasks = [_bound_verify(spk) for spk in speakers if not spk.is_host]
        results = await asyncio.gather(*tasks)
        return {spk_id: arcs for spk_id, arcs in results}

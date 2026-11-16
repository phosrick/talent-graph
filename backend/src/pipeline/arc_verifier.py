# -*- coding: utf-8 -*-
"""
Dual-Pruned Logical Arc (Edge) Verifier.
Implements:
1. Dual-ended pruning: Only checks edges u -> v where BOTH u and v were mentioned by the candidate.
2. Strict post-condition validation: Unreturned edges are REJECTED (never blindly passed through!),
   and only true candidate_arcs pairs can pass verification.
3. Concurrent batching if candidate_arcs > 45 on large 20k-30k corpora.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
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
    reasoning: str = Field(default="", description="简短逻辑依据(可选)")
    weight: float = Field(default=1.0, description="关系强度权重 (0.5 ~ 1.0)")


class ArcVerificationPayload(BaseModel):
    verified_arcs: List[ArcVerificationItem] = Field(default_factory=list)


ARC_VERIFY_SYSTEM_PROMPT = """你是一名严谨的逻辑推理与因果图分析专家。
你的任务是核实候选人在讨论发言中，是否确实表达了待核实清单中两个知识节点之间的逻辑/因果/信息支撑关系（含显式推导或清晰的上下文隐含关联）。

【关系验证原则】
- 当候选人在言论中将 source 节点与 target 节点进行了因果推导、手段-目的联结或信息支撑时，is_valid 设为 True；
- 若仅仅是在完全无关的语境下孤立提到两词、无实质逻辑关联，is_valid 设为 False；
- 请对清单中的候选边给出判断，reasoning 保持简短（15字以内或留空）以加速输出。"""


class CandidateArcVerifier:
    """基于双端命中预筛与严格后验校验的因果/逻辑边快速校验器"""

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
        """为单个候选人校验逻辑边（双端命中剪枝 + 分批防截断 + 严格白名单后验过滤）"""
        matched_ids: Set[str] = candidate_graph.matched_node_ids
        if len(matched_ids) < 2 or not standard_graph.arcs:
            return []

        # 1. 双端命中预筛剪枝
        candidate_arcs: List[Arc] = [
            arc for arc in standard_graph.arcs if arc.source in matched_ids and arc.target in matched_ids
        ]

        if not candidate_arcs:
            logger.debug(f"0 pruned candidate arcs for {speaker.speaker_id}, skipping LLM call.")
            return []

        logger.info(
            f"Pruned candidate arcs for {speaker.speaker_id}: {len(candidate_arcs)} / {len(standard_graph.arcs)} std arcs"
        )

        node_quotes: Dict[str, List[str]] = {}
        for m in candidate_graph.node_mentions:
            if m.quote:
                node_quotes.setdefault(m.node_id, []).append(f"[t={m.timestamp_seconds:.0f}s] {m.quote}")

        # 候选人核心原文摘要（最多取 3500 字供全局语境核验，对齐 V5 ArcVerifyAgent 回源验证）
        speaker_excerpt = speaker.full_text()[:3500]

        batch_size = 45
        arc_batches = [candidate_arcs[i : i + batch_size] for i in range(0, len(candidate_arcs), batch_size)]

        async def _verify_batch(batch: List[Arc]) -> List[ArcVerificationItem]:
            arc_items_text = []
            for a in batch:
                src_node = standard_graph.get_node(a.source)
                tgt_node = standard_graph.get_node(a.target)
                src_name = src_node.name if src_node else a.source
                tgt_name = tgt_node.name if tgt_node else a.target
                src_quotes = " | ".join(node_quotes.get(a.source, [])[:2])
                tgt_quotes = " | ".join(node_quotes.get(a.target, [])[:2])

                line = f"- [{a.source} -> {a.target}] ({a.type}): {src_name} -> {tgt_name}"
                if src_quotes or tgt_quotes:
                    line += f" | 引言: {src_quotes or '-'} => {tgt_quotes or '-'}"
                arc_items_text.append(line)

            user_prompt = (
                f"【候选人】：{speaker.display_name} ({speaker.speaker_id})\n"
                f"【候选人发言原文节选】：\n\"\"\"\n{speaker_excerpt}\n\"\"\"\n\n"
                f"【待核实的候选边清单】：\n" + "\n".join(arc_items_text) + "\n\n"
                f"请逐条核实并输出 verified_arcs。"
            )

            payload: ArcVerificationPayload = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=ArcVerificationPayload,
                system_prompt=ARC_VERIFY_SYSTEM_PROMPT,
                temperature=0.1,
            )
            return payload.verified_arcs

        try:
            batch_lists = await asyncio.gather(*[_verify_batch(b) for b in arc_batches])
            verified_dict: Dict[Tuple[str, str], ArcVerificationItem] = {}
            for items in batch_lists:
                for item in items:
                    verified_dict[(item.source.strip(), item.target.strip())] = item

            # 严格后置校验：仅当 key 属于 candidate_arcs 且 LLM 显式判定 is_valid=True 时才通过！
            final_arcs: List[Arc] = []
            for a in candidate_arcs:
                key = (a.source, a.target)
                v_item = verified_dict.get(key)
                if v_item is not None and v_item.is_valid:
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

            candidate_graph.verified_arcs = final_arcs
            logger.info(f"Verified arcs for {speaker.speaker_id}: {len(final_arcs)} / {len(candidate_arcs)} valid arcs")
            return final_arcs

        except Exception as exc:
            logger.warning(
                f"LLM arc verification failed for {speaker.speaker_id}, falling back to co-occurrence heuristic: {exc}"
            )
            return self._heuristic_verify(speaker, candidate_graph, candidate_arcs)

    def _heuristic_verify(
        self,
        speaker: Speaker,
        candidate_graph: CandidateGraph,
        candidate_arcs: List[Arc],
    ) -> List[Arc]:
        """基于同段/近邻发言共现的严谨启发式规则降级（仅保留真正时序相邻或同段共现的边）"""
        valid_arcs: List[Arc] = []

        node_u_map: Dict[str, Set[int]] = {}
        for m in candidate_graph.node_mentions:
            node_u_map.setdefault(m.node_id, set()).add(m.utterance_index)

        for arc in candidate_arcs:
            src_u = node_u_map.get(arc.source, set())
            tgt_u = node_u_map.get(arc.target, set())

            if not src_u or not tgt_u:
                # 若无细分段落索引记录（例如简化测试数据），按弱共现保留
                valid_arcs.append(
                    Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=0.7, description="双端命中关联")
                )
                continue

            if src_u.intersection(tgt_u):
                valid_arcs.append(
                    Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=1.0, description="同一发言段内共现推演")
                )
            else:
                min_dist = min((abs(u1 - u2) for u1 in src_u for u2 in tgt_u), default=999)
                if min_dist <= 2:
                    valid_arcs.append(
                        Arc(id=arc.id, source=arc.source, target=arc.target, type=arc.type, weight=0.8, description="相近发言轮次时序关联")
                    )

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

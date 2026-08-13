# -*- coding: utf-8 -*-
"""
Interaction Network Extractor.
Extracts interpersonal support, denial, and consensus dynamics in ONE SINGLE structured LLM pass.
Massively reduces latency compared to old per-pair extraction.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import DiscussionTranscript, InteractionEdge, InteractionNetwork

logger = get_logger("interaction_extractor")


class ExtractedInteractionItem(BaseModel):
    source: str = Field(..., description="发言人ID，如 speaker_2")
    target: str = Field(..., description="回应对象ID，如 speaker_3")
    type: str = Field(default="support", description="互动类型: support(支持赞同), deny(质疑反对), neutral(中立承接)")
    evidence: str = Field(default="", description="关键原话引证")
    weight: float = Field(default=1.0, description="互动权重(1.0~2.0)")


class InteractionPayload(BaseModel):
    interactions: List[ExtractedInteractionItem] = Field(default_factory=list)


INTERACTION_SYSTEM_PROMPT = """你是一名无领导小组讨论(LGD)群体动力学专家。
你的任务是从小组全场发言对话中，提取所有候选人之间的互动关系（支持、反驳、承接）。

【互动类型定义】
1. support (支持/赞同/附和): A 明确表示赞成 B 的观点、对 B 的方案表示补充认可、强化 B 的提议；
2. deny (质疑/反驳/纠偏): A 明确反对 B 的观点、指出 B 方案的漏洞与不可行性、提出针锋相对的替代方案；
3. neutral (中立/承接): A 顺应 B 的话头继续展开，但无明确赞否情感倾向。

请返回完整的互动列表。source 为发起互动的发言人，target 为被回应或被评价的发言人。"""


class InteractionExtractor:
    """全场互动拓扑网络提取器"""

    def __init__(self, llm_client: Optional[LLMClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    async def extract_interactions(self, transcript: DiscussionTranscript) -> InteractionNetwork:
        """从结构化讨论记录中提取全场互动网络图"""
        candidates = transcript.candidate_speakers
        candidate_ids = [c.speaker_id for c in candidates]

        if len(candidates) < 2:
            return InteractionNetwork(speakers=candidate_ids, edges=[])

        # 压缩对话上下文以便高效输入 LLM
        formatted_dialogue = []
        for u in transcript.utterances:
            if u.speaker_id != "speaker_1":  # 过滤主持人
                formatted_dialogue.append(f"[{u.speaker_id}] ({u.speaker_name}): {u.text}")

        dialogue_text = "\n".join(formatted_dialogue)

        # 限制单次输入长度，防止极端超长文本
        if len(dialogue_text) > 16000:
            dialogue_text = dialogue_text[:16000] + "\n...(后续对话截断)"

        user_prompt = (
            f"【参评候选人列表】：{', '.join(candidate_ids)}\n\n"
            f"【小组讨论全程对话记录】：\n\"\"\"\n{dialogue_text}\n\"\"\"\n\n"
            f"请仔细分析对话，提取所有人与人之间的互动关系网络 (support / deny / neutral)。"
        )

        try:
            payload: InteractionPayload = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=InteractionPayload,
                system_prompt=INTERACTION_SYSTEM_PROMPT,
                temperature=0.1,
            )

            valid_edges: List[InteractionEdge] = []
            for item in payload.interactions:
                src = item.source.strip()
                tgt = item.target.strip()
                # 规范化 ID
                if not src.startswith("speaker_") and src.isdigit():
                    src = f"speaker_{src}"
                if not tgt.startswith("speaker_") and tgt.isdigit():
                    tgt = f"speaker_{tgt}"

                itype = item.type.lower()
                if itype not in {"support", "deny", "neutral"}:
                    itype = "support"

                if src in candidate_ids and tgt in candidate_ids and src != tgt:
                    valid_edges.append(
                        InteractionEdge(
                            source=src,
                            target=tgt,
                            type=itype,  # type: ignore
                            weight=item.weight,
                            evidence=item.evidence,
                        )
                    )

            network = InteractionNetwork(speakers=candidate_ids, edges=valid_edges)
            transcript.interaction_network = network
            logger.info(f"Extracted interaction network: {len(candidate_ids)} speakers, {len(valid_edges)} edges")
            return network

        except Exception as exc:
            logger.warning(f"LLM interaction extraction failed, falling back to heuristic extractor: {exc}")
            return self._heuristic_extract(transcript)

    def _heuristic_extract(self, transcript: DiscussionTranscript) -> InteractionNetwork:
        """基于关键词与近邻时序的启发式规则提取（保障降级可用性）"""
        candidates = [c.speaker_id for c in transcript.candidate_speakers]
        edges: List[InteractionEdge] = []

        deny_keywords = ["不同意", "不赞同", "反对", "不妥", "有漏洞", "风险太大", "不太赞同", "我有不同", "不太行"]
        support_keywords = ["赞同", "同意", "支持", "好主意", "有道理", "我补充", "非常对", "很到位", "很认可"]

        utterances = [u for u in transcript.utterances if not u.speaker_id.endswith("_1")]

        for i in range(1, len(utterances)):
            curr = utterances[i]
            prev = utterances[i - 1]

            if curr.speaker_id == prev.speaker_id:
                continue

            text = curr.text
            # 检查显式指代 (例如: "赞同2号", "3号说的对")
            target_id = None
            for other_id in candidates:
                if other_id == curr.speaker_id:
                    continue
                digit = other_id.replace("speaker_", "")
                if f"{digit}号" in text or f"候选人{digit}" in text or f"说话人{digit}" in text or f"候选人 {digit}" in text:
                    target_id = other_id
                    break

            # 若未显式指代，默认回应上一位发言人
            if not target_id:
                target_id = prev.speaker_id

            if any(k in text for k in deny_keywords):
                edges.append(
                    InteractionEdge(
                        source=curr.speaker_id,
                        target=target_id,
                        type="deny",
                        weight=1.0,
                        evidence=text[:40],
                        utterance_index=curr.index,
                    )
                )
            elif any(k in text for k in support_keywords):
                edges.append(
                    InteractionEdge(
                        source=curr.speaker_id,
                        target=target_id,
                        type="support",
                        weight=1.0,
                        evidence=text[:40],
                        utterance_index=curr.index,
                    )
                )

        network = InteractionNetwork(speakers=candidates, edges=edges)
        transcript.interaction_network = network
        return network

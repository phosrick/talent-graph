# -*- coding: utf-8 -*-
"""
Interaction Network & Global Topic/Consensus Extractor.
Extracts:
1. Interpersonal support, denial, and consensus dynamics (with real turn timestamps!)
2. Core discussion topic (core_topic, for V5 6_1 FocusScorer)
3. Final group convergence decision (convergence_decision, for V5 6_3 ConvergenceScorer)
Supports concurrent sliding-window extraction for ultra-long transcripts (>16,000 chars) with zero tail truncation.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import DiscussionTranscript, InteractionEdge, InteractionNetwork, Utterance

logger = get_logger("interaction_extractor")


class ExtractedInteractionItem(BaseModel):
    source: str = Field(..., description="发起回应的发言人ID，如 speaker_2")
    target: str = Field(..., description="被回应对象ID，如 speaker_3")
    type: str = Field(default="support", description="互动类型: support(支持赞同), deny(质疑反驳), neutral(中立承接)")
    utterance_index: Optional[int] = Field(default=None, description="对应发言行前缀序号 [idx]")
    evidence: str = Field(default="", description="关键原话简短引证")
    weight: float = Field(default=1.0, description="互动权重(1.0~2.0)")


class InteractionPayload(BaseModel):
    core_topic: str = Field(default="", description="本场讨论聚焦解决的核心问题摘要（30~80字）")
    convergence_decision: str = Field(default="", description="小组在讨论末期最终收敛达成的共识方案与结论摘要（50~150字）")
    interactions: List[ExtractedInteractionItem] = Field(default_factory=list)


INTERACTION_SYSTEM_PROMPT = """你是一名无领导小组讨论(LGD)群体动力学与决策分析专家。
请阅读小组讨论对话记录，一次性完成以下三项提取：
1. core_topic: 精炼概括本场讨论围绕的核心问题/主题（供聚焦性评估）；
2. convergence_decision: 概括小组最终达成一致的决策结论或收敛方案（供收敛性评估）；
3. interactions: 提取候选人之间的所有明确互动关系（source 为当前发言人，target 为被回应或评价的发言人，utterance_index 填发言行开头的数字编号）：
   - support (支持/赞同/补充认可): A 明确赞成、采纳或正向深化 B 的观点；
   - deny (质疑/反驳/纠偏): A 明确反对 B 的观点、指出其方案漏洞或提出对立替代方案；
   - neutral (中立/承接转换): A 顺承 B 的话头推进议程，无明显赞否倾向。"""


class InteractionExtractor:
    """全场互动拓扑网络与主题/收敛共识提取器（支持超长对话并发滑窗无截断提取）"""

    def __init__(self, llm_client: Optional[LLMClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    def _build_windows(self, utterances: List[Utterance], max_chars: int = 14000, overlap_turns: int = 3) -> List[str]:
        lines = [
            f"[{u.index}] [{u.speaker_id}] ({u.speaker_name}, {u.start_time_str}): {u.text}"
            for u in utterances
        ]
        full = "\n".join(lines)
        if len(full) <= 16000:
            return [full]

        windows: List[str] = []
        start_idx = 0
        n = len(lines)
        while start_idx < n:
            cur_len = 0
            end_idx = start_idx
            while end_idx < n and (cur_len + len(lines[end_idx]) + 1 <= max_chars or end_idx == start_idx):
                cur_len += len(lines[end_idx]) + 1
                end_idx += 1
            windows.append("\n".join(lines[start_idx:end_idx]))
            if end_idx >= n:
                break
            start_idx = max(start_idx + 1, end_idx - overlap_turns)
        return windows

    async def extract_interactions(self, transcript: DiscussionTranscript) -> InteractionNetwork:
        """从结构化讨论记录中提取全场互动网络图及核心主题/收敛共识"""
        candidates = transcript.candidate_speakers
        candidate_ids = [c.speaker_id for c in candidates]
        host_ids = {s.speaker_id for s in transcript.speakers.values() if s.is_host}
        u_map: Dict[int, Utterance] = {u.index: u for u in transcript.utterances}

        valid_utterances = [u for u in transcript.utterances if u.speaker_id not in host_ids]
        if not valid_utterances:
            return InteractionNetwork(speakers=candidate_ids, edges=[])

        if len(candidates) < 2:
            # 单人发言仍提炼主题与首尾收敛文本
            transcript.core_topic = transcript.core_topic or valid_utterances[0].text[:120]
            transcript.convergence_decision = transcript.convergence_decision or valid_utterances[-1].text[:160]
            net = InteractionNetwork(speakers=candidate_ids, edges=[])
            transcript.interaction_network = net
            return net

        windows = self._build_windows(valid_utterances)

        async def _extract_window(win_text: str, win_idx: int) -> InteractionPayload:
            user_prompt = (
                f"【参评候选人列表】：{', '.join(candidate_ids)}\n\n"
                f"【小组讨论对话记录（片段 {win_idx + 1}/{len(windows)}）】：\n\"\"\"\n{win_text}\n\"\"\"\n\n"
                f"请提取本片段的人际互动关系网络 (support / deny / neutral)，并概括讨论核心主题 (core_topic) 与最终共识决议 (convergence_decision)。"
            )
            return await self.llm.chat_json(
                prompt=user_prompt,
                response_model=InteractionPayload,
                system_prompt=INTERACTION_SYSTEM_PROMPT,
                temperature=0.1,
            )

        try:
            payloads: List[InteractionPayload] = list(
                await asyncio.gather(*[_extract_window(w, i) for i, w in enumerate(windows)])
            )

            valid_edges: List[InteractionEdge] = []
            seen_edge_keys: Set[Tuple[str, str, str, Optional[int], str]] = set()

            for p in payloads:
                c_topic = getattr(p, "core_topic", "")
                if isinstance(c_topic, str) and c_topic.strip() and not transcript.core_topic:
                    transcript.core_topic = c_topic.strip()
                # 最终共识优先取最后一个时间窗口总结出的决议
                c_dec = getattr(p, "convergence_decision", "")
                if isinstance(c_dec, str) and c_dec.strip():
                    transcript.convergence_decision = c_dec.strip()

                raw_interactions = getattr(p, "interactions", [])
                if not isinstance(raw_interactions, list):
                    raw_interactions = []
                for item in raw_interactions:
                    src = item.source.strip()
                    tgt = item.target.strip()
                    if not src.startswith("speaker_") and src.isdigit():
                        src = f"speaker_{src}"
                    if not tgt.startswith("speaker_") and tgt.isdigit():
                        tgt = f"speaker_{tgt}"

                    itype = item.type.lower().strip()
                    if itype not in {"support", "deny", "neutral"}:
                        itype = "support"

                    if src in candidate_ids and tgt in candidate_ids and src != tgt:
                        u_idx = item.utterance_index
                        ts = 0.0
                        if u_idx is not None and u_idx in u_map:
                            ts = u_map[u_idx].start_seconds
                        else:
                            # 若 LLM 未填 utterance_index，通过原话片段定位真实发言轮次与时间戳
                            ev_probe = (item.evidence or "").strip()[:16]
                            for u in valid_utterances:
                                if u.speaker_id == src and (not ev_probe or ev_probe in u.text):
                                    u_idx = u.index
                                    ts = u.start_seconds
                                    break

                        dedup_key = (src, tgt, itype, u_idx, (item.evidence or "")[:20])
                        if dedup_key in seen_edge_keys:
                            continue
                        seen_edge_keys.add(dedup_key)

                        valid_edges.append(
                            InteractionEdge(
                                source=src,
                                target=tgt,
                                type=itype,  # type: ignore
                                weight=item.weight,
                                evidence=item.evidence,
                                utterance_index=u_idx,
                                timestamp_seconds=ts,
                            )
                        )

            # 兜底填充 core_topic 与 convergence_decision
            if not transcript.core_topic:
                transcript.core_topic = " ".join(u.text[:80] for u in valid_utterances[:2])
            if not transcript.convergence_decision:
                transcript.convergence_decision = " ".join(u.text for u in valid_utterances[-2:])[:300]

            network = InteractionNetwork(speakers=candidate_ids, edges=valid_edges)
            transcript.interaction_network = network
            logger.info(
                f"Extracted interaction network: {len(candidate_ids)} speakers, {len(valid_edges)} edges "
                f"across {len(windows)} window(s)"
            )
            return network

        except Exception as exc:
            logger.warning(f"LLM interaction extraction failed, falling back to heuristic extractor: {exc}")
            return self._heuristic_extract(transcript)

    def _heuristic_extract(self, transcript: DiscussionTranscript) -> InteractionNetwork:
        """基于关键词与近邻时序的启发式规则提取（保障离线/降级可用性）"""
        candidates = [c.speaker_id for c in transcript.candidate_speakers]
        host_ids = {s.speaker_id for s in transcript.speakers.values() if s.is_host}
        edges: List[InteractionEdge] = []

        deny_keywords = ["不同意", "不赞同", "反对", "不妥", "有漏洞", "风险太大", "不太赞同", "我有不同", "不太行"]
        support_keywords = ["赞同", "同意", "支持", "好主意", "有道理", "我补充", "非常对", "很到位", "很认可"]

        utterances = [u for u in transcript.utterances if u.speaker_id not in host_ids]
        if utterances:
            transcript.core_topic = transcript.core_topic or " ".join(u.text[:80] for u in utterances[:2])
            transcript.convergence_decision = (
                transcript.convergence_decision or " ".join(u.text for u in utterances[-2:])[:300]
            )

        for i in range(1, len(utterances)):
            curr = utterances[i]
            prev = utterances[i - 1]

            if curr.speaker_id == prev.speaker_id:
                continue

            text = curr.text
            target_id = None
            for other_id in candidates:
                if other_id == curr.speaker_id:
                    continue
                digit = other_id.replace("speaker_", "")
                if (
                    f"{digit}号" in text
                    or f"候选人{digit}" in text
                    or f"说话人{digit}" in text
                    or f"候选人 {digit}" in text
                    or f"说话人 {digit}" in text
                ):
                    target_id = other_id
                    break

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
                        timestamp_seconds=curr.start_seconds,
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
                        timestamp_seconds=curr.start_seconds,
                    )
                )

        network = InteractionNetwork(speakers=candidates, edges=edges)
        transcript.interaction_network = network
        return network

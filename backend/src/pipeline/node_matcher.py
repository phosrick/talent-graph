# -*- coding: utf-8 -*-
"""
Candidate Node Matcher & V5 Behavioral Sub-Metrics Sniffer.
1. Identifies standard graph node mentions across large 20k-30k corpora via concurrent node batching
   (prevents 4096/8192 output truncation and Lost-in-the-Middle recall drop on 200+ node graphs).
2. Simultaneously extracts V5's behavioral sub-dimensions in the primary pass (0 extra round-trips!):
   - 3_3 Feasibility 4 sub-dimensions (resource_awareness, risk_sensitivity, execution_feasibility, adaptability)
   - 5_1 Openness 4 sub-dimensions (listening, supporting, analysis, pushing)
   - 6_2 Explicit time/agenda control count
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.embedding import EmbeddingEngine, get_embedding_engine
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import Speaker
from src.models.graph import CandidateGraph, Node, NodeMention, StandardGraph

logger = get_logger("node_matcher")


class MatchedNodeItem(BaseModel):
    node_id: str = Field(..., description="命中的标准节点ID，如 c1, d2, u1")
    utterance_index: int = Field(default=0, description="对应命中的发言段序号 [idx]")
    quote: str = Field(default="", description="发言中支持该命中的简短原话片段(<=30字)")
    confidence: float = Field(default=0.9, description="匹配置信度(0.0~1.0)")


class NovelNodeItem(BaseModel):
    name: str = Field(..., description="提出的标准库之外创新观点短名称")
    type: str = Field(default="d", description="类型: c=概念/事实, d=创新方案, u=价值目标")
    description: str = Field(default="", description="观点简述")
    quote: str = Field(default="", description="原话片段")


class NodeMatchPayload(BaseModel):
    matched_nodes: List[MatchedNodeItem] = Field(default_factory=list)
    novel_nodes: List[NovelNodeItem] = Field(default_factory=list)
    feasibility_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="3_3 候选人所提决策的四维可行性评估(0~10分): resource_awareness, risk_sensitivity, execution_feasibility, adaptability",
    )
    openness_counts: Optional[Dict[str, int]] = Field(
        default=None,
        description="5_1 候选人开放度四项行为次数: listening(积极倾听), supporting(认可他人), analysis(分析分歧原因), pushing(推动共识)",
    )
    timing_control_count: Optional[int] = Field(
        default=None,
        description="6_2 候选人提醒时间、规划讨论阶段节奏或控场的次数",
    )


NODE_MATCH_SYSTEM_PROMPT = """你是一名严谨的无领导小组讨论(LGD)认知图谱与行为评估专家。
请对照标准参考节点清单，分析指定候选人的发言，输出 JSON：
1. matched_nodes: 候选人实质提及或讨论过的标准节点（支持同义/近义语义表达匹配，填写准确的 node_id、发言序号 utterance_index 及简短 quote）；
2. novel_nodes: 候选人提出的超出标准节点清单、具有独立建设性的新颖方案或洞见（若无则为空列表）；
3. feasibility_scores: 对其所提方案在 4 个维度打分(0~10)：resource_awareness(资源限制意识)、risk_sensitivity(风险识别与管理)、execution_feasibility(执行门槛感知)、adaptability(动态性考虑)；
4. openness_counts: 统计其在发言中表现出以下 4 类行为的次数：listening(积极倾听反馈)、supporting(尊重认可他人合理见解)、analysis(客观分析分歧根源)、pushing(主动推进团队达成共识)；
5. timing_control_count: 统计其提醒时间、分配阶段时长或推进议程转换的次数。"""

NODE_BATCH_ONLY_SYSTEM_PROMPT = """你是一名严谨的无领导小组讨论(LGD)认知图谱匹配专家。
请判断候选人发言中是否提及或讨论了给定批次中的标准节点（支持同义语义匹配），只输出 matched_nodes（novel_nodes 留空列表，其余可选评分字段填 null）。"""


class CandidateNodeMatcher:
    """候选人言论节点匹配、创新观点提取与 V5 行为特征嗅探器"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)
        self.embedding = embedding_engine or get_embedding_engine(self.settings)

    def _build_speech_windows(self, speaker: Speaker, max_chars: int = 7500) -> List[str]:
        lines = [f"[{u.index}] ({u.start_time_str}) {u.text}" for u in speaker.utterances]
        joined = "\n".join(lines)
        if len(joined) <= max_chars:
            return [joined]

        windows: List[str] = []
        buf: List[str] = []
        cur = 0
        for ln in lines:
            if cur + len(ln) + 1 > max_chars and buf:
                windows.append("\n".join(buf))
                buf = [ln]
                cur = len(ln)
            else:
                buf.append(ln)
                cur += len(ln) + 1
        if buf:
            windows.append("\n".join(buf))
        return windows

    @staticmethod
    def _format_nodes_checklist(nodes: List[Node]) -> str:
        lines = []
        for n in nodes:
            kw_str = f" [{', '.join(n.keywords[:3])}]" if n.keywords else ""
            lines.append(f"- [{n.id}] ({n.type}) {n.name}{kw_str}")
        return "\n".join(lines)

    async def match_nodes_for_speaker(
        self,
        speaker: Speaker,
        standard_graph: StandardGraph,
    ) -> CandidateGraph:
        """为单个候选人并发匹配命中的标准节点、创新节点及 V5 四维可行性/开放度/控时指标"""
        if not speaker.utterances:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        speech_windows = self._build_speech_windows(speaker, max_chars=7500)
        batch_size = max(20, self.settings.node_match_batch_size)
        all_std_nodes = standard_graph.nodes
        node_batches: List[List[Node]] = (
            [all_std_nodes[i : i + batch_size] for i in range(0, len(all_std_nodes), batch_size)]
            if all_std_nodes
            else [[]]
        )

        u_ts_map = {u.index: u.start_seconds for u in speaker.utterances}
        default_ts = speaker.utterances[0].start_seconds

        async def _match_one_pair(w_idx: int, b_idx: int, speech_txt: str, n_batch: List[Node]) -> NodeMatchPayload:
            is_primary = (w_idx == 0 and b_idx == 0)
            checklist = self._format_nodes_checklist(n_batch)
            if is_primary:
                prompt = (
                    f"【待评估候选人】：{speaker.display_name} ({speaker.speaker_id})\n\n"
                    f"【标准题库参考节点清单】：\n{checklist}\n\n"
                    f"【该候选人的发言记录】：\n\"\"\"\n{speech_txt}\n\"\"\"\n\n"
                    f"请输出命中的 matched_nodes、创新观点 novel_nodes，以及 feasibility_scores、openness_counts、timing_control_count。"
                )
                sys_p = NODE_MATCH_SYSTEM_PROMPT
            else:
                prompt = (
                    f"【待评估候选人】：{speaker.display_name} ({speaker.speaker_id})\n\n"
                    f"【标准题库参考节点清单（分批 {b_idx + 1}/{len(node_batches)}）】：\n{checklist}\n\n"
                    f"【该候选人的发言记录】：\n\"\"\"\n{speech_txt}\n\"\"\"\n\n"
                    f"请仅输出本批次中候选人命中的 matched_nodes。"
                )
                sys_p = NODE_BATCH_ONLY_SYSTEM_PROMPT

            return await self.llm.chat_json(
                prompt=prompt,
                response_model=NodeMatchPayload,
                system_prompt=sys_p,
                temperature=0.1,
            )

        try:
            tasks = [
                _match_one_pair(w_idx, b_idx, s_txt, n_batch)
                for w_idx, s_txt in enumerate(speech_windows)
                for b_idx, n_batch in enumerate(node_batches)
            ]
            payloads: List[NodeMatchPayload] = list(await asyncio.gather(*tasks))

            matched_node_dict: Dict[str, Node] = {}
            node_mentions: List[NodeMention] = []
            seen_mentions = set()
            novel_nodes: List[Node] = []
            seen_novel_names = set()
            feasibility_ratings: Dict[str, float] = {}
            openness_counts: Dict[str, int] = {}
            timing_control_count: Optional[int] = None

            for payload in payloads:
                f_scores = getattr(payload, "feasibility_scores", None)
                if isinstance(f_scores, dict) and not feasibility_ratings:
                    for k in ("resource_awareness", "risk_sensitivity", "execution_feasibility", "adaptability"):
                        if k in f_scores:
                            try:
                                feasibility_ratings[k] = float(max(0.0, min(10.0, float(f_scores[k]))))
                            except (TypeError, ValueError):
                                pass

                o_counts = getattr(payload, "openness_counts", None)
                if isinstance(o_counts, dict) and not openness_counts:
                    for k in ("listening", "supporting", "analysis", "pushing"):
                        if k in o_counts:
                            try:
                                openness_counts[k] = max(0, int(o_counts[k]))
                            except (TypeError, ValueError):
                                pass

                t_cnt = getattr(payload, "timing_control_count", None)
                if isinstance(t_cnt, (int, float)) and not isinstance(t_cnt, bool) and timing_control_count is None:
                    timing_control_count = max(0, int(t_cnt))

                p_matched = getattr(payload, "matched_nodes", None)
                if not isinstance(p_matched, list):
                    p_matched = []
                for item in p_matched:
                    nid = item.node_id.strip()
                    std_node = standard_graph.get_node(nid)
                    if std_node:
                        matched_node_dict[nid] = std_node
                        m_key = (nid, item.utterance_index)
                        if m_key not in seen_mentions:
                            seen_mentions.add(m_key)
                            ts = u_ts_map.get(item.utterance_index, default_ts)
                            node_mentions.append(
                                NodeMention(
                                    node_id=nid,
                                    utterance_index=item.utterance_index,
                                    timestamp_seconds=ts,
                                    quote=item.quote,
                                    confidence=min(1.0, max(0.0, item.confidence)),
                                )
                            )

                p_novel = getattr(payload, "novel_nodes", None)
                if not isinstance(p_novel, list):
                    p_novel = []
                for nov in p_novel:
                    clean_nov = nov.name.strip()
                    if not clean_nov or clean_nov in seen_novel_names:
                        continue
                    seen_novel_names.add(clean_nov)
                    nov_type = nov.type.lower()
                    if nov_type not in {"c", "d", "u"}:
                        nov_type = "d"
                    novel_nodes.append(
                        Node(
                            id=f"novel_{speaker.speaker_id}_{len(novel_nodes) + 1}",
                            name=clean_nov,
                            type=nov_type,  # type: ignore
                            description=nov.description,
                            keywords=[nov.quote] if nov.quote else [],
                        )
                    )

            node_mentions.sort(key=lambda m: (m.timestamp_seconds, m.utterance_index))

            cand_graph = CandidateGraph(
                speaker_id=speaker.speaker_id,
                matched_nodes=list(matched_node_dict.values()),
                node_mentions=node_mentions,
                novel_nodes=novel_nodes,
                feasibility_ratings=feasibility_ratings,
                openness_counts=openness_counts,
                timing_control_count=timing_control_count,
            )
            logger.info(
                f"Matched nodes for {speaker.speaker_id}: {len(cand_graph.matched_nodes)} std nodes "
                f"({len(node_mentions)} mentions), {len(novel_nodes)} novel nodes"
            )
            return cand_graph

        except Exception as exc:
            logger.warning(
                f"LLM node matching failed for {speaker.speaker_id}, falling back to CPU embedding & heuristic: {exc}"
            )
            return self._heuristic_match(speaker, standard_graph)

    def _heuristic_match(
        self,
        speaker: Speaker,
        standard_graph: StandardGraph,
    ) -> CandidateGraph:
        """基于关键词包含与纯 CPU 向量相似度的无缝降级匹配"""
        matched_dict: Dict[str, Node] = {}
        mentions: List[NodeMention] = []

        if not standard_graph.nodes or not speaker.utterances:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        candidate_sentences: List[str] = []
        sentence_meta: List[Tuple[int, float, str]] = []
        for u in speaker.utterances:
            chunks = [
                s.strip()
                for s in u.text.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").splitlines()
                if len(s.strip()) > 3
            ]
            for c in chunks:
                candidate_sentences.append(c)
                sentence_meta.append((u.index, u.start_seconds, c))

        if not candidate_sentences:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        sent_vecs = self.embedding.embed_texts(candidate_sentences) if self.embedding is not None else None

        for node in standard_graph.nodes:
            node_matched = False
            best_mention = None

            keywords = [node.name] + [k for k in node.keywords if len(k) >= 2]
            for u_idx, ts, sent in sentence_meta:
                for kw in keywords:
                    if kw in sent:
                        node_matched = True
                        best_mention = NodeMention(
                            node_id=node.id,
                            utterance_index=u_idx,
                            timestamp_seconds=ts,
                            quote=sent[:50],
                            confidence=0.95,
                        )
                        break
                if node_matched:
                    break

            if not node_matched and self.embedding is not None and sent_vecs is not None:
                try:
                    node_text = f"{node.name} {node.description}".strip()
                    node_vec = self.embedding.embed_text(node_text)
                    similarities = self.embedding.compute_similarity(node_vec, sent_vecs)
                    best_idx = int(similarities.argmax())
                    best_sim = float(similarities[best_idx])

                    if best_sim >= 0.72:
                        node_matched = True
                        u_idx, ts, sent = sentence_meta[best_idx]
                        best_mention = NodeMention(
                            node_id=node.id,
                            utterance_index=u_idx,
                            timestamp_seconds=ts,
                            quote=sent[:50],
                            confidence=round(best_sim, 2),
                        )
                except Exception as e:
                    logger.debug(f"Embedding match error for node {node.id}: {e}")

            if node_matched and best_mention:
                matched_dict[node.id] = node
                mentions.append(best_mention)

        mentions.sort(key=lambda m: (m.timestamp_seconds, m.utterance_index))
        return CandidateGraph(
            speaker_id=speaker.speaker_id,
            matched_nodes=list(matched_dict.values()),
            node_mentions=mentions,
            novel_nodes=[],
        )

    async def match_nodes_for_all_speakers(
        self,
        speakers: List[Speaker],
        standard_graph: StandardGraph,
        max_concurrency: int = 10,
    ) -> Dict[str, CandidateGraph]:
        """高并发批量处理所有候选人的节点命中匹配"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bound_match(spk: Speaker) -> tuple[str, CandidateGraph]:
            async with semaphore:
                cand_graph = await self.match_nodes_for_speaker(spk, standard_graph)
                return spk.speaker_id, cand_graph

        tasks = [_bound_match(spk) for spk in speakers if not spk.is_host]
        results = await asyncio.gather(*tasks)
        return {spk_id: g for spk_id, g in results}

# -*- coding: utf-8 -*-
"""
Candidate Node Matcher.
Identifies standard graph node mentions and novel divergent nodes from candidate speech.
Employs concurrent batch LLM matching with pure-CPU embedding & keyword fallback.
"""

import asyncio
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.embedding import EmbeddingEngine, get_embedding_engine
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import Speaker, Utterance
from src.models.graph import CandidateGraph, Node, NodeMention, StandardGraph

logger = get_logger("node_matcher")


class MatchedNodeItem(BaseModel):
    node_id: str = Field(..., description="命中的标准节点ID，如 c1, d2, u1")
    utterance_index: int = Field(default=0, description="对应命中的发言段序号")
    quote: str = Field(default="", description="发言中支持该命中的关键原话语句")
    confidence: float = Field(default=0.9, description="匹配置信度(0.0~1.0)")


class NovelNodeItem(BaseModel):
    name: str = Field(..., description="提出的标准库之外创新观点短名称")
    type: str = Field(default="d", description="类型: c=概念/事实, d=创新方案, u=价值目标")
    description: str = Field(default="", description="观点详细阐述")
    quote: str = Field(default="", description="原话片段")


class NodeMatchPayload(BaseModel):
    matched_nodes: List[MatchedNodeItem] = Field(default_factory=list)
    novel_nodes: List[NovelNodeItem] = Field(default_factory=list)


NODE_MATCH_SYSTEM_PROMPT = """你是一名严谨的无领导小组讨论(LGD)言论分析与认知图谱专家。
你的任务是分析指定候选人的全场所有发言，判断其发言中：
1. 命中了参考题库标准图谱中的哪些节点 (c=事实/背景信息, d=决策/行动举措, u=目标/效用)；
2. 提出了哪些超出标准图谱之外、具有独立价值的创新发散观点 (novel_nodes)。

【匹配标准】
- 只有候选人确实表达了与标准节点相符的核心语义或事实时才算命中，切忌强行附会；
- 请指出命中的具体发言段序号 (utterance_index) 和原话引用 (quote)；
- 若候选人提出了独特且有建设性的新颖策略或洞见，将其归入 novel_nodes。"""


class CandidateNodeMatcher:
    """候选人言论节点匹配与创新观点提取器"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)
        self.embedding = embedding_engine or get_embedding_engine(self.settings)

    async def match_nodes_for_speaker(
        self,
        speaker: Speaker,
        standard_graph: StandardGraph,
    ) -> CandidateGraph:
        """为单位候选人匹配命中的标准节点及新观点"""
        if not speaker.utterances:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        # 整理候选人的全部发言
        speech_lines = []
        for u in speaker.utterances:
            speech_lines.append(f"[{u.index}] ({u.start_time_str}) {u.text}")
        speaker_corpus = "\n".join(speech_lines)

        # 限制单人发言输入长度
        if len(speaker_corpus) > 8000:
            speaker_corpus = speaker_corpus[:8000] + "\n...(部分长文本截断)"

        # 整理标准图谱的节点清单
        std_nodes_info = []
        for n in standard_graph.nodes:
            kw_str = f" [关键词: {', '.join(n.keywords)}]" if n.keywords else ""
            desc_str = f" - {n.description}" if n.description else ""
            std_nodes_info.append(f"- [{n.id}] ({n.type}) {n.name}{desc_str}{kw_str}")
        nodes_checklist = "\n".join(std_nodes_info)

        user_prompt = (
            f"【待评估候选人】：{speaker.display_name} ({speaker.speaker_id})\n\n"
            f"【标准题库参考节点清单】：\n{nodes_checklist}\n\n"
            f"【该候选人的全部发言记录】：\n\"\"\"\n{speaker_corpus}\n\"\"\"\n\n"
            f"请仔细对照候选人发言，输出其命中的标准节点列表 (matched_nodes) 以及提出的额外创新观点 (novel_nodes)。"
        )

        try:
            payload: NodeMatchPayload = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=NodeMatchPayload,
                system_prompt=NODE_MATCH_SYSTEM_PROMPT,
                temperature=0.1,
            )

            # 组装结果
            matched_node_dict: Dict[str, Node] = {}
            node_mentions: List[NodeMention] = []

            for item in payload.matched_nodes:
                nid = item.node_id.strip()
                std_node = standard_graph.get_node(nid)
                if std_node:
                    matched_node_dict[nid] = std_node
                    # 查找对应 utterance 的时间戳
                    ts = 0.0
                    for u in speaker.utterances:
                        if u.index == item.utterance_index:
                            ts = u.start_seconds
                            break

                    node_mentions.append(
                        NodeMention(
                            node_id=nid,
                            utterance_index=item.utterance_index,
                            timestamp_seconds=ts,
                            quote=item.quote,
                            confidence=min(1.0, max(0.0, item.confidence)),
                        )
                    )

            novel_nodes: List[Node] = []
            for idx, nov in enumerate(payload.novel_nodes, 1):
                nov_type = nov.type.lower()
                if nov_type not in {"c", "d", "u"}:
                    nov_type = "d"
                novel_nodes.append(
                    Node(
                        id=f"novel_{speaker.speaker_id}_{idx}",
                        name=nov.name,
                        type=nov_type,  # type: ignore
                        description=nov.description,
                        keywords=[nov.quote] if nov.quote else [],
                    )
                )

            cand_graph = CandidateGraph(
                speaker_id=speaker.speaker_id,
                matched_nodes=list(matched_node_dict.values()),
                node_mentions=node_mentions,
                novel_nodes=novel_nodes,
            )
            logger.info(
                f"Matched nodes for {speaker.speaker_id}: {len(cand_graph.matched_nodes)} std nodes ({len(node_mentions)} mentions), {len(novel_nodes)} novel nodes"
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
        """基于关键词包含与 CPU 向量相似度的无缝降级匹配"""
        matched_dict: Dict[str, Node] = {}
        mentions: List[NodeMention] = []

        if not standard_graph.nodes or not speaker.utterances:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        # 提取候选人所有句子
        candidate_sentences = []
        sentence_meta = []  # (utterance_index, timestamp, text)
        for u in speaker.utterances:
            chunks = [s.strip() for s in u.text.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").splitlines() if len(s.strip()) > 3]
            for c in chunks:
                candidate_sentences.append(c)
                sentence_meta.append((u.index, u.start_seconds, c))

        if not candidate_sentences:
            return CandidateGraph(speaker_id=speaker.speaker_id)

        # 针对每个标准节点进行关键词与语义匹配
        for node in standard_graph.nodes:
            node_matched = False
            best_mention = None

            # 1. 关键词与名称直接匹配
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

            # 2. 若未通过关键词命中，尝试纯 CPU 语义相似度匹配
            if not node_matched and self.embedding is not None:
                try:
                    node_text = f"{node.name} {node.description}"
                    node_vec = self.embedding.embed_text(node_text)
                    sent_vecs = self.embedding.embed_texts(candidate_sentences)
                    similarities = self.embedding.compute_similarity(node_vec, sent_vecs)
                    best_idx = int(similarities.argmax())
                    best_sim = float(similarities[best_idx])

                    if best_sim >= 0.72:  # 高置信度阈值
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

# -*- coding: utf-8 -*-
"""
Standard Graph Extractor.
Extracts Concept Nodes (c), Decision Nodes (d), Utility Nodes (u), and Arcs from Expert Corpus Reference Answers.
"""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.embedding import get_embedding_engine
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.graph import Arc, Node, StandardGraph

logger = get_logger("graph_extractor")


class ExtractedNode(BaseModel):
    id: str = Field(..., description="节点ID，如 c1, c2, d1, d2, u1")
    name: str = Field(..., description="节点简短名称")
    type: str = Field(..., description="节点类型: c=概念/事实/背景, d=决策/行动措施, u=效用/目标")
    description: str = Field(default="", description="节点详细说明")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")


class ExtractedArc(BaseModel):
    source: str = Field(..., description="起点节点ID (如 d1)")
    target: str = Field(..., description="终点节点ID (如 u1)")
    type: str = Field(default="causal", description="边类型: causal(因果影响), info(信息支持), update(状态更新)")
    description: str = Field(default="", description="逻辑关联说明")


class GraphExtractionPayload(BaseModel):
    title: str = Field(default="讨论案例标准图谱", description="案例标题")
    nodes: List[ExtractedNode] = Field(default_factory=list, description="提取的所有节点")
    arcs: List[ExtractedArc] = Field(default_factory=list, description="提取的所有逻辑边")


EXTRACTION_SYSTEM_PROMPT = """你是一名资深的决策分析与图论建模专家。
请仔细阅读给定的无领导小组讨论“标准参考答案/案例专家材料”，构建完整的因果与影响图谱(Influence Diagram)。

【节点分类规范】
1. 概念/信息节点 (c, concept/chance): 案例背景、行业事实、客观约束条件、外部市场信息等；
2. 决策节点 (d, decision): 具体的解决策略、行动举措、执行方案、管理动作等；
3. 效用/目标节点 (u, utility): 决策希望达成的最终目标、衡量标准、商业价值或社会效益（如利润提升、品牌破圈、风险降低）。

【边关系规范】
1. info (信息边): 概念事实 c 为决策方案 d 提供决策依据 (c -> d)；
2. causal (因果边): 决策方案 d 引发后续决策或直接作用于效用目标 u (d -> u, d -> d)；
3. update (更新边): 外部状态变化带来的动态影响。

请确保节点划分颗粒度明确、覆盖全面且不重复，并给出清晰合理的有向边连接。"""


class StandardGraphExtractor:
    """自适应标准图谱提取器 (支持单次直提与超长语料智能分块并发合并)"""

    def __init__(self, llm_client: Optional[LLMClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    def _split_text_into_chunks(self, text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
        """将超长参考语料按自然段落与标点智能切分为带重叠窗口的文本块"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        total_len = len(text)
        while start < total_len:
            end = min(start + chunk_size, total_len)
            if end < total_len:
                # 优先寻找双换行段落分隔
                newline_pos = text.rfind("\n\n", start + chunk_size // 2, end)
                if newline_pos == -1:
                    # 次选单换行
                    newline_pos = text.rfind("\n", start + chunk_size // 2, end)
                if newline_pos == -1:
                    # 再次选句号
                    period_pos = text.rfind("。", start + chunk_size // 2, end)
                    if period_pos != -1:
                        end = period_pos + 1
                else:
                    end = newline_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= total_len:
                break
            start = max(start + 1, end - overlap)

        return chunks

    async def _extract_single_chunk(
        self,
        chunk_text: str,
        chunk_idx: int = 0,
        total_chunks: int = 1,
    ) -> GraphExtractionPayload:
        """抽取单个分块的子图谱"""
        prefix = f"（分块 {chunk_idx + 1}/{total_chunks}）" if total_chunks > 1 else ""
        user_prompt = (
            f"【案例与标准参考答案内容{prefix}】：\n"
            f"\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"请严格按照规范提取出本内容涉及的全部概念节点(c)、决策节点(d)、效用节点(u)以及它们之间的有向逻辑边关系。"
        )

        return await self.llm.chat_json(
            prompt=user_prompt,
            response_model=GraphExtractionPayload,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.1,
        )

    def _merge_extracted_payloads(
        self,
        payloads: List[GraphExtractionPayload],
        corpus_id: str,
        corpus_hash: str,
        title: str = "",
    ) -> StandardGraph:
        """
        跨分块智能图谱合并与语义去重算法：
        1. 按 c(概念), d(决策), u(效用) 分类，基于文本与语义相似度合并同义节点
        2. 全局重新分配规范化 ID (c1..cn, d1..dn, u1..un)
        3. 重映射所有分块提取的有向边并剔除孤立自环与重复边
        """
        embedding_engine = None
        try:
            embedding_engine = get_embedding_engine(self.settings)
        except Exception:
            pass

        # 映射表: (chunk_idx, orig_node_id) -> global_unified_id
        orig_to_global_map: Dict[Tuple[int, str], str] = {}
        unified_nodes_by_type: Dict[str, List[dict]] = {"c": [], "d": [], "u": []}

        for chunk_idx, payload in enumerate(payloads):
            for n in payload.nodes:
                n_type = n.type.lower()
                if n_type not in {"c", "d", "u"}:
                    n_type = n.id[0].lower() if n.id and n.id[0].lower() in {"c", "d", "u"} else "c"

                existing_group = unified_nodes_by_type[n_type]
                matched_idx = -1

                # 1. 优先完全一致匹配
                for idx, exist_n in enumerate(existing_group):
                    if exist_n["name"].strip() == n.name.strip():
                        matched_idx = idx
                        break

                # 2. 向量余弦相似度匹配 (高阈值 0.88 保证精确)
                if matched_idx == -1 and embedding_engine is not None and len(existing_group) > 0:
                    try:
                        n_vec = embedding_engine.embed_text(n.name)
                        exist_vecs = [exist_n["_vec"] for exist_n in existing_group if exist_n.get("_vec") is not None]
                        if len(exist_vecs) == len(existing_group):
                            sims = embedding_engine.compute_similarity(n_vec, exist_vecs)
                            if len(sims) > 0 and float(sims.max()) >= 0.88:
                                matched_idx = int(sims.argmax())
                    except Exception:
                        pass

                if matched_idx != -1:
                    # 合并到已有节点
                    target_global_id = existing_group[matched_idx]["id"]
                    orig_to_global_map[(chunk_idx, n.id)] = target_global_id
                    existing_group[matched_idx]["keywords"] = list(
                        set(existing_group[matched_idx]["keywords"] + n.keywords)
                    )
                    if n.description and n.description not in existing_group[matched_idx]["description"]:
                        existing_group[matched_idx]["description"] += f"；{n.description}"
                else:
                    # 创建新的全局规范节点
                    new_idx = len(existing_group) + 1
                    global_id = f"{n_type}{new_idx}"
                    orig_to_global_map[(chunk_idx, n.id)] = global_id

                    node_vec = None
                    if embedding_engine is not None:
                        try:
                            node_vec = embedding_engine.embed_text(n.name)
                        except Exception:
                            pass

                    existing_group.append({
                        "id": global_id,
                        "name": n.name.strip(),
                        "type": n_type,
                        "description": n.description,
                        "keywords": list(set(n.keywords)),
                        "_vec": node_vec,
                    })

        # 组装全局 Node 对象
        final_nodes: List[Node] = []
        for n_type in ["c", "d", "u"]:
            for item in unified_nodes_by_type[n_type]:
                final_nodes.append(
                    Node(
                        id=item["id"],
                        name=item["name"],
                        type=item["type"],
                        description=item["description"],
                        keywords=item["keywords"],
                    )
                )

        # 重映射与合并边
        final_arcs: List[Arc] = []
        seen_arcs: Set[Tuple[str, str, str]] = set()

        for chunk_idx, payload in enumerate(payloads):
            for a in payload.arcs:
                arc_type = a.type.lower()
                if arc_type not in {"causal", "info", "update"}:
                    arc_type = "causal"

                src_id = orig_to_global_map.get((chunk_idx, a.source))
                dst_id = orig_to_global_map.get((chunk_idx, a.target))

                if not src_id or not dst_id:
                    for n in final_nodes:
                        if not src_id and n.name == a.source.strip():
                            src_id = n.id
                        if not dst_id and n.name == a.target.strip():
                            dst_id = n.id

                if src_id and dst_id and src_id != dst_id:
                    arc_key = (src_id, dst_id, arc_type)
                    if arc_key not in seen_arcs:
                        seen_arcs.add(arc_key)
                        final_arcs.append(
                            Arc(
                                source=src_id,
                                target=dst_id,
                                type=arc_type,
                                description=a.description,
                            )
                        )

        final_title = title or (payloads[0].title if payloads and payloads[0].title else "讨论案例标准图谱")

        return StandardGraph(
            corpus_id=corpus_id,
            corpus_hash=corpus_hash,
            title=final_title,
            nodes=final_nodes,
            arcs=final_arcs,
        )

    async def extract_graph(
        self,
        corpus_text: str,
        corpus_id: str = "default_corpus",
        title: str = "",
    ) -> StandardGraph:
        """
        自适应标准图谱提取：
        - 短语料 (<= 阈值): 一次性全局提取，图谱因果完整性最优
        - 超长语料 (> 阈值): 智能分块并发抽取 + 跨块语义去重图谱合并
        """
        corpus_text = corpus_text.strip()
        if not corpus_text:
            raise ValueError("Corpus text cannot be empty.")

        corpus_len = len(corpus_text)
        corpus_hash = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
        threshold = self.settings.corpus_auto_chunk_threshold

        logger.info(f"Extracting Standard Graph (len={corpus_len}, threshold={threshold}, sha256={corpus_hash[:8]}...)")

        if corpus_len <= threshold:
            # 1. 短语料单次提取
            payload = await self._extract_single_chunk(corpus_text, chunk_idx=0, total_chunks=1)
            graph = self._merge_extracted_payloads([payload], corpus_id=corpus_id, corpus_hash=corpus_hash, title=title)
        else:
            # 2. 超长语料自适应分块并发提取
            chunks = self._split_text_into_chunks(
                corpus_text,
                chunk_size=self.settings.corpus_chunk_size,
                overlap=self.settings.corpus_chunk_overlap,
            )
            logger.info(
                f"⚡ [ADAPTIVE CHUNKING] Corpus len={corpus_len} > {threshold}, split into {len(chunks)} chunks for concurrent extraction"
            )
            tasks = [
                self._extract_single_chunk(chunk, chunk_idx=i, total_chunks=len(chunks))
                for i, chunk in enumerate(chunks)
            ]
            payloads = await asyncio.gather(*tasks)
            graph = self._merge_extracted_payloads(payloads, corpus_id=corpus_id, corpus_hash=corpus_hash, title=title)

        logger.info(f"Successfully extracted Standard Graph: {len(graph.nodes)} nodes, {len(graph.arcs)} arcs")
        return graph


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Extract Standard Graph from Corpus Text")
    parser.add_argument("--corpus", required=True, help="Path to corpus.txt")
    parser.add_argument("--output", default="standard_graph.json", help="Path to save output JSON")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        exit(1)

    text = corpus_path.read_text(encoding="utf-8")
    extractor = StandardGraphExtractor()
    graph = asyncio.run(extractor.extract_graph(text, corpus_id=corpus_path.stem))
    Path(args.output).write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    print(f"Standard Graph saved to {args.output} with {len(graph.nodes)} nodes, {len(graph.arcs)} arcs")

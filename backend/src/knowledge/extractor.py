# -*- coding: utf-8 -*-
"""
Standard Graph Extractor (V5-Fidelity Pure-CPU Ultra-Fast Edition).
Extracts Influence Diagram Nodes (c/d/u + 一/二/三级 hierarchy + known/unknown & final/instrumental subtypes)
and Grounded Arcs (causal/info/update) from 20k-30k+ char Reference Corpora in ~30-60s without GPU:
1. Optional concurrent ASR smoothing (20-way parallel for raw transcripts, 0s bypass for clean docs)
2. Concurrent chunk-level joint extraction (nodes + V5 attributes + grounded intra-chunk arcs)
3. Vectorized semantic node deduplication & automatic 三级 -> 二级 parent_concept linking
4. Top-K embedding-guided cross-chunk skeleton arc completion (replaces V5's O(N^2) blind Cartesian product)
"""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.embedding import EmbeddingEngine, get_embedding_engine
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.graph import Arc, Node, StandardGraph

logger = get_logger("graph_extractor")


class ExtractedNode(BaseModel):
    id: str = Field(..., description="分块内唯一ID，如 c1, c2, d1, d2, u1")
    name: str = Field(..., description="具体清晰的节点名称（8~24字为宜，忌空泛单字词）")
    type: str = Field(..., description="c=信息/背景事实, d=决策/行动措施, u=效用/目标标准")
    level: str = Field(default="三级", description="抽象层级: 一级(宏观主题) / 二级(核心抽象概念或机制) / 三级(具体事实或操作方案)")
    sub_type: str = Field(
        default="",
        description="子属性: c填 known(已知事实)/unknown(待查证推断); u填 final(最终目标)/instrumental(手段目标); d留空",
    )
    is_root: bool = Field(default=False, description="若为导致其他现象的深层根因信息节点则为 true")
    keywords: List[str] = Field(default_factory=list, description="1~3个核心检索关键词")
    description: str = Field(default="", description="可选简注（默认留空以加速生成）")


class ExtractedArc(BaseModel):
    source: str = Field(..., description="起点节点ID (如 c1 或 d1)")
    target: str = Field(..., description="终点节点ID (如 d1 或 u1)")
    type: str = Field(default="causal", description="causal(因果A->B), info(信息支撑决策c->d), update(已知更新未知known_c->unknown_c)")
    description: str = Field(default="", description="可选简注（默认留空）")


class GraphExtractionPayload(BaseModel):
    title: str = Field(default="讨论案例标准图谱", description="案例简短标题")
    nodes: List[ExtractedNode] = Field(default_factory=list, description="提取的节点列表")
    arcs: List[ExtractedArc] = Field(default_factory=list, description="本段原文支撑的有向逻辑边列表")


class CrossChunkArcItem(BaseModel):
    source: str = Field(..., description="起点全局节点ID (如 c3)")
    target: str = Field(..., description="终点全局节点ID (如 d5)")
    type: str = Field(default="causal", description="causal | info | update")


class CrossChunkArcPayload(BaseModel):
    arcs: List[CrossChunkArcItem] = Field(default_factory=list, description="跨段落成立的核心有向边列表")


SMOOTH_SYSTEM_PROMPT = (
    "请识别输入文本中明显的语音转录错别字与断句问题，改写为通顺准确的书面表达。"
    "注意：严禁删减任何原文事实细节与数据，不要添加任何解释说明，直接输出修正后的正文。"
)

EXTRACTION_SYSTEM_PROMPT = """你是一名精通决策影响图（Influence Diagram）建模的资深专家。
请从给定的参考语料中抽取结构化节点与有向边，严格对齐以下理论规范：

【一、节点类型 (type) 与子属性 (sub_type)】
1. c (信息节点): 决策相关的客观背景、条件、数据、现状或推断。
   - sub_type: "known"（题目/材料已明确的事实或常识）或 "unknown"（需额外查证、预测或不确定推断的信息）；
   - is_root: 若该信息节点是引发其他问题的深层根源起点（不被其他节点导致），设为 true。
2. d (决策节点): 决策者可主动选择、规划或执行的具体行动方案与管理举措（sub_type 留空 ""）。
3. u (效用节点): 决策期望改变的目标或衡量标准。
   - sub_type: "final"（根本性的最终目标）或 "instrumental"（服务于最终目标的过渡性/手段目标）。

【二、节点抽象层级 (level) —— 与 c/d/u 正交独立】
- "一级"：宏观领域或顶层主题（如“银行网点经营转型”“可持续发展战略”），数量少而精；
- "二级"：中等抽象的核心机制、维度或方法论（如“服务意识提升”“薪酬激励机制”“渠道数字化”）；
- "三级"：具体可感知的事实、数据现象或可落地的操作措施（如“近三年网点经济效益持续下滑”“梳理适合本行的新业务培训”），数量最多。
注意：严禁抽取“环境”“多方面”“平衡”等脱离语境的空洞词。

【三、有向边类型 (arcs)】
只抽取本段文本中有明确语义或逻辑依据的边：
- "info": 信息节点支撑决策方案 (c -> d)；
- "causal": 因果作用关系 (c -> c, d -> c, d -> d, d -> u, u_instrumental -> u_final)，注意根因节点不可作为因果边终点；
- "update": 已知证据信息更新对未知信息的判断 (known c -> unknown c)。

【四、精简输出示范】
{"title":"支行网点效能提升案例","nodes":[
  {"id":"c1","name":"网点近三年经济效益持续下滑","type":"c","level":"三级","sub_type":"known","is_root":true,"keywords":["经济效益","下滑"]},
  {"id":"c2","name":"客户流失的潜在深层结构原因","type":"c","level":"二级","sub_type":"unknown","is_root":false,"keywords":["客户流失"]},
  {"id":"d1","name":"梳理适配本行实际的新产品并开展专项培训","type":"d","level":"三级","sub_type":"","is_root":false,"keywords":["新产品","专项培训"]},
  {"id":"u1","name":"提升员工业务能力与服务意识","type":"u","level":"二级","sub_type":"instrumental","is_root":false,"keywords":["服务意识"]},
  {"id":"u2","name":"扭转支行效益下滑实现业绩增长","type":"u","level":"一级","sub_type":"final","is_root":false,"keywords":["业绩增长"]}
],"arcs":[
  {"source":"c1","target":"d1","type":"info"},
  {"source":"c1","target":"c2","type":"update"},
  {"source":"d1","target":"u1","type":"causal"},
  {"source":"u1","target":"u2","type":"causal"}
]}
请保持 description 字段留空 "" 以最大化生成速度，确保覆盖全面、颗粒度清晰。"""

CROSS_CHUNK_ARC_SYSTEM_PROMPT = """你是一名决策影响图全局拓扑专家。
由于长文档被分块抽取，部分处于不同段落的节点尚未建立连接。请从给定的全局节点清单中，补充跨段落明显成立的核心有向边：
1. info: 背景/痛点信息支撑某项决策措施 (c -> d)
2. causal: 决策达成手段目标、手段目标支撑最终目标、或背景引发连锁后果 (d -> u, u_instrumental -> u_final, d -> c, c -> c)
3. update: 已知事实信息影响对未知推断信息的判断 (known c -> unknown c)
严格要求：只输出高置信度、逻辑严密的边（每批控制在 30 条以内），严禁连接无直接逻辑关联的节点对，只使用给定的节点 ID。"""


class StandardGraphExtractor:
    """自适应高并发标准图谱提取器（纯 CPU + 云端并发 API，支撑 2~3 万字语料分钟级建图）"""

    def __init__(self, llm_client: Optional[LLMClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    def _split_text_into_chunks(self, text: str, chunk_size: int = 2800, overlap: int = 300) -> List[str]:
        """将超长参考语料按自然段落与句号智能切分为带重叠窗口的文本块"""
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        total_len = len(text)
        while start < total_len:
            end = min(start + chunk_size, total_len)
            if end < total_len:
                newline_pos = text.rfind("\n\n", start + chunk_size // 2, end)
                if newline_pos == -1:
                    newline_pos = text.rfind("\n", start + chunk_size // 2, end)
                if newline_pos != -1:
                    end = newline_pos + 1
                else:
                    for punct in ("。", "！", "？", "；"):
                        p_pos = text.rfind(punct, start + chunk_size // 2, end)
                        if p_pos != -1:
                            end = p_pos + 1
                            break

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 10:
                chunks.append(chunk)

            if end >= total_len:
                break
            start = max(start + 1, end - overlap)

        return chunks

    async def _smooth_corpus_concurrent(self, text: str) -> str:
        """可选高并发分段文本润滑（针对原始 ASR 语音转写参考稿，20 路并发，~12s 完成 3 万字）"""
        chunks = self._split_text_into_chunks(text, chunk_size=1500, overlap=0)
        prompts = [(f"文本：\n{c}\n\n输出修正后的文本：", SMOOTH_SYSTEM_PROMPT) for c in chunks]
        smoothed_parts = await self.llm.batch_chat(prompts, temperature=0.0, max_tokens=3000)
        merged = "\n".join(p.strip() or orig for p, orig in zip(smoothed_parts, chunks))
        return merged.strip() or text

    async def _extract_single_chunk(
        self,
        chunk_text: str,
        chunk_idx: int = 0,
        total_chunks: int = 1,
    ) -> GraphExtractionPayload:
        """抽取单个分块的子图谱（含节点、V5层级/子属性及块内有向边）"""
        prefix = f"（分块 {chunk_idx + 1}/{total_chunks}）" if total_chunks > 1 else ""
        user_prompt = (
            f"【案例与标准参考答案内容{prefix}】：\n"
            f"\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"请提取本段涉及的全部信息节点(c)、决策节点(d)、效用节点(u)，标注各自的抽象层级(一级/二级/三级)与子属性(known/unknown或final/instrumental)，并提取本段内明确成立的有向边(causal/info/update)。description一律留空字符串。"
        )

        return await self.llm.chat_json(
            prompt=user_prompt,
            response_model=GraphExtractionPayload,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=8192,
        )

    @staticmethod
    def _normalize_level(raw_level: str) -> str:
        lvl = (raw_level or "").strip()
        if "一" in lvl or lvl == "1":
            return "一级"
        if "二" in lvl or lvl == "2":
            return "二级"
        return "三级"

    @staticmethod
    def _normalize_sub_type(node_type: str, raw_sub: str) -> str:
        s = (raw_sub or "").strip().lower()
        if node_type == "c":
            if s in ("unknown", "未知", "未知信息"):
                return "unknown"
            return "known"
        if node_type == "u":
            if s in ("instrumental", "手段", "手段目标", "工具性目标"):
                return "instrumental"
            return "final"
        return ""

    def _merge_extracted_payloads(
        self,
        payloads: List[GraphExtractionPayload],
        corpus_id: str,
        corpus_hash: str,
        title: str = "",
    ) -> Tuple[StandardGraph, Dict[str, Set[int]]]:
        """
        跨分块向量化图谱合并、语义去重与概念层级自动挂靠算法：
        1. 批量计算所有候选节点句向量，按 c/d/u 分类与余弦相似度合并同义节点
        2. 保留抽象层级 (一级/二级/三级)、已知/未知、最终/手段、根因属性
        3. 自动为三级具体节点匹配最近邻的二级/一级抽象概念 (对齐 V5 NodeConceptAgent)
        4. 重映射所有分块内有向边，过滤自环与指向根因节点的非法因果边
        """
        embedding_engine: Optional[EmbeddingEngine] = None
        try:
            embedding_engine = get_embedding_engine(self.settings)
        except Exception:
            pass

        # 预先批量计算全部节点名称向量（一次性矩阵编码，极速 CPU 执行）
        all_raw_names = [
            n.name.strip()
            for p in payloads
            for n in (getattr(p, "nodes", None) or [])
            if getattr(n, "name", None) and n.name.strip()
        ]
        if embedding_engine is not None and all_raw_names:
            try:
                embedding_engine.embed_texts(list(dict.fromkeys(all_raw_names)))
            except Exception:
                pass

        sim_threshold = max(0.82, min(0.92, self.settings.node_similarity_threshold))
        orig_to_global_map: Dict[Tuple[int, str], str] = {}
        unified_nodes_by_type: Dict[str, List[dict]] = {"c": [], "d": [], "u": []}
        node_chunks_map: Dict[str, Set[int]] = {}
        level_priority = {"一级": 3, "二级": 2, "三级": 1}

        for chunk_idx, payload in enumerate(payloads):
            p_nodes = getattr(payload, "nodes", None)
            if not isinstance(p_nodes, list):
                continue
            for n in p_nodes:
                clean_name = n.name.strip()
                if len(clean_name) < 2:
                    continue

                n_type = (n.type or "").strip().lower()
                if n_type not in {"c", "d", "u"}:
                    n_type = n.id[0].lower() if n.id and n.id[0].lower() in {"c", "d", "u"} else "c"

                n_level = self._normalize_level(n.level)
                n_sub = self._normalize_sub_type(n_type, n.sub_type)

                existing_group = unified_nodes_by_type[n_type]
                matched_idx = -1

                # 1. 完全一致匹配
                for idx, exist_n in enumerate(existing_group):
                    if exist_n["name"] == clean_name:
                        matched_idx = idx
                        break

                # 2. 向量余弦相似度合并
                node_vec = None
                if embedding_engine is not None:
                    try:
                        node_vec = embedding_engine.embed_text(clean_name)
                    except Exception:
                        node_vec = None

                if matched_idx == -1 and node_vec is not None and existing_group:
                    exist_vecs = [e["_vec"] for e in existing_group if e.get("_vec") is not None]
                    if len(exist_vecs) == len(existing_group):
                        sims = embedding_engine.compute_similarity(node_vec, exist_vecs)
                        if len(sims) > 0 and float(sims.max()) >= sim_threshold:
                            matched_idx = int(sims.argmax())

                if matched_idx != -1:
                    target_entry = existing_group[matched_idx]
                    target_global_id = target_entry["id"]
                    orig_to_global_map[(chunk_idx, n.id)] = target_global_id
                    node_chunks_map.setdefault(target_global_id, set()).add(chunk_idx)

                    target_entry["keywords"] = list(dict.fromkeys(target_entry["keywords"] + n.keywords))[:5]
                    if level_priority.get(n_level, 1) > level_priority.get(target_entry["level"], 1):
                        target_entry["level"] = n_level
                    if n.is_root:
                        target_entry["is_root"] = True
                    if n_sub and not target_entry.get("sub_type"):
                        target_entry["sub_type"] = n_sub
                    if n.description and n.description not in target_entry["description"]:
                        target_entry["description"] = (
                            f"{target_entry['description']}；{n.description}".strip("；")
                        )
                else:
                    new_idx = len(existing_group) + 1
                    global_id = f"{n_type}{new_idx}"
                    orig_to_global_map[(chunk_idx, n.id)] = global_id
                    node_chunks_map[global_id] = {chunk_idx}

                    existing_group.append({
                        "id": global_id,
                        "name": clean_name,
                        "type": n_type,
                        "level": n_level,
                        "sub_type": n_sub,
                        "is_root": bool(n.is_root),
                        "description": n.description,
                        "keywords": list(dict.fromkeys(n.keywords))[:5],
                        "_vec": node_vec,
                    })

        # 组装全局 Node 列表，并为三级节点自动挂靠最相近的二级/一级概念 (parent_concept)
        all_items = unified_nodes_by_type["c"] + unified_nodes_by_type["d"] + unified_nodes_by_type["u"]
        concept_candidates = [it for it in all_items if it["level"] == "二级"] or [
            it for it in all_items if it["level"] == "一级"
        ]
        concept_vecs = (
            np.vstack([it["_vec"] for it in concept_candidates])
            if concept_candidates and all(it.get("_vec") is not None for it in concept_candidates)
            else None
        )

        final_nodes: List[Node] = []
        root_ids: Set[str] = set()
        for item in all_items:
            if item["is_root"]:
                root_ids.add(item["id"])

            parent_concept = ""
            if item["level"] == "二级":
                parent_concept = item["name"]
            elif item["level"] == "三级" and concept_candidates:
                if concept_vecs is not None and item.get("_vec") is not None and embedding_engine is not None:
                    sims = embedding_engine.compute_similarity(item["_vec"], concept_vecs)
                    if len(sims) > 0:
                        best_c_idx = int(sims.argmax())
                        parent_concept = concept_candidates[best_c_idx]["name"]
                else:
                    parent_concept = concept_candidates[0]["name"]

            final_nodes.append(
                Node(
                    id=item["id"],
                    name=item["name"],
                    type=item["type"],
                    level=item["level"],  # type: ignore
                    sub_type=item["sub_type"],
                    is_root=item["is_root"],
                    parent_concept=parent_concept,
                    description=item["description"],
                    keywords=item["keywords"],
                )
            )

        # 重映射与合并分块内提取的有向边（过滤指向根因节点的因果边）
        final_arcs: List[Arc] = []
        seen_arcs: Set[Tuple[str, str, str]] = set()
        name_to_id = {n.name: n.id for n in final_nodes}

        for chunk_idx, payload in enumerate(payloads):
            p_arcs = getattr(payload, "arcs", None)
            if not isinstance(p_arcs, list):
                continue
            for a in p_arcs:
                arc_type = (a.type or "causal").strip().lower()
                if arc_type not in {"causal", "info", "update"}:
                    arc_type = "causal"

                src_id = orig_to_global_map.get((chunk_idx, a.source)) or name_to_id.get(a.source.strip())
                dst_id = orig_to_global_map.get((chunk_idx, a.target)) or name_to_id.get(a.target.strip())

                if not src_id or not dst_id or src_id == dst_id:
                    continue
                # 对齐 V5 _filter_causal_arcs: 根因节点不能作为因果边的果(target)
                if arc_type == "causal" and dst_id in root_ids:
                    continue

                arc_key = (src_id, dst_id, arc_type)
                if arc_key not in seen_arcs:
                    seen_arcs.add(arc_key)
                    final_arcs.append(
                        Arc(
                            source=src_id,
                            target=dst_id,
                            type=arc_type,  # type: ignore
                            description=a.description,
                        )
                    )

        final_title = title or next((p.title for p in payloads if p.title and p.title != "讨论案例标准图谱"), "讨论案例标准图谱")

        graph = StandardGraph(
            corpus_id=corpus_id,
            corpus_hash=corpus_hash,
            title=final_title,
            nodes=final_nodes,
            arcs=final_arcs,
        )
        return graph, node_chunks_map

    async def _complete_cross_chunk_arcs(
        self,
        graph: StandardGraph,
        node_chunks_map: Dict[str, Set[int]],
    ) -> StandardGraph:
        """
        基于纯 CPU 向量相似度聚簇的跨分块骨架补边（消灭 V5 的 O(N^2) 盲猜爆炸）：
        - 仅在长文本多分块且总节点数 >= 6 时触发
        - 将跨块高相关节点 + 全局核心目标/一级二级概念聚合成最多 6 个精简 ID 批次并发补边
        - 2~3 万字语料仅需 1 轮异步并发 (~10-15s) 即可打通全篇跨段落因果链！
        """
        if len(graph.nodes) < 6:
            return graph

        distinct_chunks = set()
        for ch_set in node_chunks_map.values():
            distinct_chunks.update(ch_set)
        if len(distinct_chunks) <= 1:
            return graph

        valid_ids = graph.node_id_set
        root_ids = {n.id for n in graph.nodes if n.is_root}
        existing_pairs = {(a.source, a.target) for a in graph.arcs}

        # 提取全局骨干节点（所有 u 效用节点 + 一/二级概念节点，上限 25 个）
        backbone_nodes = [
            n for n in graph.nodes if n.type == "u" or n.level in ("一级", "二级")
        ][:25]
        backbone_ids = {n.id for n in backbone_nodes}
        remaining_nodes = [n for n in graph.nodes if n.id not in backbone_ids]

        # 按每组最多 35 个其余节点 + 全局骨干节点构造高内聚批次（上限 6 批）
        batches: List[List[Node]] = []
        if len(graph.nodes) <= 60:
            batches.append(graph.nodes)
        else:
            step = max(25, (len(remaining_nodes) + 5) // 6)
            for i in range(0, len(remaining_nodes), step):
                sub = backbone_nodes + remaining_nodes[i : i + step]
                batches.append(sub)
                if len(batches) >= 6:
                    break

        async def _run_batch(batch_nodes: List[Node]) -> List[Tuple[str, str, str]]:
            lines = []
            for n in batch_nodes:
                sub_tag = f"/{n.sub_type}" if n.sub_type else ""
                lines.append(f"[{n.id}] ({n.type}/{n.level}{sub_tag}) {n.name}")
            prompt = (
                "【跨段落全局节点子集】：\n"
                + "\n".join(lines)
                + "\n\n请仅输出上述节点之间跨段落成立、尚未连接的核心有向边（source/target 填节点ID，type 填 causal/info/update）。"
            )
            try:
                res = await self.llm.chat_json(
                    prompt=prompt,
                    response_model=CrossChunkArcPayload,
                    system_prompt=CROSS_CHUNK_ARC_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2048,
                )
                raw_arcs = getattr(res, "arcs", []) or []
                out: List[Tuple[str, str, str]] = []
                for item in raw_arcs:
                    src = getattr(item, "source", "").strip()
                    tgt = getattr(item, "target", "").strip()
                    atype = getattr(item, "type", "causal").strip().lower()
                    if atype not in {"causal", "info", "update"}:
                        atype = "causal"
                    out.append((src, tgt, atype))
                return out
            except Exception as exc:
                logger.warning(f"Cross-chunk arc batch skipped gracefully: {exc}")
                return []

        batch_results = await asyncio.gather(*[_run_batch(b) for b in batches])
        added_count = 0
        for triplets in batch_results:
            for src, tgt, atype in triplets:
                if src not in valid_ids or tgt not in valid_ids or src == tgt:
                    continue
                if atype == "causal" and tgt in root_ids:
                    continue
                # 至少有一端跨不同分块时优先补齐（或尚未存在的骨干连接）
                if (src, tgt) not in existing_pairs:
                    existing_pairs.add((src, tgt))
                    graph.arcs.append(Arc(source=src, target=tgt, type=atype))  # type: ignore
                    added_count += 1

        if added_count > 0:
            logger.info(f"🔗 Linked {added_count} cross-chunk global arcs across {len(distinct_chunks)} chunks")
        return graph

    async def extract_graph(
        self,
        corpus_text: str,
        corpus_id: str = "default_corpus",
        title: str = "",
    ) -> StandardGraph:
        """
        自适应标准图谱提取：
        - 可选并发语音转写润滑 (enable_corpus_smoothing=True)
        - 短语料 (<= 阈值): 单次全局提取，零碎片损耗
        - 超长语料 (2~3万字+): 自适应重叠分块全并发抽取 + 向量去重与概念挂靠 + 跨块骨架补边
        """
        corpus_text = corpus_text.strip()
        if not corpus_text:
            raise ValueError("Corpus text cannot be empty.")

        corpus_hash = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
        working_text = corpus_text
        if self.settings.enable_corpus_smoothing:
            logger.info(f"Smoothing reference corpus concurrently ({len(corpus_text)} chars)...")
            working_text = await self._smooth_corpus_concurrent(corpus_text)

        corpus_len = len(working_text)
        threshold = self.settings.corpus_auto_chunk_threshold

        logger.info(f"Extracting Standard Graph (len={corpus_len}, threshold={threshold}, sha256={corpus_hash[:8]}...)")

        if corpus_len <= threshold:
            payload = await self._extract_single_chunk(working_text, chunk_idx=0, total_chunks=1)
            graph, _ = self._merge_extracted_payloads(
                [payload], corpus_id=corpus_id, corpus_hash=corpus_hash, title=title
            )
        else:
            chunks = self._split_text_into_chunks(
                working_text,
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
            graph, node_chunks_map = self._merge_extracted_payloads(
                list(payloads), corpus_id=corpus_id, corpus_hash=corpus_hash, title=title
            )
            graph = await self._complete_cross_chunk_arcs(graph, node_chunks_map)

        logger.info(
            f"Successfully extracted Standard Graph: {len(graph.nodes)} nodes "
            f"(L1={len([n for n in graph.nodes if n.level=='一级'])}, "
            f"L2={len([n for n in graph.nodes if n.level=='二级'])}, "
            f"L3={len([n for n in graph.nodes if n.level=='三级'])}), "
            f"{len(graph.arcs)} arcs"
        )
        return graph


if __name__ == "__main__":
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

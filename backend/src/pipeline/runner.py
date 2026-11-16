# -*- coding: utf-8 -*-
"""
End-to-End Discussion Graph Pipeline Runner.
Orchestrates:
1. Standard Graph Retrieval (0s hit via CacheManager)
2. Discussion Transcript Parsing (Timestamp & Speaker cleanup)
3. Interaction Network Extraction (Support/Denial dynamics)
4. Multi-Speaker Node Mention Sniffing (Embedding + Concurrent Batch Matching)
5. Dual-Pruned Arc Verification (Pruned $O(V)$ rather than $O(V^2)$)
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from configs.settings import Settings, get_settings
from src.core.embedding import EmbeddingEngine, get_embedding_engine
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.knowledge.cache import GraphCacheManager, get_graph_cache_manager
from src.models.discussion import DiscussionTranscript, InteractionNetwork
from src.models.graph import CandidateGraph, StandardGraph
from src.pipeline.arc_verifier import CandidateArcVerifier
from src.pipeline.interaction import InteractionExtractor
from src.pipeline.node_matcher import CandidateNodeMatcher
from src.pipeline.parser import DiscussionParser

logger = get_logger("pipeline_runner")


class PipelineResult(BaseModel):
    """流水线产出的全量结构化图谱与讨论数据"""
    group_id: str
    corpus_id: str
    standard_graph: StandardGraph
    transcript: DiscussionTranscript
    candidate_graphs: Dict[str, CandidateGraph] = Field(default_factory=dict)
    interaction_network: InteractionNetwork
    elapsed_seconds: float = Field(default=0.0, description="Pipeline 全程总耗时(秒)")


class PipelineRunner:
    """全自动端到端图谱解析流水线调度器"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        cache_manager: Optional[GraphCacheManager] = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)
        self.embedding = embedding_engine or get_embedding_engine(self.settings)
        self.cache_manager = cache_manager or get_graph_cache_manager(self.settings)

        self.parser = DiscussionParser(host_speaker_id=self.settings.host_speaker_id)
        self.interaction_extractor = InteractionExtractor(llm_client=self.llm, settings=self.settings)
        self.node_matcher = CandidateNodeMatcher(
            llm_client=self.llm,
            embedding_engine=self.embedding,
            settings=self.settings,
        )
        self.arc_verifier = CandidateArcVerifier(llm_client=self.llm, settings=self.settings)

    async def run(
        self,
        corpus_text: str,
        discussion_text: str,
        group_id: str = "default_group",
        corpus_id: str = "default_corpus",
        title: str = "无领导小组讨论",
        force_refresh_corpus: bool = False,
    ) -> PipelineResult:
        """
        运行完整 Pipeline：
        1. 获取/提取标准图谱 (带 0s 缓存)
        2. 清洗讨论文本
        3. 提取全场人际互动网络
        4. 并发匹配所有人言论节点
        5. 并发校验所有人的逻辑因果边
        """
        start_time = time.perf_counter()
        logger.info(f"🚀 [PIPELINE START] Processing group={group_id}, corpus={corpus_id}")

        # Step 1: 加载标准图谱 (若已存在则 0 秒返回)
        standard_graph = await self.cache_manager.get_or_extract(
            corpus_text=corpus_text,
            corpus_id=corpus_id,
            title=title,
            force_refresh=force_refresh_corpus,
        )

        # Step 2: 解析讨论文本
        transcript = self.parser.parse_text(
            raw_text=discussion_text,
            group_id=group_id,
            title=title,
        )
        candidate_speakers = transcript.candidate_speakers
        logger.info(f"Parsed {len(candidate_speakers)} candidates from transcript")

        # Step 3: 并行发起：全场互动提取 + 各候选人节点匹配
        interaction_task = self.interaction_extractor.extract_interactions(transcript)
        node_match_task = self.node_matcher.match_nodes_for_all_speakers(
            speakers=candidate_speakers,
            standard_graph=standard_graph,
            max_concurrency=self.settings.llm_max_concurrency,
        )

        interaction_network, candidate_graphs = await asyncio.gather(
            interaction_task,
            node_match_task,
        )

        # Step 4: 并行校验各候选人的逻辑边 (基于双端命中的极速剪枝)
        arc_verify_task = self.arc_verifier.verify_arcs_for_all_speakers(
            speakers=candidate_speakers,
            candidate_graphs=candidate_graphs,
            standard_graph=standard_graph,
            max_concurrency=self.settings.llm_max_concurrency,
        )
        await arc_verify_task

        total_elapsed = time.perf_counter() - start_time
        logger.info(
            f"✅ [PIPELINE COMPLETE] Group {group_id} finished in {total_elapsed:.2f}s across {len(candidate_speakers)} candidates!"
        )

        return PipelineResult(
            group_id=group_id,
            corpus_id=corpus_id,
            standard_graph=standard_graph,
            transcript=transcript,
            candidate_graphs=candidate_graphs,
            interaction_network=interaction_network,
            elapsed_seconds=round(total_elapsed, 3),
        )

    async def run_files(
        self,
        corpus_path: Union[str, Path],
        discussion_path: Union[str, Path],
        group_id: Optional[str] = None,
        corpus_id: Optional[str] = None,
        title: str = "无领导小组讨论",
        force_refresh_corpus: bool = False,
    ) -> PipelineResult:
        c_path = Path(corpus_path)
        d_path = Path(discussion_path)

        if not c_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {c_path}")
        if not d_path.exists():
            raise FileNotFoundError(f"Discussion file not found: {d_path}")

        c_text = DiscussionParser.read_text_with_fallback(c_path)
        d_text = DiscussionParser.read_text_with_fallback(d_path)

        gid = group_id or d_path.stem
        cid = corpus_id or c_path.stem

        return await self.run(
            corpus_text=c_text,
            discussion_text=d_text,
            group_id=gid,
            corpus_id=cid,
            title=title,
            force_refresh_corpus=force_refresh_corpus,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Talent Graph Pipeline")
    parser.add_argument("--corpus", required=True, help="Path to corpus.txt")
    parser.add_argument("--group", required=True, help="Path to group_*.txt transcript")
    parser.add_argument("--output", default="pipeline_result.json", help="Path to save result JSON")
    args = parser.parse_args()

    runner = PipelineRunner()
    res = asyncio.run(runner.run_files(args.corpus, args.group))
    Path(args.output).write_text(res.model_dump_json(indent=2), encoding="utf-8")
    print(f"Pipeline finished in {res.elapsed_seconds}s. Output saved to {args.output}")

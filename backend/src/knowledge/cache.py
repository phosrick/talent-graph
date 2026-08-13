# -*- coding: utf-8 -*-
"""
Knowledge Graph Local / Disk Cache Manager.
Enables 0-second instant graph retrieval when corpus text has already been parsed.
"""

import hashlib
from pathlib import Path
from typing import Optional, Union
from configs.settings import Settings, get_settings
from src.core.logger import get_logger
from src.knowledge.extractor import StandardGraphExtractor
from src.models.graph import StandardGraph

logger = get_logger("graph_cache")


class GraphCacheManager:
    """标准题库图谱缓存管理器"""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        extractor: Optional[StandardGraphExtractor] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir or self.settings.get_corpus_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor or StandardGraphExtractor(settings=self.settings)

    def _compute_hash(self, corpus_text: str) -> str:
        """计算语料文本的 SHA256 唯一指纹"""
        return hashlib.sha256(corpus_text.strip().encode("utf-8")).hexdigest()

    def get_cache_path(self, corpus_hash: str) -> Path:
        return self.cache_dir / f"{corpus_hash}.json"

    def has_cache(self, corpus_text: str) -> bool:
        """检查语料文本是否已有缓存"""
        c_hash = self._compute_hash(corpus_text)
        return self.get_cache_path(c_hash).exists()

    def load_from_cache(self, corpus_text: str) -> Optional[StandardGraph]:
        """从本地磁盘缓存加载图谱（0毫秒）"""
        c_hash = self._compute_hash(corpus_text)
        cache_file = self.get_cache_path(c_hash)
        if cache_file.exists():
            try:
                data = cache_file.read_text(encoding="utf-8")
                graph = StandardGraph.model_validate_json(data)
                logger.info(f"⚡ [CACHE HIT] Loaded Standard Graph instantly for hash {c_hash[:8]} ({len(graph.nodes)} nodes, {len(graph.arcs)} arcs)")
                return graph
            except Exception as exc:
                logger.warning(f"Corrupted cache file at {cache_file}, will re-extract: {exc}")
                return None
        return None

    def save_to_cache(self, graph: StandardGraph) -> Path:
        """将标准图谱保存至磁盘缓存"""
        if not graph.corpus_hash:
            # 如果未设置，根据节点与边计算一个标识
            graph.corpus_hash = hashlib.sha256(graph.model_dump_json().encode("utf-8")).hexdigest()

        cache_file = self.get_cache_path(graph.corpus_hash)
        cache_file.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved Standard Graph cache to {cache_file}")
        return cache_file

    async def get_or_extract(
        self,
        corpus_text: str,
        corpus_id: str = "default_corpus",
        title: str = "",
        force_refresh: bool = False,
    ) -> StandardGraph:
        """
        获取标准图谱核心入口：若命中缓存则 0 秒返回，否则发起 LLM 异步提取并缓存。
        """
        if not force_refresh:
            cached = self.load_from_cache(corpus_text)
            if cached is not None:
                return cached

        # 缓存未命中，触发提取
        graph = await self.extractor.extract_graph(corpus_text, corpus_id=corpus_id, title=title)
        self.save_to_cache(graph)
        return graph


_global_graph_cache: Optional[GraphCacheManager] = None


def get_graph_cache_manager(settings: Optional[Settings] = None) -> GraphCacheManager:
    global _global_graph_cache
    if _global_graph_cache is None:
        _global_graph_cache = GraphCacheManager(settings=settings)
    return _global_graph_cache

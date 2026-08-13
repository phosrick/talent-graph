# -*- coding: utf-8 -*-
"""Unit tests for CPU Embedding Engine"""

import time
import numpy as np
from src.core.embedding import EmbeddingEngine, get_embedding_engine


def test_embedding_shape_and_norm():
    engine = get_embedding_engine()
    vec = engine.embed_text("无领导小组讨论人才评估")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert vec.dtype == np.float32

    # 模长应近似为 1.0 (归一化)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_batch_embedding_and_similarity():
    engine = get_embedding_engine()
    texts = [
        "我们需要制定一个详细的线上营销策略",
        "线上营销方案和推广计划至关重要",
        "今天中午吃什么外卖好呢",
    ]
    vecs = engine.embed_texts(texts)
    assert vecs.shape == (3, 512)

    # 前两条高度语义相关，相似度应显著高于第三条无关内容
    sim_1_2 = engine.cosine_similarity(vecs[0], vecs[1])
    sim_1_3 = engine.cosine_similarity(vecs[0], vecs[2])
    assert sim_1_2 > sim_1_3
    assert sim_1_2 > 0.2


def test_find_most_similar():
    engine = get_embedding_engine()
    query = "降低产品生产与运营成本"
    candidates = [
        "提升员工福利待遇",
        "削减开支以降低产品成本和经营开销",
        "开展海外品牌营销推广",
        "招募研发算法工程师",
    ]
    results = engine.find_most_similar(query, candidates, top_k=2)
    assert len(results) == 2
    # 最匹配的应是 candidate 1 (削减开支以降低产品成本...)
    top_idx, top_score = results[0]
    assert top_idx == 1
    assert top_score > 0.2


def test_embedding_performance_benchmark():
    """测试纯 CPU 向量计算耗时 (< 2ms)"""
    engine = get_embedding_engine()
    text = "在本次讨论中，2号候选人提出了清晰的目标拆解路径，并协调了团队分歧达成共识。"

    # 预热
    engine.embed_text(text)

    start = time.perf_counter()
    for _ in range(100):
        engine.embed_text(f"测试文本 {_} " + text)
    duration_ms = (time.perf_counter() - start) * 1000 / 100

    print(f"\nAverage Embedding Latency per text: {duration_ms:.3f} ms")
    assert duration_ms < 2.0, f"Latency {duration_ms}ms exceeded 2ms target"


def test_compute_similarity():
    """测试 compute_similarity 兼容单/多向量点乘"""
    engine = get_embedding_engine()
    q = engine.embed_text("制定预算方案")
    docs = [
        engine.embed_text("制定资金预算与开支方案"),
        engine.embed_text("中午去吃汉堡"),
    ]
    sims = engine.compute_similarity(q, docs)
    assert isinstance(sims, np.ndarray)
    assert len(sims) == 2
    assert sims[0] > sims[1]


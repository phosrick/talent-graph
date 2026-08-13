# -*- coding: utf-8 -*-
"""
Unit Tests for Module 4: 16 Pure-Mathematical Quant Scorers and In-Memory Scoring Engine.
"""

import time
import pytest
from src.core.embedding import EmbeddingEngine
from src.models.discussion import DiscussionTranscript, InteractionEdge, InteractionNetwork, Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, NodeMention, StandardGraph
from src.models.score import ALL_DIMENSIONS
from src.scoring.base import ScoringContext
from src.scoring.engine import ScoringEngine


@pytest.fixture
def mock_scoring_context():
    # 构造标准图谱
    nodes = [
        Node(id="c1", name="宏观市场背景", type="c", description="经济环境", keywords=["市场"]),
        Node(id="c2", name="客户核心痛点", type="c", description="需求不足", keywords=["痛点"]),
        Node(id="d1", name="推出创新产品方案", type="d", description="新产品", keywords=["方案"]),
        Node(id="d2", name="实施成本控制战略", type="d", description="降本", keywords=["成本"]),
        Node(id="u1", name="利润率提升20%", type="u", description="盈利目标", keywords=["利润"]),
        Node(id="u2", name="品牌美誉度提升", type="u", description="品牌效应", keywords=["品牌"]),
    ]
    arcs = [
        Arc(source="c1", target="d1", type="info", description="市场驱动方案"),
        Arc(source="c2", target="d1", type="info", description="痛点催生方案"),
        Arc(source="d1", target="u1", type="causal", description="方案达成利润"),
        Arc(source="d2", target="u1", type="causal", description="降本达成利润"),
        Arc(source="d1", target="u2", type="causal", description="方案塑造品牌"),
    ]
    std_graph = StandardGraph(corpus_id="corp_1", title="商业实战案例", nodes=nodes, arcs=arcs)

    # 构造 3 位候选人的发言与画像
    u_spk2_1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="基于宏观市场背景与客户痛点，我建议推出创新产品方案！", start_time_str="00:10", start_seconds=10.0)
    u_spk2_2 = Utterance(index=2, speaker_id="speaker_2", speaker_name="候选人 2", text="新方案不仅能让利润率提升20%，还能带来品牌美誉度提升。", start_time_str="02:30", start_seconds=150.0)
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u_spk2_1, u_spk2_2], is_host=False, total_duration_seconds=90.0, turn_count=2)

    u_spk3_1 = Utterance(index=1, speaker_id="speaker_3", speaker_name="候选人 3", text="我赞同2号，同时我们必须实施成本控制战略确保盈利。", start_time_str="01:00", start_seconds=60.0)
    spk3 = Speaker(speaker_id="speaker_3", display_name="候选人 3", utterances=[u_spk3_1], is_host=False, total_duration_seconds=45.0, turn_count=1)

    u_spk4_1 = Utterance(index=3, speaker_id="speaker_4", speaker_name="候选人 4", text="我们还可以尝试私域裂变运营，开辟新增长极。", start_time_str="03:00", start_seconds=180.0)
    spk4 = Speaker(speaker_id="speaker_4", display_name="候选人 4", utterances=[u_spk4_1], is_host=False, total_duration_seconds=40.0, turn_count=1)

    transcript = DiscussionTranscript(
        group_id="group_test",
        title="商业实战讨论",
        speakers={"speaker_2": spk2, "speaker_3": spk3, "speaker_4": spk4},
        utterances=[u_spk2_1, u_spk3_1, u_spk2_2, u_spk4_1],
        total_duration_seconds=220.0,
    )

    # 候选人个人图谱
    cg2 = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=[nodes[0], nodes[1], nodes[2], nodes[4], nodes[5]],
        node_mentions=[
            NodeMention(node_id="c1", utterance_index=0, timestamp_seconds=10.0, quote="宏观市场背景"),
            NodeMention(node_id="c2", utterance_index=0, timestamp_seconds=10.0, quote="客户痛点"),
            NodeMention(node_id="d1", utterance_index=0, timestamp_seconds=10.0, quote="推出创新产品方案"),
            NodeMention(node_id="u1", utterance_index=2, timestamp_seconds=150.0, quote="利润率提升20%"),
            NodeMention(node_id="u2", utterance_index=2, timestamp_seconds=150.0, quote="品牌美誉度提升"),
        ],
        verified_arcs=[arcs[0], arcs[1], arcs[2], arcs[4]],
        novel_nodes=[],
    )

    cg3 = CandidateGraph(
        speaker_id="speaker_3",
        matched_nodes=[nodes[3], nodes[4]],
        node_mentions=[
            NodeMention(node_id="d2", utterance_index=1, timestamp_seconds=60.0, quote="实施成本控制战略"),
            NodeMention(node_id="u1", utterance_index=1, timestamp_seconds=60.0, quote="确保盈利"),
        ],
        verified_arcs=[arcs[3]],
        novel_nodes=[],
    )

    cg4 = CandidateGraph(
        speaker_id="speaker_4",
        matched_nodes=[],
        node_mentions=[],
        verified_arcs=[],
        novel_nodes=[Node(id="novel_spk4_1", name="私域裂变运营", type="d", description="新获客路径")],
    )

    # 互动网络
    interaction = InteractionNetwork(
        speakers=["speaker_2", "speaker_3", "speaker_4"],
        edges=[
            InteractionEdge(source="speaker_3", target="speaker_2", type="support", weight=1.0, evidence="我赞同2号"),
            InteractionEdge(source="speaker_4", target="speaker_2", type="neutral", weight=1.0, evidence="我们还可以尝试"),
        ],
    )

    return ScoringContext(
        group_id="group_test",
        corpus_id="corp_1",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs={"speaker_2": cg2, "speaker_3": cg3, "speaker_4": cg4},
        interaction_network=interaction,
        embedding_engine=None,
    )


def test_scoring_engine_full_run_and_latency(mock_scoring_context):
    engine = ScoringEngine()

    start_t = time.perf_counter()
    matrix = engine.evaluate_all(mock_scoring_context)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    # 性能断言：纯内存计算必须 < 50ms
    assert elapsed_ms < 50.0

    # 完整性断言：包含所有 3 位候选人
    assert len(matrix.participants) == 3
    assert "speaker_2" in matrix.participants
    assert "speaker_3" in matrix.participants
    assert "speaker_4" in matrix.participants

    # 维度对齐断言：每个候选人都必须具备完整的 16 个维度得分
    expected_dim_ids = {d.id for d in ALL_DIMENSIONS}
    assert len(expected_dim_ids) == 16

    for spk_id, profile in matrix.participants.items():
        assert set(profile.dimension_scores.keys()) == expected_dim_ids
        for dim_id, d_score in profile.dimension_scores.items():
            assert 60.0 <= d_score.score <= 95.0
            assert isinstance(d_score.key_points, list)
            assert isinstance(d_score.metrics, dict)

    # 排名与均值计算断言
    assert matrix.team_average_total_score > 0
    assert len(matrix.dimension_averages) == 16
    assert len(matrix.dimension_max_scores) == 16
    assert len(matrix.dimension_min_scores) == 16

    # 检查综合排名前后
    ranks = [p.overall_rank for p in matrix.participants.values()]
    assert set(ranks) == {1, 2, 3}


def test_collaboration_centrality_support_ranking(mock_scoring_context):
    engine = ScoringEngine()
    matrix = engine.evaluate_all(mock_scoring_context)
    s2_53 = matrix.participants["speaker_2"].dimension_scores["5_3"].score
    s3_53 = matrix.participants["speaker_3"].dimension_scores["5_3"].score
    assert s2_53 > s3_53, f"Speaker 2 with incoming support ({s2_53}) should score higher than Speaker 3 ({s3_53}) on 5_3"


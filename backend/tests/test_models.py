# -*- coding: utf-8 -*-
"""Unit tests for Domain Models"""

import networkx as nx
from src.models import (
    ALL_DIMENSIONS,
    DIMENSION_MAP,
    Arc,
    CandidateGraph,
    ComprehensiveDiagnosis,
    DiscussionTranscript,
    InteractionEdge,
    InteractionNetwork,
    Node,
    ParticipantReport,
    ParticipantScoreProfile,
    RadarPoint,
    ReportData,
    ScoreMatrix,
    Speaker,
    StandardGraph,
    Utterance,
)


def test_16_dimensions_alignment():
    """验证 16 个能力维度定义是否完整且与老版本 100% 对齐"""
    assert len(ALL_DIMENSIONS) == 16
    expected_ids = {
        "1_1", "1_2", "1_3",
        "2_1", "2_2", "2_3",
        "3_1", "3_2", "3_3",
        "4",
        "5_1", "5_2", "5_3",
        "6_1", "6_2", "6_3",
    }
    actual_ids = set(DIMENSION_MAP.keys())
    assert actual_ids == expected_ids


def test_graph_networkx_roundtrip():
    """测试 StandardGraph 与 NetworkX 双向转换与图论属性"""
    nodes = [
        Node(id="c1", name="市场现状", type="c", description="事实描述"),
        Node(id="d1", name="推出新产品", type="d", description="决策方案"),
        Node(id="u1", name="提升市占率", type="u", description="战略目标"),
    ]
    arcs = [
        Arc(source="c1", target="d1", type="info"),
        Arc(source="d1", target="u1", type="causal"),
    ]
    graph = StandardGraph(corpus_id="test_corp", title="测试案例", nodes=nodes, arcs=arcs)

    G = graph.to_networkx()
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes) == 3
    assert len(G.edges) == 2
    assert nx.has_path(G, "c1", "u1")

    # 还原
    restored = StandardGraph.from_networkx(G, corpus_id="test_corp", title="测试案例")
    assert len(restored.nodes) == 3
    assert len(restored.arcs) == 2
    assert restored.get_node("d1").type == "d"


def test_discussion_models():
    """测试发言转录与互动网络构建"""
    u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="说话人 2", text="我认为我们应该先开拓线上渠道", start_time_str="01:10")
    u2 = Utterance(index=1, speaker_id="speaker_3", speaker_name="说话人 3", text="非常赞同2号的观点，线上成本更低", start_time_str="01:30")

    assert u1.word_count > 0

    edge = InteractionEdge(source="speaker_3", target="speaker_2", type="support", weight=1.0, evidence="非常赞同2号的观点")
    net = InteractionNetwork(speakers=["speaker_2", "speaker_3"], edges=[edge])
    G = net.to_networkx()

    assert G.has_edge("speaker_3", "speaker_2")
    assert G["speaker_3"]["speaker_2"]["weight"] == 1.0


def test_score_matrix_aggregations():
    """测试评分矩阵的均分、极值与排名计算"""
    matrix = ScoreMatrix(group_id="group_271")
    p1 = ParticipantScoreProfile(speaker_id="speaker_1", total_score=85.0)
    p2 = ParticipantScoreProfile(speaker_id="speaker_2", total_score=92.0)

    for dim in ALL_DIMENSIONS:
        from src.models.score import DimensionScore
        p1.dimension_scores[dim.id] = DimensionScore(dimension_id=dim.id, dimension_name=dim.name, score=80.0)
        p2.dimension_scores[dim.id] = DimensionScore(dimension_id=dim.id, dimension_name=dim.name, score=90.0)

    matrix.participants["speaker_1"] = p1
    matrix.participants["speaker_2"] = p2
    matrix.compute_aggregations()

    assert matrix.dimension_averages["1_1"] == 85.0
    assert matrix.dimension_max_scores["1_1"] == 90.0
    assert matrix.participants["speaker_2"].overall_rank == 1
    assert matrix.participants["speaker_1"].overall_rank == 2


def test_report_data_serialization():
    """测试前后端统一 ReportData 序列化契约"""
    report = ReportData(
        task_id="task_123",
        group_id="271",
        participants=[
            ParticipantReport(
                speaker_id="speaker_2",
                display_name="候选人 2",
                total_score=88.5,
                overall_rank=1,
                radar_data=[
                    RadarPoint(dimension_id="1_1", dimension_name="逻辑思维", score=90.0, team_avg=82.0)
                ],
                diagnosis=ComprehensiveDiagnosis(
                    summary="表现卓越，逻辑清晰",
                    strengths=["善于归纳总结", "主导讨论收敛"],
                    weaknesses=["偶有急躁"],
                    growth_suggestions=["多倾听少数人意见"],
                    tags=["核心主导者"],
                ),
            )
        ]
    )

    json_str = report.model_dump_json()
    assert "task_123" in json_str
    assert "核心主导者" in json_str

    restored = ReportData.model_validate_json(json_str)
    assert restored.get_participant("speaker_2").total_score == 88.5

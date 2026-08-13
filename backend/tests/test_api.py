# -*- coding: utf-8 -*-
"""
Unit Tests for Module 5: Diagnostic Summarizer, Report Generator, and FastAPI Endpoints.
"""

from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import pytest
from server import app
from src.core.llm_client import LLMClient
from src.models.discussion import DiscussionTranscript, InteractionNetwork, Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, StandardGraph
from src.models.report import ComprehensiveDiagnosis, ReportData
from src.models.score import ScoreMatrix
from src.pipeline.runner import PipelineResult
from src.reporting.generator import ReportGenerator
from src.reporting.summarizer import CandidateSummarizer
from src.scoring.base import ScoringContext
from src.scoring.engine import ScoringEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "talent_graph"


@pytest.mark.asyncio
async def test_candidate_summarizer_heuristic_and_mock():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=ComprehensiveDiagnosis(
            summary="候选人表现沉稳，逻辑严密",
            strengths=["系统思维全面", "主导核心议题收敛"],
            weaknesses=["缺少对细节成本的考量"],
            growth_suggestions=["多关注底层执行边界"],
            tags=["战略领航者", "严谨推演派"],
        )
    )

    summarizer = CandidateSummarizer(llm_client=mock_llm)

    u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="我的方案如下...", start_time_str="01:00", start_seconds=60.0)
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)

    matrix = ScoreMatrix(group_id="group_1")
    engine = ScoringEngine()
    std_graph = StandardGraph(corpus_id="c1", title="案例", nodes=[Node(id="c1", name="节点", type="c")], arcs=[])
    transcript = DiscussionTranscript(group_id="group_1", title="讨论", speakers={"speaker_2": spk2}, utterances=[u1])
    interaction = InteractionNetwork(speakers=["speaker_2"], edges=[])

    context = ScoringContext(
        group_id="group_1",
        corpus_id="c1",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs={"speaker_2": CandidateGraph(speaker_id="speaker_2", matched_nodes=[std_graph.nodes[0]])},
        interaction_network=interaction,
    )
    score_matrix = engine.evaluate_all(context)

    diag = await summarizer.generate_diagnosis(spk2, score_matrix.participants["speaker_2"], score_matrix)
    assert len(diag.strengths) == 2
    assert "战略领航者" in diag.tags
    assert len(diag.summary) > 0


@pytest.mark.asyncio
async def test_report_generator_end_to_end(tmp_path):
    u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="我们应当快速行动。", start_time_str="01:00", start_seconds=60.0)
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)
    std_node = Node(id="c1", name="背景", type="c")
    std_graph = StandardGraph(corpus_id="c1", title="案例大纲", nodes=[std_node], arcs=[])
    transcript = DiscussionTranscript(group_id="test_g", title="测试讨论", speakers={"speaker_2": spk2}, utterances=[u1])
    interaction = InteractionNetwork(speakers=["speaker_2"], edges=[])

    pipeline_res = PipelineResult(
        group_id="test_g",
        corpus_id="c1",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs={"speaker_2": CandidateGraph(speaker_id="speaker_2", matched_nodes=[std_node])},
        interaction_network=interaction,
        elapsed_seconds=1.2,
    )

    mock_summarizer = MagicMock(spec=CandidateSummarizer)
    mock_summarizer.summarize_all_participants = AsyncMock(
        return_value={"speaker_2": ComprehensiveDiagnosis(summary="整体优秀", strengths=["表现优秀"], tags=["领航者"])}
    )

    generator = ReportGenerator(summarizer=mock_summarizer)
    report_data: ReportData = await generator.generate_report(pipeline_res, task_id="task_unit_test")

    assert report_data.task_id == "task_unit_test"
    assert report_data.team_overview.participant_count == 1
    assert len(report_data.participants) == 1
    assert report_data.participants[0].speaker_id == "speaker_2"
    assert len(report_data.participants[0].radar_data) == 16


def test_assessment_endpoints(client):
    """测试评估提交与数据获取端点"""
    submit_resp = client.post(
        "/api/v1/assessment/submit",
        json={
            "corpus_text": "这是一个测试案例参考语料",
            "discussion_text": "说话人 1 00:01 我们开始讨论方案。\n说话人 2 00:15 我同意这个方案。",
            "group_id": "test_g1",
            "title": "测试评估"
        }
    )
    assert submit_resp.status_code == 200
    task_id = submit_resp.json()["task_id"]
    assert task_id.startswith("task_")

    # 查询任务状态
    status_resp = client.get(f"/api/v1/assessment/{task_id}")
    assert status_resp.status_code == 200

    # 检查打印模板路由
    print_resp = client.get(f"/report-print/{task_id}/speaker_2")
    assert print_resp.status_code == 200


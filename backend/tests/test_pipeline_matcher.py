# -*- coding: utf-8 -*-
"""
Unit Tests for Module 3: Node Matching, Dual-Pruned Arc Verification, and Pipeline Runner.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from src.core.embedding import EmbeddingEngine
from src.core.llm_client import LLMClient
from src.knowledge.cache import GraphCacheManager
from src.models.discussion import DiscussionTranscript, Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, StandardGraph
from src.pipeline.arc_verifier import ArcVerificationItem, ArcVerificationPayload, CandidateArcVerifier
from src.pipeline.node_matcher import CandidateNodeMatcher, MatchedNodeItem, NodeMatchPayload, NovelNodeItem
from src.pipeline.runner import PipelineRunner


@pytest.fixture
def mock_standard_graph():
    return StandardGraph(
        corpus_id="test_case",
        title="测试案例",
        nodes=[
            Node(id="c1", name="市场竞争激烈", type="c", description="行业红海", keywords=["竞争", "市场"]),
            Node(id="c2", name="预算紧张", type="c", description="资金有限", keywords=["预算", "资金"]),
            Node(id="d1", name="开发AI新功能", type="d", description="推出智能化模块", keywords=["AI", "新功能", "开发"]),
            Node(id="d2", name="开展全员降本", type="d", description="缩减运营成本", keywords=["降本", "缩减"]),
            Node(id="u1", name="提升市场占有率", type="u", description="增加用户与份额", keywords=["占有率", "份额"]),
        ],
        arcs=[
            Arc(source="c1", target="d1", type="info", description="市场竞争驱动AI创新"),
            Arc(source="d1", target="u1", type="causal", description="AI功能拉动份额增长"),
            Arc(source="c2", target="d2", type="info", description="预算紧张推动降本"),
        ],
    )


@pytest.fixture
def mock_speakers():
    u1 = Utterance(
        index=0,
        speaker_id="speaker_2",
        speaker_name="候选人 2",
        text="我认为目前市场竞争很激烈，我们应当尽快开发AI新功能，从而有效提升市场占有率！",
        start_time_str="01:00",
        start_seconds=60.0,
    )
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)

    u2 = Utterance(
        index=1,
        speaker_id="speaker_3",
        speaker_name="候选人 3",
        text="我完全赞同2号。面对当前预算紧张的现实，我们要精打细算。",
        start_time_str="01:30",
        start_seconds=90.0,
    )
    spk3 = Speaker(speaker_id="speaker_3", display_name="候选人 3", utterances=[u2], is_host=False)

    return [spk2, spk3]


@pytest.mark.asyncio
async def test_node_matcher_llm_mock(mock_standard_graph, mock_speakers):
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=NodeMatchPayload(
            matched_nodes=[
                MatchedNodeItem(node_id="c1", utterance_index=0, quote="市场竞争很激烈", confidence=0.95),
                MatchedNodeItem(node_id="d1", utterance_index=0, quote="尽快开发AI新功能", confidence=0.95),
                MatchedNodeItem(node_id="u1", utterance_index=0, quote="提升市场占有率", confidence=0.9),
            ],
            novel_nodes=[
                NovelNodeItem(name="成立敏捷突击队", type="d", description="快速试错机制", quote="组建小团队"),
            ],
        )
    )

    matcher = CandidateNodeMatcher(llm_client=mock_llm)
    cand_graph = await matcher.match_nodes_for_speaker(mock_speakers[0], mock_standard_graph)

    assert len(cand_graph.matched_nodes) == 3
    assert "c1" in cand_graph.matched_node_ids
    assert "d1" in cand_graph.matched_node_ids
    assert "u1" in cand_graph.matched_node_ids
    assert len(cand_graph.node_mentions) == 3
    assert len(cand_graph.novel_nodes) == 1
    assert cand_graph.novel_nodes[0].name == "成立敏捷突击队"


@pytest.mark.asyncio
async def test_node_matcher_heuristic_fallback(mock_standard_graph, mock_speakers):
    # 模拟 LLM 抛出异常，触发降级匹配
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(side_effect=RuntimeError("LLM API Offline"))

    matcher = CandidateNodeMatcher(llm_client=mock_llm)
    cand_graph = await matcher.match_nodes_for_speaker(mock_speakers[0], mock_standard_graph)

    # 验证启发式降级是否通过关键词命中了 c1, d1, u1
    assert "c1" in cand_graph.matched_node_ids
    assert "d1" in cand_graph.matched_node_ids
    assert "u1" in cand_graph.matched_node_ids


@pytest.mark.asyncio
async def test_dual_pruned_arc_verifier(mock_standard_graph, mock_speakers):
    cand_graph = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=[
            mock_standard_graph.get_node("c1"),
            mock_standard_graph.get_node("d1"),
            mock_standard_graph.get_node("u1"),
        ],
        node_mentions=[],
    )

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=ArcVerificationPayload(
            verified_arcs=[
                ArcVerificationItem(source="c1", target="d1", is_valid=True, reasoning="因果明确", weight=1.0),
                ArcVerificationItem(source="d1", target="u1", is_valid=True, reasoning="手段目的明确", weight=1.0),
            ]
        )
    )

    verifier = CandidateArcVerifier(llm_client=mock_llm)
    verified = await verifier.verify_arcs_for_speaker(mock_speakers[0], cand_graph, mock_standard_graph)

    # 验证剪枝效果：标准图谱有 3 条边，由于候选人未提及 c2/d2，c2->d2 被自动剪枝，仅向 LLM 提交 2 条边验证
    assert len(verified) == 2
    arc_ids = {f"{a.source}->{a.target}" for a in verified}
    assert "c1->d1" in arc_ids
    assert "d1->u1" in arc_ids
    assert "c2->d2" not in arc_ids


@pytest.mark.asyncio
async def test_arc_verifier_rejects_unreturned_and_false_arcs(mock_standard_graph, mock_speakers):
    """验证边校验器严格拒绝 is_valid=False 及 LLM 未返回的候选边（修复原默认放行 Bug）"""
    cand_graph = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=[
            mock_standard_graph.get_node("c1"),
            mock_standard_graph.get_node("d1"),
            mock_standard_graph.get_node("u1"),
        ],
        node_mentions=[],
    )
    mock_llm = MagicMock(spec=LLMClient)
    # 仅返回 c1->d1 为 True，省略 d1->u1
    mock_llm.chat_json = AsyncMock(
        return_value=ArcVerificationPayload(
            verified_arcs=[
                ArcVerificationItem(source="c1", target="d1", is_valid=True),
            ]
        )
    )
    verifier = CandidateArcVerifier(llm_client=mock_llm)
    verified = await verifier.verify_arcs_for_speaker(mock_speakers[0], cand_graph, mock_standard_graph)
    assert len(verified) == 1
    assert verified[0].source == "c1" and verified[0].target == "d1"



@pytest.mark.asyncio
async def test_pipeline_runner_end_to_end(mock_standard_graph, tmp_path):
    # Mock LLM and Cache
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock()

    # 模拟抽取标准图谱、互动提取、节点匹配、边校验
    mock_cache = MagicMock(spec=GraphCacheManager)
    mock_cache.get_or_extract = AsyncMock(return_value=mock_standard_graph)

    runner = PipelineRunner(llm_client=mock_llm, cache_manager=mock_cache)

    sample_transcript = """
    说话人 1 00:00
    大家欢迎参加讨论。
    说话人 2 00:15
    我认为当前市场竞争激烈，必须开发AI新功能以提升市场占有率。
    说话人 3 00:45
    我赞同2号的观点，不过预算很紧张，我们需要降本增效。
    """

    result = await runner.run(
        corpus_text="测试语料内容",
        discussion_text=sample_transcript,
        group_id="test_group_1",
    )

    assert result.group_id == "test_group_1"
    assert len(result.transcript.candidate_speakers) == 2
    assert "speaker_2" in result.candidate_graphs
    assert "speaker_3" in result.candidate_graphs
    assert result.elapsed_seconds >= 0.0

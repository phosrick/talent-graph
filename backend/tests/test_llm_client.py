# -*- coding: utf-8 -*-
"""Unit tests for LLM Client"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel, Field
from configs.settings import Settings
from src.core.llm_client import LLMClient


class MockSampleResponse(BaseModel):
    summary: str
    score: int = Field(default=100)


@pytest.mark.asyncio
async def test_json_parsing_robustness():
    """测试各种格式的 JSON 解析容错能力"""
    client = LLMClient(Settings(deepseek_api_key="mock_key"))

    # 1. 正常标准 JSON
    res1 = client._parse_json_to_model('{"summary": "test1", "score": 90}', MockSampleResponse)
    assert res1.summary == "test1"
    assert res1.score == 90

    # 2. 带 markdown ```json 代码块
    res2 = client._parse_json_to_model('```json\n{"summary": "test2", "score": 85}\n```', MockSampleResponse)
    assert res2.summary == "test2"
    assert res2.score == 85

    # 3. 前后带自然语言噪声文本
    res3 = client._parse_json_to_model('Here is the result:\n```json\n{"summary": "test3", "score": 95}\n```\nHope this helps!', MockSampleResponse)
    assert res3.summary == "test3"
    assert res3.score == 95

    # 4. 纯花括号内嵌在文本中
    res4 = client._parse_json_to_model('Output is {"summary": "test4", "score": 70} completed.', MockSampleResponse)
    assert res4.summary == "test4"
    assert res4.score == 70


@pytest.mark.asyncio
async def test_llm_client_mock_chat():
    """测试通过 Mock OpenAI 接口验证 chat_text 与 chat_json 流程"""
    client = LLMClient(Settings(deepseek_api_key="mock_key", llm_max_concurrency=5))

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"summary": "Great analysis", "score": 98}\n```'
    mock_completion.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)):
        # 测试 chat_text
        text = await client.chat_text("Hello")
        assert "Great analysis" in text

        # 测试 chat_json
        parsed = await client.chat_json("Hello", MockSampleResponse)
        assert parsed.summary == "Great analysis"
        assert parsed.score == 98

    stats = client.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_success"] == 2

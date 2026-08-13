# -*- coding: utf-8 -*-
"""
High-Performance Asynchronous LLM Client for DeepSeek / OpenAI / DashScope.
Features:
- Built-in Semaphore Rate-Limiter (default MAX_CONCURRENCY=20)
- Exponential Backoff Retry for network glitches and 429 rate limits
- Pydantic V2 Structured JSON extraction with robust fallback parser
- Batch concurrent execution helper
"""

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from configs.settings import Settings, get_settings
from src.core.logger import get_logger

logger = get_logger("llm_client")
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """高性能异步限流 LLM 客户端"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)

        # 初始化底层 AsyncOpenAI 客户端 (根据 provider 分发)
        provider = self.settings.llm_provider.lower()
        if provider in ("dashscope", "qwen"):
            api_key = self.settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY") or "mock_key"
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.default_model = self.settings.qwen_model_name
        elif provider == "openai":
            api_key = self.settings.openai_api_key or os.environ.get("OPENAI_API_KEY") or "mock_key"
            base_url = self.settings.openai_base_url
            self.default_model = self.settings.openai_model
        else:  # deepseek / default / mock
            api_key = self.settings.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY") or "mock_key"
            base_url = self.settings.deepseek_base_url
            self.default_model = self.settings.deepseek_model

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.settings.llm_timeout,
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=self.settings.llm_max_concurrency * 2,
                max_connections=self.settings.llm_max_concurrency * 3,
            ),
        )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,  # 使用自定义智能指数退避重试
        )

        self.total_requests = 0
        self.total_success = 0
        self.total_failures = 0
        self.total_latency_seconds = 0.0

    async def chat_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """发起单次文本生成请求（带并发信号量与重试保护）"""
        target_model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            self.total_requests += 1
            start_time = time.perf_counter()
            retries = self.settings.llm_max_retries

            for attempt in range(retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content or ""
                    latency = time.perf_counter() - start_time
                    self.total_latency_seconds += latency
                    self.total_success += 1
                    return content.strip()

                except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
                    is_last = attempt == retries
                    if is_last:
                        self.total_failures += 1
                        logger.error(
                            f"LLM request failed after {retries} retries. Model: {target_model}, Error: {exc}"
                        )
                        raise

                    backoff = (2 ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"LLM request attempt {attempt + 1} failed: {exc}. Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)

                except Exception as exc:
                    self.total_failures += 1
                    logger.error(f"Unexpected LLM exception: {exc}")
                    raise

        return ""

    async def chat_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "You are an expert AI assistant that responds in strict JSON format.",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> T:
        """结构化 JSON 提取并自动映射为 Pydantic 数据模型"""
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"【OUTPUT FORMAT REQUIREMENT】\n"
            f"Please output strictly in valid JSON matching the following JSON Schema. Do NOT include markdown comments outside the JSON object.\n"
            f"```json\n{schema_json}\n```"
        )

        raw_text = await self.chat_text(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        return self._parse_json_to_model(raw_text, response_model)

    def _parse_json_to_model(self, raw_text: str, response_model: Type[T]) -> T:
        """鲁棒解析 JSON 字符串（支持提取 ```json 块及容错修复）"""
        cleaned = raw_text.strip()
        # 匹配 markdown 代码块
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if json_match:
            cleaned = json_match.group(1).strip()

        # 尝试直接反序列化
        try:
            return response_model.model_validate_json(cleaned)
        except Exception:
            pass

        # 尝试使用 python json loads 再转
        try:
            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except Exception as exc:
            # 尝试提取最外层花括号
            brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(1))
                    return response_model.model_validate(data)
                except Exception:
                    pass
            logger.error(f"Failed to parse LLM JSON output into {response_model.__name__}. Raw text:\n{raw_text}")
            raise ValueError(f"Failed to parse LLM response into {response_model.__name__}: {exc}") from exc

    async def batch_chat(
        self,
        prompts: List[Tuple[str, str]],  # (prompt, system_prompt)
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> List[str]:
        """批量高并发执行文本请求（受限于信号量）"""
        tasks = [
            self.chat_text(
                prompt=p,
                system_prompt=sp,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for p, sp in prompts
        ]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """获取调用量与性能统计"""
        avg_lat = (
            round(self.total_latency_seconds / max(self.total_success, 1), 3)
            if self.total_success > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_failures": self.total_failures,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "avg_latency_seconds": avg_lat,
        }

    async def close(self) -> None:
        """关闭 HTTP 连接池"""
        await self._client.close()


_global_llm_client: Optional[LLMClient] = None


def get_llm_client(settings: Optional[Settings] = None) -> LLMClient:
    """获取全局单例 LLM 客户端"""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient(settings)
    return _global_llm_client

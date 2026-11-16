# -*- coding: utf-8 -*-
"""
TalentGraph Global Settings Center
Based on Pydantic V2 and pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Union
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 默认顶层数据根目录: talent_graph/data
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "talent_graph/backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # 1. LLM 核心配置 (默认优先使用 DeepSeek)
    # -------------------------------------------------------------------------
    llm_provider: Literal["deepseek", "openai", "dashscope", "qwen", "mock"] = Field(
        default="deepseek",
        description="LLM 服务提供商",
    )

    # DeepSeek 配置
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API 密钥",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API 基础地址",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("deepseek_model", "deepseek_model_name"),
        description="DeepSeek 模型名称 (如 deepseek-chat)",
    )

    # OpenAI 备用配置
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API 密钥",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API 基础地址",
    )
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("openai_model", "openai_model_name"),
        description="OpenAI 模型名称",
    )

    # 阿里通义千问 备用配置
    dashscope_api_key: Optional[str] = Field(
        default=None,
        description="阿里云 DashScope API 密钥",
    )
    qwen_model_name: str = Field(
        default="qwen-plus",
        description="DashScope 模型名称",
    )

    # 通用 LLM 调用参数
    llm_max_concurrency: int = Field(
        default=20,
        description="LLM 最大异步并发请求数",
        ge=1,
        le=100,
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM 请求超时时间 (秒)",
        gt=0,
    )
    llm_max_retries: int = Field(
        default=3,
        description="LLM 失败指数退避重试次数",
        ge=0,
    )

    # -------------------------------------------------------------------------
    # 2. 向量计算配置 (纯 CPU 轻量 ONNX 或 API)
    # -------------------------------------------------------------------------
    embedding_provider: Literal["onnx", "api", "bge", "local", "auto", "mock"] = Field(
        default="onnx",
        description="向量计算引擎提供商: onnx(本地CPU ONNX) | api(百炼/OpenAI向量API) | auto(自动优选)",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        validation_alias=AliasChoices("embedding_model", "bge_model_name"),
        description="向量模型名称或本地 ONNX 目录/文件路径",
    )
    dashscope_embedding_model: str = Field(
        default="text-embedding-v3",
        description="阿里云百炼云端向量模型名称 (512维)",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI 云端向量模型名称 (512维)",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="向量计算批次大小",
        ge=1,
    )

    # -------------------------------------------------------------------------
    # 3. 业务数据与项目目录配置 (顶层解耦与项目隔离)
    # -------------------------------------------------------------------------
    node_similarity_threshold: float = Field(
        default=0.85,
        description="节点匹配语义相似度阈值",
        ge=0.0,
        le=1.0,
    )
    corpus_auto_chunk_threshold: int = Field(
        default=4500,
        description="触发自适应分块抽取的参考语料长度阈值 (字符数)",
    )
    corpus_chunk_size: int = Field(
        default=2800,
        description="超长参考语料分块大小 (字符数)",
    )
    corpus_chunk_overlap: int = Field(
        default=300,
        description="超长参考语料分块重叠大小 (字符数)",
    )
    enable_corpus_smoothing: bool = Field(
        default=False,
        description="是否对参考语料启用高并发 ASR 语音纠错润滑 (纯文字题本默认关闭以0秒直通)",
    )
    node_match_batch_size: int = Field(
        default=70,
        description="超大规模标准图谱下候选人节点匹配的单批节点上限 (防止长Prompt漏召回或截断)",
        ge=20,
        le=200,
    )
    host_speaker_id: str = Field(
        default="auto",
        description="主持人标识: 'auto'=智能检测主持人避免误删1号选手, ''=全员参评无主持人, 或指定如 'speaker_1'",
    )
    data_root_dir: Path = Field(
        default=DEFAULT_DATA_ROOT,
        description="业务数据根目录 (与后端代码解耦，支持外部挂载)",
    )
    corpus_cache_dir: Optional[Path] = Field(
        default=None,
        description="标准题库图谱缓存路径 (默认: data/cache/corpus)",
    )
    projects_dir: Optional[Path] = Field(
        default=None,
        description="按项目/会议场次隔离归档路径 (默认: data/projects)",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="兼容旧接口的输出路径",
    )

    # -------------------------------------------------------------------------
    # 4. Web 服务配置
    # -------------------------------------------------------------------------
    app_host: str = Field(
        default="0.0.0.0",
        description="FastAPI 监听地址",
    )
    app_port: int = Field(
        default=8000,
        description="FastAPI 监听端口",
    )
    log_level: str = Field(
        default="INFO",
        description="日志级别",
    )
    cors_origins: Union[str, List[str]] = Field(
        default="*",
        description="CORS 跨域白名单",
    )

    # -------------------------------------------------------------------------
    # 5. Playwright PDF 导出配置
    # -------------------------------------------------------------------------
    pdf_export_enabled: bool = Field(
        default=True,
        description="是否启用 Playwright PDF 导出",
    )
    pdf_timeout_ms: int = Field(
        default=30000,
        description="PDF 渲染超时时间 (毫秒)",
    )
    pdf_wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="networkidle",
        description="PDF 页面加载等待策略",
    )

    # -------------------------------------------------------------------------
    # 6. Mock 与独立运行规范配置
    # -------------------------------------------------------------------------
    mock_email: bool = Field(
        default=True,
        description="邮件功能 Mock 模式 (仅打 Log，不发起真实 SMTP 发送)",
    )
    enable_telemetry: bool = Field(
        default=False,
        description="是否开启任何外部埋点上报 (默认关闭，纯粹无副作用)",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [orig.strip() for orig in v.split(",") if orig.strip()]
        return v

    def get_corpus_cache_dir(self) -> Path:
        return self.corpus_cache_dir or (self.data_root_dir / "cache" / "corpus")

    def get_projects_dir(self) -> Path:
        return self.projects_dir or self.output_dir or (self.data_root_dir / "projects")

    def get_project_dir(self, project_id: str) -> Path:
        """获取指定项目/会议场次的独立归档目录: data/projects/{project_id}"""
        return self.get_projects_dir() / project_id

    def get_project_report_file(self, project_id: str) -> Path:
        """获取项目全量报告数据文件: data/projects/{project_id}/report.json"""
        return self.get_project_dir(project_id) / "report.json"

    def get_project_graphs_dir(self, project_id: str) -> Path:
        """获取项目拓扑图谱子目录: data/projects/{project_id}/graphs"""
        return self.get_project_dir(project_id) / "graphs"

    def get_project_pdf_dir(self, project_id: str) -> Path:
        """获取项目 PDF 报告导出子目录: data/projects/{project_id}/pdf_reports"""
        return self.get_project_dir(project_id) / "pdf_reports"

    def ensure_directories(self) -> None:
        """确保运行时核心目录结构存在"""
        self.get_corpus_cache_dir().mkdir(parents=True, exist_ok=True)
        self.get_projects_dir().mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """获取全局单例配置实例（带 LRU 缓存）"""
    s = Settings()
    s.ensure_directories()
    return s

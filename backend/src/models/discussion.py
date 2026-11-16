# -*- coding: utf-8 -*-
"""
Domain models for Group Discussion Parsing, Utterances, Speakers, and Interaction Networks.
"""

from typing import Dict, List, Literal, Optional
import networkx as nx
from pydantic import BaseModel, Field, model_validator


class Utterance(BaseModel):
    """候选人单次发言片段"""
    index: int = Field(..., description="全场发言次序编号 (从 0 开始)")
    speaker_id: str = Field(..., description="发言人标识，如 speaker_2")
    speaker_name: str = Field(..., description="发言人展示名，如 说话人 2")
    text: str = Field(..., description="发言原文文本")
    start_time_str: str = Field(default="00:00", description="开始时间字符串 (如 01:25)")
    end_time_str: Optional[str] = Field(default=None, description="结束时间字符串")
    start_seconds: float = Field(default=0.0, description="开始秒数")
    end_seconds: float = Field(default=0.0, description="结束秒数")
    word_count: int = Field(default=0, description="字数统计")

    @model_validator(mode="after")
    def compute_word_count(self) -> "Utterance":
        if not self.word_count and self.text:
            self.word_count = len(self.text.strip())
        return self


class Speaker(BaseModel):
    """候选人/发言人个体信息"""
    speaker_id: str = Field(..., description="唯一编号，如 speaker_2")
    display_name: str = Field(default="", description="展示姓名或代号")
    is_host: bool = Field(default=False, description="是否为主持人/引导员")
    utterances: List[Utterance] = Field(default_factory=list, description="该候选人的所有发言列表")
    total_words: int = Field(default=0, description="总发言字数")
    total_duration_seconds: float = Field(default=0.0, description="总发言时长(秒)")
    turn_count: int = Field(default=0, description="总发言轮次")

    def full_text(self) -> str:
        """拼接该候选人的全部发言文本"""
        return "\n".join(u.text for u in self.utterances)


InteractionType = Literal["support", "deny", "neutral"]


class InteractionEdge(BaseModel):
    """互动关系边 (支持/肯定、反驳/否定、中立承接)"""
    source: str = Field(..., description="表达互动意向的候选人 ID (如 speaker_2)")
    target: str = Field(..., description="被回应的候选人 ID (如 speaker_3)")
    type: InteractionType = Field(..., description="互动类型: support(支持), deny(反驳), neutral(中立)")
    weight: float = Field(default=1.0, description="互动强度或频次")
    evidence: str = Field(default="", description="提取依据语句片段")
    utterance_index: Optional[int] = Field(default=None, description="发生的发言次序号")
    timestamp_seconds: float = Field(default=0.0, description="互动发生时间戳(秒)")


class InteractionNetwork(BaseModel):
    """全场多角色协作/博弈互动拓扑网络"""
    speakers: List[str] = Field(default_factory=list, description="参与互动的候选人 ID 列表")
    edges: List[InteractionEdge] = Field(default_factory=list, description="全部互动边")

    def to_networkx(self) -> nx.DiGraph:
        """构建 NetworkX 加权有向互动图，供 5_3 等中心性算法瞬间完成计算 (对齐 V5 协作网络有向加权)"""
        G = nx.DiGraph()
        for spk in self.speakers:
            G.add_node(spk)
        for e in self.edges:
            signed_w = e.weight if e.type == "support" else (-e.weight if e.type == "deny" else 0.5 * e.weight)
            # 如果已存在多条边，累加权重与带符号协作权重
            if G.has_edge(e.source, e.target):
                G[e.source][e.target]["weight"] += e.weight
                G[e.source][e.target]["signed_weight"] = G[e.source][e.target].get("signed_weight", 0.0) + signed_w
                if e.type not in G[e.source][e.target].get("types", []):
                    G[e.source][e.target]["types"].append(e.type)
            else:
                G.add_edge(
                    e.source,
                    e.target,
                    weight=e.weight,
                    signed_weight=signed_w,
                    type=e.type,
                    types=[e.type],
                    evidence=e.evidence,
                )
        return G


class DiscussionTranscript(BaseModel):
    """讨论全程结构化转录文本数据集"""
    group_id: str = Field(default="", description="讨论小组标识，如 271_1")
    title: str = Field(default="", description="讨论场次主题")
    core_topic: str = Field(default="", description="6_1 提炼出的讨论核心议题焦点")
    convergence_decision: str = Field(default="", description="6_3 讨论最终收敛达成的共识决议")
    speakers: Dict[str, Speaker] = Field(default_factory=dict, description="候选人字典 (speaker_id -> Speaker)")
    utterances: List[Utterance] = Field(default_factory=list, description="按时间线排列的全局发言序列")
    total_duration_seconds: float = Field(default=0.0, description="讨论总时长(秒)")
    interaction_network: Optional[InteractionNetwork] = Field(default=None, description="全场互动网络图")

    @property
    def candidate_speakers(self) -> List[Speaker]:
        """过滤掉主持人后的真实候选人列表"""
        return [s for s in self.speakers.values() if not s.is_host]

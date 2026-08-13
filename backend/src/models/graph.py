# -*- coding: utf-8 -*-
"""
Domain models for Knowledge Graph, Influence Diagrams, and Topological Networks.
"""

from typing import Any, Dict, List, Literal, Optional, Set
import networkx as nx
from pydantic import BaseModel, Field, model_validator

NodeType = Literal["c", "d", "u"]  # c: concept(概念/事实), d: decision(决策/方案), u: utility(效用/目标)
ArcType = Literal["causal", "info", "update"]  # 因果边, 信息边, 更新边


class Node(BaseModel):
    """图谱知识节点"""
    id: str = Field(..., description="节点唯一标识，如 c1, d2, u1")
    name: str = Field(..., description="节点短名称")
    type: NodeType = Field(..., description="节点类型: c=概念/事实, d=决策/方案, u=效用/目标")
    description: str = Field(default="", description="节点详细说明/内涵")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    category: Optional[str] = Field(default=None, description="所属分类")
    embedding: Optional[List[float]] = Field(default=None, description="向量表示")

    @model_validator(mode="after")
    def populate_type_from_id(self) -> "Node":
        if not self.type and self.id:
            prefix = self.id[0].lower()
            if prefix in {"c", "d", "u"}:
                self.type = prefix  # type: ignore
        return self


class Arc(BaseModel):
    """图谱逻辑边 (有向关系)"""
    id: Optional[str] = Field(default=None, description="边的唯一标识")
    source: str = Field(..., description="起点节点 ID")
    target: str = Field(..., description="终点节点 ID")
    type: ArcType = Field(default="causal", description="边类型: causal(因果), info(信息), update(更新)")
    weight: float = Field(default=1.0, description="权重 (0.0 ~ 1.0)")
    description: str = Field(default="", description="逻辑关联阐释")

    @model_validator(mode="after")
    def generate_id_if_empty(self) -> "Arc":
        if not self.id:
            self.id = f"{self.source}->{self.target}"
        return self


class StandardGraph(BaseModel):
    """题库标准参考答案影响图谱"""
    corpus_id: str = Field(default="", description="题库语料标识")
    corpus_hash: str = Field(default="", description="语料文本 SHA256 指纹，用于 0 秒缓存命中")
    title: str = Field(default="", description="题目/案例标题")
    nodes: List[Node] = Field(default_factory=list, description="知识节点列表")
    arcs: List[Arc] = Field(default_factory=list, description="逻辑边列表")

    def get_node(self, node_id: str) -> Optional[Node]:
        """按 ID 查询节点"""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    @property
    def node_id_set(self) -> Set[str]:
        return {n.id for n in self.nodes}

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """按类型筛选节点 (c, d, u)"""
        return [n for n in self.nodes if n.type == node_type]

    def to_networkx(self) -> nx.DiGraph:
        """转换为 NetworkX 有向图，供纯数学评分器极速计算"""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(
                node.id,
                name=node.name,
                type=node.type,
                description=node.description,
                keywords=node.keywords,
            )
        for arc in self.arcs:
            G.add_edge(
                arc.source,
                arc.target,
                id=arc.id,
                type=arc.type,
                weight=arc.weight,
                description=arc.description,
            )
        return G

    @classmethod
    def from_networkx(cls, G: nx.DiGraph, corpus_id: str = "", title: str = "") -> "StandardGraph":
        """从 NetworkX 有向图还原为领域对象"""
        nodes = []
        for n_id, data in G.nodes(data=True):
            node_type = data.get("type", "c")
            if node_type not in {"c", "d", "u"}:
                node_type = n_id[0].lower() if n_id and n_id[0].lower() in {"c", "d", "u"} else "c"
            nodes.append(
                Node(
                    id=str(n_id),
                    name=data.get("name", str(n_id)),
                    type=node_type,
                    description=data.get("description", ""),
                    keywords=data.get("keywords", []),
                )
            )
        arcs = []
        for u, v, data in G.edges(data=True):
            arcs.append(
                Arc(
                    id=data.get("id", f"{u}->{v}"),
                    source=str(u),
                    target=str(v),
                    type=data.get("type", "causal"),
                    weight=data.get("weight", 1.0),
                    description=data.get("description", ""),
                )
            )
        return cls(corpus_id=corpus_id, title=title, nodes=nodes, arcs=arcs)


class NodeMention(BaseModel):
    """候选人单次命中节点的上下文"""
    node_id: str
    utterance_index: int = Field(..., description="命中发言序号")
    timestamp_seconds: float = Field(default=0.0, description="命中时间戳(秒)")
    quote: str = Field(default="", description="命中语句片段")
    confidence: float = Field(default=1.0, description="匹配置信度")


class CandidateGraph(BaseModel):
    """单个候选人发言映射出的个人知识因果子图"""
    speaker_id: str
    matched_nodes: List[Node] = Field(default_factory=list)
    node_mentions: List[NodeMention] = Field(default_factory=list)
    verified_arcs: List[Arc] = Field(default_factory=list)
    novel_nodes: List[Node] = Field(default_factory=list, description="提出的标准库之外创新节点")

    @property
    def matched_node_ids(self) -> Set[str]:
        return {n.id for n in self.matched_nodes}

    def to_networkx(self) -> nx.DiGraph:
        """构建候选人个人拓扑子图"""
        G = nx.DiGraph()
        for node in self.matched_nodes + self.novel_nodes:
            G.add_node(node.id, name=node.name, type=node.type)
        for arc in self.verified_arcs:
            G.add_edge(arc.source, arc.target, type=arc.type, weight=arc.weight)
        return G

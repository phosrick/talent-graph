# -*- coding: utf-8 -*-
"""
Domain models for Knowledge Graph, Influence Diagrams, and Topological Networks.
"""

from typing import Any, Dict, List, Literal, Optional, Set
import networkx as nx
from pydantic import BaseModel, Field, model_validator

NodeType = Literal["c", "d", "u"]  # c: concept/chance(信息/事实), d: decision(决策/方案), u: utility(效用/目标)
NodeLevel = Literal["一级", "二级", "三级"]  # 一级: 宏观主题, 二级: 抽象概念, 三级: 具体事实/措施
ArcType = Literal["causal", "info", "update"]  # 因果边, 信息边, 更新边


class Node(BaseModel):
    """图谱知识节点 (完整保留 V5 影响图正交属性: 类型 + 抽象层级 + 细分角色)"""
    id: str = Field(..., description="节点唯一标识，如 c1, d2, u1")
    name: str = Field(..., description="节点短名称")
    type: NodeType = Field(..., description="节点类型: c=信息/事实, d=决策/方案, u=效用/目标")
    level: NodeLevel = Field(default="三级", description="抽象层级: 一级(宏观主题) / 二级(抽象概念) / 三级(具体案例事实)")
    sub_type: str = Field(
        default="",
        description="节点子分类: c节点为 known(已知)/unknown(未知); u节点为 final(最终目标)/instrumental(手段目标)",
    )
    is_root: bool = Field(default=False, description="是否为根因信息节点 (Root Chance Node)")
    parent_concept: str = Field(default="", description="三级具体节点挂靠的二级抽象概念名称或ID")
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
        if self.level == "二级" and not self.parent_concept:
            self.parent_concept = self.name
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

    @property
    def has_explicit_levels(self) -> bool:
        """检查图谱是否包含显式的抽象层级划分 (一级/二级)"""
        return any(n.level in ("一级", "二级") for n in self.nodes)

    @property
    def concept_node_ids(self) -> Set[str]:
        """概念节点集合 (一级 + 二级)"""
        return {n.id for n in self.nodes if n.level in ("一级", "二级")}

    @property
    def fact_node_ids(self) -> Set[str]:
        """事实/具体节点集合 (三级)"""
        return {n.id for n in self.nodes if n.level == "三级"}

    @property
    def final_target_ids(self) -> Set[str]:
        """最终目标节点集合 (u 且 sub_type='final')"""
        return {n.id for n in self.nodes if n.type == "u" and n.sub_type == "final"}

    @property
    def instrumental_target_ids(self) -> Set[str]:
        """手段目标节点集合 (u 且 sub_type='instrumental')"""
        return {n.id for n in self.nodes if n.type == "u" and n.sub_type == "instrumental"}

    def to_networkx(self) -> nx.DiGraph:
        """转换为 NetworkX 有向图，供纯数学评分器极速计算"""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(
                node.id,
                name=node.name,
                type=node.type,
                level=node.level,
                sub_type=node.sub_type,
                is_root=node.is_root,
                parent_concept=node.parent_concept,
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
            level = data.get("level", "三级")
            if level not in {"一级", "二级", "三级"}:
                level = "三级"
            nodes.append(
                Node(
                    id=str(n_id),
                    name=data.get("name", str(n_id)),
                    type=node_type,
                    level=level,
                    sub_type=data.get("sub_type", ""),
                    is_root=bool(data.get("is_root", False)),
                    parent_concept=data.get("parent_concept", ""),
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
    feasibility_ratings: Dict[str, float] = Field(
        default_factory=dict,
        description="3_3 方案可行性四维评分 (resource_awareness, risk_sensitivity, execution_feasibility, adaptability)",
    )
    openness_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="5_1 开放度四维行为计数 (listening, supporting, analysis, pushing)",
    )
    timing_control_count: Optional[int] = Field(
        default=None,
        description="6_2 控时与议程节奏管理显式次数",
    )

    @property
    def matched_node_ids(self) -> Set[str]:
        return {n.id for n in self.matched_nodes}

    def to_networkx(self) -> nx.DiGraph:
        """构建候选人个人拓扑子图"""
        G = nx.DiGraph()
        for node in self.matched_nodes + self.novel_nodes:
            G.add_node(node.id, name=node.name, type=node.type, level=node.level, sub_type=node.sub_type)
        for arc in self.verified_arcs:
            G.add_edge(arc.source, arc.target, type=arc.type, weight=arc.weight)
        return G

# -*- coding: utf-8 -*-
"""
End-to-End Report Generator.
Aggregates Pipeline, Scoring, and Summaries into the unified ReportData JSON schema.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from configs.settings import Settings, get_settings
from src.core.embedding import EmbeddingEngine, get_embedding_engine
from src.core.logger import get_logger
from src.models.discussion import InteractionNetwork
from src.models.graph import CandidateGraph, StandardGraph
from src.models.report import (
    CollaborationNetworkDTO,
    ComprehensiveDiagnosis,
    ParticipantReport,
    RadarPoint,
    ReportData,
    TeamOverview,
)
from src.models.score import ALL_DIMENSIONS, ScoreMatrix
from src.pipeline.runner import PipelineResult
from src.reporting.summarizer import CandidateSummarizer
from src.scoring.base import ScoringContext
from src.scoring.engine import ScoringEngine

logger = get_logger("report_generator")


class ReportGenerator:
    """完整评测报告生成器"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        summarizer: Optional[CandidateSummarizer] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.settings = settings or get_settings()
        self.scoring_engine = scoring_engine or ScoringEngine(self.settings)
        self.summarizer = summarizer or CandidateSummarizer(settings=self.settings)
        self.embedding_engine = embedding_engine or get_embedding_engine(self.settings)

    def _build_collaboration_network_dto(self, network: InteractionNetwork) -> CollaborationNetworkDTO:
        """构建供前端 ECharts / 拓扑组件直接消费的 DTO"""
        nodes = []
        for spk_id in network.speakers:
            nodes.append({
                "id": spk_id,
                "name": spk_id.replace("speaker_", "候选人 "),
                "symbolSize": 30,
            })

        edges = []
        for edge in network.edges:
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "value": edge.weight,
                "lineStyle": {
                    "color": "#10B981" if edge.type == "support" else ("#EF4444" if edge.type == "deny" else "#9CA3AF"),
                    "width": max(1.0, edge.weight * 2.0),
                },
                "evidence": edge.evidence,
            })

        return CollaborationNetworkDTO(nodes=nodes, edges=edges)

    async def generate_report(
        self,
        pipeline_result: PipelineResult,
        task_id: Optional[str] = None,
        project_id: str = "default_project",
    ) -> ReportData:
        """
        全量端到端报告生成：
        1. 纯内存 16 维评分
        2. 高并发生成诊断评语
        3. 组装前后端统一的 ReportData 契约
        """
        tid = task_id or f"{pipeline_result.group_id}_{int(datetime.now().timestamp())}"
        logger.info(f"📄 [REPORT GENERATION] Generating complete ReportData for task_id={tid}")

        # 1. 评分
        context = ScoringContext(
            group_id=pipeline_result.group_id,
            corpus_id=pipeline_result.corpus_id,
            standard_graph=pipeline_result.standard_graph,
            transcript=pipeline_result.transcript,
            candidate_graphs=pipeline_result.candidate_graphs,
            interaction_network=pipeline_result.interaction_network,
            embedding_engine=self.embedding_engine,
        )
        score_matrix = self.scoring_engine.evaluate_all(context)

        # 2. 诊断评语 (12 人并发生成)
        diagnoses = await self.summarizer.summarize_all_participants(
            speakers=pipeline_result.transcript.speakers,
            score_matrix=score_matrix,
            candidate_graphs=pipeline_result.candidate_graphs,
            max_concurrency=self.settings.llm_max_concurrency,
        )

        # 3. 组装候选人报告列表
        participant_reports: List[ParticipantReport] = []
        total_participants = len(score_matrix.participants)

        for spk_id, profile in score_matrix.participants.items():
            speaker = pipeline_result.transcript.speakers.get(spk_id)
            cand_graph = pipeline_result.candidate_graphs.get(spk_id)
            diag = diagnoses.get(spk_id, ComprehensiveDiagnosis())

            # 将 LLM 子维度诊断优劣势回填至每个 DimensionScore (兼容 V5 advantage / toImprove 明细)
            for dim_def in ALL_DIMENSIONS:
                dim_id = dim_def.id
                ds_obj = profile.dimension_scores.get(dim_id)
                if ds_obj is None:
                    continue
                fb = diag.dimension_feedback.get(dim_id) if diag.dimension_feedback else None
                t_avg = score_matrix.dimension_averages.get(dim_id, 75.0)
                short_name = dim_def.name.split(" - ")[-1]
                if fb and (fb.advantage or fb.toImprove):
                    ds_obj.advantages = fb.advantage or ds_obj.key_points[:2]
                    ds_obj.to_improve = fb.toImprove
                else:
                    ds_obj.advantages = list(ds_obj.key_points[:2]) or [f"{short_name}表现平稳"]
                    ds_obj.to_improve = (
                        [f"可在复杂情境下进一步深化{short_name}的应用"]
                        if ds_obj.score >= t_avg
                        else [f"{short_name}得分低于组内均值({t_avg})，建议针对性加强练习"]
                    )

            # 组装雷达图数据
            radar_points: List[RadarPoint] = []
            for dim_def in ALL_DIMENSIONS:
                dim_id = dim_def.id
                dim_score_obj = profile.dimension_scores.get(dim_id)
                score_val = dim_score_obj.score if dim_score_obj else 65.0
                team_avg = score_matrix.dimension_averages.get(dim_id, 75.0)

                radar_points.append(
                    RadarPoint(
                        dimension_id=dim_id,
                        dimension_name=dim_def.name.split(" - ")[-1],
                        category=dim_def.category,
                        score=score_val,
                        team_avg=team_avg,
                    )
                )

            p_report = ParticipantReport(
                speaker_id=spk_id,
                display_name=speaker.display_name if speaker else spk_id,
                total_score=profile.total_score,
                overall_rank=profile.overall_rank,
                participant_total=total_participants,
                radar_data=radar_points,
                dimension_details=profile.dimension_scores,
                diagnosis=diag,
                matched_nodes_count=len(cand_graph.matched_nodes) if cand_graph else 0,
                speaking_word_count=speaker.total_words if speaker else 0,
                speaking_turns=speaker.turn_count if speaker else 0,
            )
            participant_reports.append(p_report)

        # 按排名降序排列候选人
        participant_reports.sort(key=lambda p: p.overall_rank)

        # 4. 组装团队大盘
        team_overview = TeamOverview(
            participant_count=total_participants,
            average_total_score=score_matrix.team_average_total_score,
            average_scores=score_matrix.dimension_averages,
            collaboration_network=self._build_collaboration_network_dto(pipeline_result.interaction_network),
        )

        report_data = ReportData(
            task_id=tid,
            project_id=project_id,
            group_id=pipeline_result.group_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            corpus_title=pipeline_result.standard_graph.title or "无领导小组讨论评估案例",
            team_overview=team_overview,
            participants=participant_reports,
        )

        # 保存至按项目隔离的业务数据目录
        project_dir = self.settings.get_project_dir(tid)
        graphs_dir = self.settings.get_project_graphs_dir(tid)
        pdf_dir = self.settings.get_project_pdf_dir(tid)

        project_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # 1. 核心自包含评测报告契约
        report_file = self.settings.get_project_report_file(tid)
        report_file.write_text(report_data.model_dump_json(indent=2), encoding="utf-8")

        # 2. 项目元信息简报
        meta_data = {
            "task_id": tid,
            "project_id": project_id,
            "group_id": pipeline_result.group_id,
            "corpus_id": pipeline_result.corpus_id,
            "corpus_title": report_data.corpus_title,
            "generated_at": report_data.generated_at,
            "participant_count": total_participants,
            "average_total_score": score_matrix.team_average_total_score,
            "pipeline_elapsed_seconds": pipeline_result.elapsed_seconds,
        }
        (project_dir / "meta.json").write_text(
            json.dumps(meta_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 3. 详细图谱数据归档
        (graphs_dir / "standard_graph.json").write_text(
            pipeline_result.standard_graph.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (graphs_dir / "interaction_network.json").write_text(
            pipeline_result.interaction_network.model_dump_json(indent=2),
            encoding="utf-8",
        )

        cand_graphs_dir = graphs_dir / "candidates"
        cand_graphs_dir.mkdir(parents=True, exist_ok=True)
        for spk_id, cg in pipeline_result.candidate_graphs.items():
            (cand_graphs_dir / f"{spk_id}.json").write_text(
                cg.model_dump_json(indent=2),
                encoding="utf-8",
            )

        logger.info(f"✅ [PROJECT ARCHIVED] Successfully saved Project artifacts to {project_dir}")
        return report_data

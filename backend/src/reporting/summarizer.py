# -*- coding: utf-8 -*-
"""
Candidate Comprehensive Diagnosis Summarizer.
Generates holistic executive diagnosis, core strengths, weaknesses, growth suggestions,
and personality talent tags in ONE SINGLE LLM request per candidate.
"""

import asyncio
from typing import Dict, List, Optional
from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import Speaker
from src.models.graph import CandidateGraph
from src.models.report import ComprehensiveDiagnosis
from src.models.score import DIMENSION_MAP, ParticipantScoreProfile, ScoreMatrix

logger = get_logger("diagnosis_summarizer")


DIAGNOSIS_SYSTEM_PROMPT = """你是一名资深的企业高管人才测评与无领导小组讨论(LGD)评审专家。
你的任务是根据候选人在 16 项能力维度的具体得分、全场排名、关键行为佐证要点以及团队对比基准，为其撰写一份客观、专业、有深度且极具建设性的综合诊断评语。

【输出规范】
1. summary (总体评价): 150-250字，全面概括候选人在本次讨论中的角色定位、思维特征、行为模式与整体表现；
2. strengths (核心优势): 2-3条，提炼其最具竞争力的能力高光点（结合高分维度与行为佐证）；
3. weaknesses (主要薄弱点): 1-2条，指出其在讨论中暴露出的明显短板或错失的机会（结合低分维度）；
4. growth_suggestions (进阶建议): 2-3条，提供针对性的职场与领导力发展行动建议；
5. tags (人才标签): 2-4个极具概括力的专业标签（如: "战略破局者", "严谨推演派", "敏锐协同者"）。"""


class CandidateSummarizer:
    """候选人全维综合诊断生成器"""

    def __init__(self, llm_client: Optional[LLMClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm_client or get_llm_client(self.settings)

    async def generate_diagnosis(
        self,
        speaker: Speaker,
        profile: ParticipantScoreProfile,
        score_matrix: ScoreMatrix,
        candidate_graph: Optional[CandidateGraph] = None,
    ) -> ComprehensiveDiagnosis:
        """为单名候选人生成全维诊断评语"""
        dim_summary_lines = []
        # 按得分从高到低排序维度
        sorted_dims = sorted(
            profile.dimension_scores.values(),
            key=lambda d: d.score,
            reverse=True,
        )

        for ds in sorted_dims:
            team_avg = score_matrix.dimension_averages.get(ds.dimension_id, 75.0)
            diff_str = f"+{ds.score - team_avg:.1f}" if ds.score >= team_avg else f"{ds.score - team_avg:.1f}"
            kp_str = f" [表现亮点: {'; '.join(ds.key_points)}]" if ds.key_points else ""
            dim_summary_lines.append(
                f"- 【{ds.dimension_name} ({ds.dimension_id})】: 得分 {ds.score} (团队均分 {team_avg}, 差异 {diff_str}, 全场排名 {ds.rank}){kp_str}"
            )

        dimension_text = "\n".join(dim_summary_lines)

        user_prompt = (
            f"【参评候选人】：{speaker.display_name} ({speaker.speaker_id})\n"
            f"【全场综合总分】：{profile.total_score} 分 (全场总排名: 第 {profile.overall_rank} 名 / 共 {len(score_matrix.participants)} 人, 团队均分 {score_matrix.team_average_total_score})\n"
            f"【发言概况】：全场发言 {speaker.turn_count} 轮次，累计字数 {speaker.total_words} 字，发言总时长 {speaker.total_duration_seconds:.1f} 秒\n\n"
            f"【16 维能力细分得分清单】：\n{dimension_text}\n\n"
            f"请严格按 JSON 格式生成该候选人的全维综合诊断评价 (ComprehensiveDiagnosis)。"
        )

        try:
            diagnosis: ComprehensiveDiagnosis = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=ComprehensiveDiagnosis,
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
                temperature=0.3,
            )
            logger.info(f"Generated comprehensive diagnosis for {speaker.speaker_id}")
            return diagnosis

        except Exception as exc:
            logger.warning(
                f"LLM diagnosis generation failed for {speaker.speaker_id}, falling back to rule-based template: {exc}"
            )
            return self._heuristic_diagnosis(speaker, profile, score_matrix)

    def _heuristic_diagnosis(
        self,
        speaker: Speaker,
        profile: ParticipantScoreProfile,
        score_matrix: ScoreMatrix,
    ) -> ComprehensiveDiagnosis:
        """基于得分排名的启发式兜底评语生成"""
        sorted_dims = sorted(
            profile.dimension_scores.values(),
            key=lambda d: d.score,
            reverse=True,
        )
        top_dims = sorted_dims[:3]
        bottom_dims = sorted_dims[-2:]

        top_names = "、".join([d.dimension_name.split(" - ")[-1] for d in top_dims])
        bottom_names = "、".join([d.dimension_name.split(" - ")[-1] for d in bottom_dims])

        summary = (
            f"{speaker.display_name} 在本次讨论中总分 {profile.total_score} 分，排名第 {profile.overall_rank} 名。"
            f"在 {top_names} 等维度展现了明显优势；而在 {bottom_names} 方面仍有一定提升空间。"
        )

        strengths = []
        for d in top_dims:
            kp = d.key_points[0] if d.key_points else f"在{d.dimension_name}方面表现稳健"
            strengths.append(f"{d.dimension_name.split(' - ')[-1]}突出：{kp}")

        weaknesses = []
        for d in bottom_dims:
            weaknesses.append(f"{d.dimension_name.split(' - ')[-1]}偏弱：得分低于组内平均水平，需加强针对性锻炼。")

        suggestions = [
            f"继续保持在 {top_dims[0].dimension_name.split(' - ')[-1]} 上的领导力与推进力。",
            f"在未来讨论中多关注 {bottom_dims[0].dimension_name.split(' - ')[-1]} 的平衡与协同。",
        ]

        tags = [top_dims[0].dimension_name.split(" - ")[-1] + "达人", "稳健参与者"]

        return ComprehensiveDiagnosis(
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            growth_suggestions=suggestions,
            tags=tags,
        )

    async def summarize_all_participants(
        self,
        speakers: Dict[str, Speaker],
        score_matrix: ScoreMatrix,
        candidate_graphs: Dict[str, CandidateGraph],
        max_concurrency: int = 10,
    ) -> Dict[str, ComprehensiveDiagnosis]:
        """高并发批量生成所有候选人的综合诊断"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bound_summarize(spk_id: str) -> tuple[str, ComprehensiveDiagnosis]:
            async with semaphore:
                spk = speakers.get(spk_id)
                prof = score_matrix.participants.get(spk_id)
                cg = candidate_graphs.get(spk_id)
                if not spk or not prof:
                    return spk_id, ComprehensiveDiagnosis()
                diag = await self.generate_diagnosis(spk, prof, score_matrix, cg)
                return spk_id, diag

        candidate_ids = list(score_matrix.participants.keys())
        tasks = [_bound_summarize(spk_id) for spk_id in candidate_ids]
        results = await asyncio.gather(*tasks)
        return {spk_id: diag for spk_id, diag in results}

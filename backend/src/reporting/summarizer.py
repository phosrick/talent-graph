# -*- coding: utf-8 -*-
"""
Candidate Comprehensive Diagnosis Summarizer (V5-Fidelity Single-Pass Edition).
Generates in ONE SINGLE concurrent LLM request per candidate:
1. Holistic executive summary, strengths, weaknesses, growth_suggestions, and talent tags
2. Six capability category reviews (category_summaries -> aligns with V5 7-paragraph summary.py)
3. Per-dimension advantage / toImprove bullet points (dimension_feedback -> aligns with V5 refined_summary.py)
"""

import asyncio
from typing import Dict, Optional
from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, get_llm_client
from src.core.logger import get_logger
from src.models.discussion import Speaker
from src.models.graph import CandidateGraph
from src.models.report import ComprehensiveDiagnosis, DimensionFeedback
from src.models.score import ALL_DIMENSIONS, ParticipantScoreProfile, ScoreMatrix

logger = get_logger("diagnosis_summarizer")


DIAGNOSIS_SYSTEM_PROMPT = """你是一名资深的企业人才测评与无领导小组讨论(LGD)评审专家。
请根据候选人在 16 项能力维度的量化得分、全场排名、图论行为证据以及团队均分基准，一次性输出结构化 JSON 综合诊断报告：

【输出字段规范】
1. summary: 150-220字总体综合评价，概括角色定位、思维特质与全场表现；
2. strengths: 2-3条核心优势亮点（结合高分维度与行为实证）；
3. weaknesses: 1-3条主要短板与错失机会（结合低于均分维度）；
4. growth_suggestions: 2-3条可落地的进阶行动建议；
5. tags: 2-4个凝练专业的人才标签（如 "战略破局者", "严谨推演派", "共识粘合剂"）；
6. category_summaries: 针对六大能力模块（"思维能力"、"决策能力"、"创新能力"、"思维激发"、"冲突协调"、"推进能力"）各给出一句精炼评述（每项30-60字）；
7. dimension_feedback: 针对 16 个维度编号（如 "1_1", "1_2" ... "6_3"），各给出 1 条 advantage（亮点表现）与 1 条 toImprove（提升方向），紧扣量化证据且与整体评语保持一致、不自相矛盾。"""


class CandidateSummarizer:
    """候选人全维综合诊断生成器（1人1次并发调用，完整覆盖 V5 七段综述与16维优劣势）"""

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
        sorted_dims = sorted(
            profile.dimension_scores.values(),
            key=lambda d: d.score,
            reverse=True,
        )

        for ds in sorted_dims:
            team_avg = score_matrix.dimension_averages.get(ds.dimension_id, 75.0)
            diff_str = f"+{ds.score - team_avg:.1f}" if ds.score >= team_avg else f"{ds.score - team_avg:.1f}"
            kp_str = f" [量化证据: {'; '.join(ds.key_points)}]" if ds.key_points else ""
            dim_summary_lines.append(
                f"- 【{ds.dimension_name} ({ds.dimension_id})】: 得分 {ds.score} "
                f"(团队均分 {team_avg}, 差值 {diff_str}, 排名 {ds.rank}){kp_str}"
            )

        dimension_text = "\n".join(dim_summary_lines)
        matched_preview = ""
        if candidate_graph and candidate_graph.matched_nodes:
            sample_names = [n.name for n in candidate_graph.matched_nodes[:8]]
            matched_preview = f"【命中核心议题节点节选】：{', '.join(sample_names)}\n"

        user_prompt = (
            f"【参评候选人】：{speaker.display_name} ({speaker.speaker_id})\n"
            f"【全场综合总分】：{profile.total_score} 分 (全场排名: 第 {profile.overall_rank} 名 / 共 {len(score_matrix.participants)} 人, "
            f"团队均分 {score_matrix.team_average_total_score})\n"
            f"【发言概况】：发言 {speaker.turn_count} 轮，累计 {speaker.total_words} 字，总时长 {speaker.total_duration_seconds:.1f} 秒\n"
            f"{matched_preview}\n"
            f"【16 维能力细分得分与图论证据清单】：\n{dimension_text}\n\n"
            f"请严格按 JSON 格式生成该候选人的 ComprehensiveDiagnosis（包含六大模块综述 category_summaries 与 16 维 dimension_feedback）。"
        )

        try:
            diagnosis: ComprehensiveDiagnosis = await self.llm.chat_json(
                prompt=user_prompt,
                response_model=ComprehensiveDiagnosis,
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=8192,
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
        """基于得分排名的启发式兜底评语生成（同时填充六大模块与 16 维优劣势）"""
        sorted_dims = sorted(
            profile.dimension_scores.values(),
            key=lambda d: d.score,
            reverse=True,
        )
        if not sorted_dims:
            return ComprehensiveDiagnosis(summary=f"{speaker.display_name} 暂无足够评估数据。")

        top_dims = sorted_dims[:3]
        bottom_dims = sorted_dims[-2:]

        top_names = "、".join([d.dimension_name.split(" - ")[-1] for d in top_dims])
        bottom_names = "、".join([d.dimension_name.split(" - ")[-1] for d in bottom_dims])

        summary = (
            f"{speaker.display_name} 在本次讨论中综合得分 {profile.total_score} 分，位列第 {profile.overall_rank} 名。"
            f"在 {top_names} 等维度展现了突出优势；在 {bottom_names} 方面仍有进一步提升空间。"
        )

        strengths = [
            f"{d.dimension_name.split(' - ')[-1]}突出：{d.key_points[0] if d.key_points else '高于团队基准表现'}"
            for d in top_dims
        ]
        weaknesses = [
            f"{d.dimension_name.split(' - ')[-1]}有待加强：当前得分({d.score})低于高分段水平，建议强化主动输出。"
            for d in bottom_dims
        ]
        suggestions = [
            f"持续发挥在 {top_dims[0].dimension_name.split(' - ')[-1]} 上的带动作用。",
            f"后续讨论中注意补强 {bottom_dims[0].dimension_name.split(' - ')[-1]} 的论证深度与互动频次。",
        ]
        tags = [top_dims[0].dimension_name.split(" - ")[-1] + "突出", "稳健协作者"]

        # 六大类小结
        cat_groups: Dict[str, list] = {}
        for dim_def in ALL_DIMENSIONS:
            ds = profile.dimension_scores.get(dim_def.id)
            if ds:
                cat_groups.setdefault(dim_def.category, []).append(ds.score)
        category_summaries = {
            cat: f"{cat}模块均分 {sum(vals)/len(vals):.1f} 分，整体表现{'扎实稳健' if sum(vals)/len(vals) >= 75 else '尚有提升空间'}。"
            for cat, vals in cat_groups.items()
        }

        # 16 维优劣势
        dim_feedback: Dict[str, DimensionFeedback] = {}
        for ds in sorted_dims:
            t_avg = score_matrix.dimension_averages.get(ds.dimension_id, 75.0)
            short_n = ds.dimension_name.split(" - ")[-1]
            adv = ds.key_points[:2] if ds.key_points else [f"在{short_n}维度保持基本参与"]
            imp = (
                [f"继续巩固{short_n}在不同议题阶段的稳定性"]
                if ds.score >= t_avg
                else [f"相较团队均分({t_avg})偏弱，需增加{short_n}相关有效表达"]
            )
            dim_feedback[ds.dimension_id] = DimensionFeedback(advantage=adv, toImprove=imp)

        return ComprehensiveDiagnosis(
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            growth_suggestions=suggestions,
            tags=tags,
            category_summaries=category_summaries,
            dimension_feedback=dim_feedback,
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

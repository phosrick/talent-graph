# -*- coding: utf-8 -*-
"""
Discussion Transcript Parser.
Robustly parses multi-speaker leaderless group discussion transcripts into structured Utterances and Speakers.
Supports:
- Both multi-line speaker headers and single-line inline formats (e.g., '说话人 1 00:15 大家好...')
- Multi-encoding fallback (utf-8, utf-8-sig, gbk, gb18030)
- Smart moderator detection ('auto' mode) so real Candidate #1 ('说话人 1') is never accidentally dropped!
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from src.core.logger import get_logger
from src.models.discussion import DiscussionTranscript, Speaker, Utterance

logger = get_logger("discussion_parser")

_HOST_KEYWORDS = (
    "欢迎参加",
    "欢迎大家",
    "各位候选人",
    "讨论正式开始",
    "现在开始讨论",
    "请大家开始",
    "时间到",
    "我是主持人",
    "宣读规则",
    "请注意时间规则",
)


class DiscussionParser:
    """讨论文本时间戳与发言人结构化解析器"""

    def __init__(self, host_speaker_id: Optional[str] = "auto"):
        """
        Args:
            host_speaker_id:
                - "auto": 智能识别 speaker_1 是否为真主持人（避免误删真实 1 号候选人）
                - "": 全员均为参评候选人，无主持人
                - 具体 ID (如 "speaker_1"): 显式指定主持人 ID
        """
        self.host_speaker_id = host_speaker_id if host_speaker_id is not None else ""

    @staticmethod
    def time_str_to_seconds(time_str: str) -> float:
        """转换 01:25 或 00:01:25 为秒数"""
        parts = time_str.strip().split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
        return 0.0

    @staticmethod
    def normalize_speaker_id(raw_name: str, treat_spk1_as_host: bool = False) -> Tuple[str, str]:
        """
        归一化发言人标识与展示名称
        例如: "说话人 2" -> ("speaker_2", "候选人 2")
        """
        cleaned = raw_name.strip().replace("：", "").replace(":", "")

        if "主持" in cleaned or "考官" in cleaned:
            return "speaker_host", "主持人"

        digit_match = re.search(r"(\d+)", cleaned)
        if digit_match:
            num = digit_match.group(1)
            speaker_id = f"speaker_{num}"
            if speaker_id == "speaker_1" and treat_spk1_as_host:
                display_name = "主持人 / 说话人 1"
            else:
                display_name = f"候选人 {num}"
            return speaker_id, display_name

        safe_id = re.sub(r"[^\w]", "_", cleaned)
        return f"speaker_{safe_id}", cleaned

    @staticmethod
    def _is_procedural_host(spk: Speaker, all_speakers: Dict[str, Speaker]) -> bool:
        """
        智能判定 speaker_1 是否真的是串场主持人（而非 1 号考生）：
        仅当场上至少有 2 人、且 speaker_1 发言极短（<= 2 轮且总字数 <= 100）并带有典型主持引导词时，才判为主持人。
        """
        if len(all_speakers) < 2:
            return False
        if spk.speaker_id == "speaker_host":
            return True
        if spk.speaker_id != "speaker_1":
            return False

        full_txt = spk.full_text()
        has_host_cue = any(kw in full_txt for kw in _HOST_KEYWORDS)
        if spk.turn_count <= 2 and spk.total_words <= 100 and has_host_cue:
            return True
        return False

    def parse_text(
        self,
        raw_text: str,
        group_id: str = "default_group",
        title: str = "小组讨论",
    ) -> DiscussionTranscript:
        """核心解析逻辑：从原始文本提取发言人、时间戳与发言序列"""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Transcript text cannot be empty.")

        # 独立头行: "说话人 1 00:15" / "[00:15] 说话人 1" / "候选人2："
        pattern_speaker_header = re.compile(
            r"^(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*)?"
            r"(主持人|考官|说话人\s*\d+|候选人\s*\d+|\d+号(?:候选人)?|发言人\s*\d+|speaker[_\s]*\d+|[^\s：:]{1,12})"
            r"\s*(?:\(?\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\)?)?[：:]?$",
            re.IGNORECASE,
        )
        # 单行内联格式 (兼容 V5 转写稿): "说话人 1 00:15 各位好，我认为..." 或 "说话人2：我认为..."
        pattern_inline_turn = re.compile(
            r"^(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*)?"
            r"(主持人|考官|说话人\s*\d+|候选人\s*\d+|\d+号(?:候选人)?|发言人\s*\d+|speaker[_\s]*\d+)"
            r"(?:\s+\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+|\s*[：:]\s*)(.+)$",
            re.IGNORECASE,
        )

        explicit_host = self.host_speaker_id == "speaker_1"
        utterances: List[Utterance] = []
        current_speaker_id = ""
        current_speaker_name = ""
        current_time_str = "00:00"
        current_text_buf: List[str] = []
        total_duration = 0.0

        def flush_current():
            nonlocal current_text_buf, utterances
            if current_speaker_id and current_text_buf:
                text_content = " ".join(current_text_buf).strip()
                if text_content:
                    start_sec = self.time_str_to_seconds(current_time_str)
                    u = Utterance(
                        index=len(utterances),
                        speaker_id=current_speaker_id,
                        speaker_name=current_speaker_name,
                        text=text_content,
                        start_time_str=current_time_str,
                        start_seconds=start_sec,
                    )
                    utterances.append(u)
                current_text_buf = []

        for line in lines:
            inline_match = pattern_inline_turn.match(line)
            if inline_match:
                t_pre, spk_raw, t_suf, body = inline_match.groups()
                flush_current()
                spk_id, spk_display = self.normalize_speaker_id(spk_raw, treat_spk1_as_host=explicit_host)
                current_speaker_id = spk_id
                current_speaker_name = spk_display
                current_time_str = t_pre or t_suf or current_time_str
                if body and body.strip():
                    current_text_buf.append(body.strip())
                continue

            match = pattern_speaker_header.match(line)
            if match:
                time_prefix, spk_raw, time_suffix = match.groups()
                # 若无时间戳且无冒号且并非标准“说话人/候选人”标识，视作普通正文句
                is_std_tag = bool(
                    re.match(
                        r"^(主持人|考官|说话人\s*\d+|候选人\s*\d+|\d+号|发言人\s*\d+|speaker[_\s]*\d+)$",
                        spk_raw.strip(),
                        re.IGNORECASE,
                    )
                )
                has_time_or_colon = bool(time_prefix or time_suffix or line.endswith(("：", ":")))
                if len(spk_raw) > 15 or (not is_std_tag and not has_time_or_colon):
                    current_text_buf.append(line)
                    continue

                flush_current()
                spk_id, spk_display = self.normalize_speaker_id(spk_raw, treat_spk1_as_host=explicit_host)
                current_speaker_id = spk_id
                current_speaker_name = spk_display
                current_time_str = time_prefix or time_suffix or current_time_str
            else:
                current_text_buf.append(line)

        flush_current()

        # 若所有的原始时间戳全为 0（部分 V5 文本稿省略了时间戳），按字数语速 (~4字/秒) 合成单调时间轴
        all_zero_ts = len(utterances) > 1 and all(u.start_seconds == 0.0 for u in utterances)
        if all_zero_ts:
            running_sec = 0.0
            for u in utterances:
                u.start_seconds = round(running_sec, 1)
                mins, secs = divmod(int(running_sec), 60)
                u.start_time_str = f"{mins:02d}:{secs:02d}"
                dur = max(3.0, u.word_count / 4.0)
                running_sec += dur
                u.end_seconds = round(running_sec, 1)
            total_duration = running_sec
        else:
            for i in range(len(utterances)):
                if i + 1 < len(utterances):
                    next_start = utterances[i + 1].start_seconds
                    if next_start > utterances[i].start_seconds:
                        utterances[i].end_seconds = next_start
                        utterances[i].end_time_str = utterances[i + 1].start_time_str
                    else:
                        utterances[i].end_seconds = utterances[i].start_seconds + max(2.0, utterances[i].word_count / 4.0)
                else:
                    utterances[i].end_seconds = utterances[i].start_seconds + max(2.0, utterances[i].word_count / 4.0)
                    total_duration = max(total_duration, utterances[i].end_seconds)

        # 归集 Speaker 画像
        speakers: Dict[str, Speaker] = {}
        for u in utterances:
            if u.speaker_id not in speakers:
                speakers[u.speaker_id] = Speaker(
                    speaker_id=u.speaker_id,
                    display_name=u.speaker_name,
                    is_host=False,
                )
            spk = speakers[u.speaker_id]
            spk.utterances.append(u)
            spk.total_words += u.word_count
            spk.turn_count += 1
            duration = max(0.0, u.end_seconds - u.start_seconds)
            spk.total_duration_seconds += duration

        # 判定主持人身份
        for spk_id, spk in speakers.items():
            if spk_id == "speaker_host":
                spk.is_host = True
            elif self.host_speaker_id == "auto":
                if self._is_procedural_host(spk, speakers):
                    spk.is_host = True
                    spk.display_name = "主持人 / 说话人 1"
            elif self.host_speaker_id and spk_id == self.host_speaker_id:
                spk.is_host = True
                if spk_id == "speaker_1":
                    spk.display_name = "主持人 / 说话人 1"

        transcript = DiscussionTranscript(
            group_id=group_id,
            title=title,
            speakers=speakers,
            utterances=utterances,
            total_duration_seconds=round(total_duration, 1),
        )

        candidate_count = len(transcript.candidate_speakers)
        logger.info(
            f"Parsed transcript {group_id}: {len(utterances)} utterances, {len(speakers)} total speakers "
            f"({candidate_count} candidates), total duration {total_duration:.1f}s"
        )
        return transcript

    @staticmethod
    def read_text_with_fallback(path: Path) -> str:
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    def parse_file(
        self,
        file_path: Union[str, Path],
        group_id: Optional[str] = None,
        title: str = "小组讨论",
    ) -> DiscussionTranscript:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        text = self.read_text_with_fallback(path)
        gid = group_id or path.stem
        return self.parse_text(text, group_id=gid, title=title)

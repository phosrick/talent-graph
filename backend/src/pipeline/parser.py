# -*- coding: utf-8 -*-
"""
Discussion Transcript Parser.
Robustly parses multi-speaker leaderless group discussion transcripts into structured Utterances and Speakers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from src.core.logger import get_logger
from src.models.discussion import DiscussionTranscript, Speaker, Utterance

logger = get_logger("discussion_parser")


class DiscussionParser:
    """讨论文本时间戳与发言人结构化解析器"""

    def __init__(self, host_speaker_id: str = "speaker_1"):
        self.host_speaker_id = host_speaker_id

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
    def normalize_speaker_id(raw_name: str) -> Tuple[str, str]:
        """
        归一化发言人标识与展示名称
        例如: "说话人 2" -> ("speaker_2", "候选人 2")
        "2号候选人" -> ("speaker_2", "候选人 2")
        """
        cleaned = raw_name.strip().replace("：", "").replace(":", "")

        # 匹配数字序号
        digit_match = re.search(r"(\d+)", cleaned)
        if digit_match:
            num = digit_match.group(1)
            speaker_id = f"speaker_{num}"
            if speaker_id == "speaker_1":
                display_name = "主持人 / 说话人 1"
            else:
                display_name = f"候选人 {num}"
            return speaker_id, display_name

        # 非数字自定义名称 (如 "张三")
        safe_id = re.sub(r"[^\w]", "_", cleaned)
        return f"speaker_{safe_id}", cleaned

    def parse_text(
        self,
        raw_text: str,
        group_id: str = "default_group",
        title: str = "小组讨论",
    ) -> DiscussionTranscript:
        """
        核心解析逻辑：从原始文本提取发言人、时间戳与发言序列
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Transcript text cannot be empty.")

        # 模式 1: 说话人 1 00:15  或  [00:15] 说话人 1
        # 模式 2: 说话人 2 (01:23)
        pattern_speaker_header = re.compile(
            r"^(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*)?(说话人\s*\d+|候选人\s*\d+|\d+号(?:候选人)?|发言人\s*\d+|[^\s：:]+)\s*(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?)?[：:]?$"
        )

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
            match = pattern_speaker_header.match(line)
            if match:
                # 检查是否匹配到了发言人行
                time_prefix, spk_raw, time_suffix = match.groups()
                time_str = time_prefix or time_suffix or current_time_str

                # 如果看起来像正常的正文句子而非发言人头，不按头处理
                if len(spk_raw) > 15:
                    current_text_buf.append(line)
                    continue

                flush_current()
                spk_id, spk_display = self.normalize_speaker_id(spk_raw)
                current_speaker_id = spk_id
                current_speaker_name = spk_display
                current_time_str = time_str
            else:
                current_text_buf.append(line)

        flush_current()

        # 计算每个发言段的结束时间与讨论总时长
        for i in range(len(utterances)):
            if i + 1 < len(utterances):
                utterances[i].end_seconds = utterances[i + 1].start_seconds
                utterances[i].end_time_str = utterances[i + 1].start_time_str
            else:
                utterances[i].end_seconds = utterances[i].start_seconds + (utterances[i].word_count / 4.0)
                total_duration = utterances[i].end_seconds

        # 归集 Speaker 画像
        speakers: Dict[str, Speaker] = {}
        for u in utterances:
            if u.speaker_id not in speakers:
                is_host = u.speaker_id == self.host_speaker_id
                speakers[u.speaker_id] = Speaker(
                    speaker_id=u.speaker_id,
                    display_name=u.speaker_name,
                    is_host=is_host,
                )
            spk = speakers[u.speaker_id]
            spk.utterances.append(u)
            spk.total_words += u.word_count
            spk.turn_count += 1
            duration = max(0.0, u.end_seconds - u.start_seconds)
            spk.total_duration_seconds += duration

        transcript = DiscussionTranscript(
            group_id=group_id,
            title=title,
            speakers=speakers,
            utterances=utterances,
            total_duration_seconds=round(total_duration, 1),
        )

        candidate_count = len(transcript.candidate_speakers)
        logger.info(
            f"Parsed transcript {group_id}: {len(utterances)} utterances, {len(speakers)} total speakers ({candidate_count} candidates), total duration {total_duration:.1f}s"
        )
        return transcript

    def parse_file(
        self,
        file_path: Union[str, Path],
        group_id: Optional[str] = None,
        title: str = "小组讨论",
    ) -> DiscussionTranscript:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        text = path.read_text(encoding="utf-8")
        gid = group_id or path.stem
        return self.parse_text(text, group_id=gid, title=title)

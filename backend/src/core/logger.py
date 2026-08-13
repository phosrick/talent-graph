# -*- coding: utf-8 -*-
"""Structured logging based on Loguru"""

import sys
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    """配置 Loguru 结构化控制台输出"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level.upper(),
        colorize=True,
    )
    return logger


def get_logger(name: str = "talent_graph"):
    return logger.bind(name=name)

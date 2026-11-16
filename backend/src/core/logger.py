# -*- coding: utf-8 -*-
"""Structured logging based on Loguru (with stdlib logging fallback)"""

import logging
import sys

try:
    from loguru import logger as _loguru_logger

    def setup_logger(log_level: str = "INFO"):
        """配置 Loguru 结构化控制台输出"""
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level.upper(),
            colorize=True,
        )
        return _loguru_logger

    def get_logger(name: str = "talent_graph"):
        return _loguru_logger.bind(name=name)

except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    )

    def setup_logger(log_level: str = "INFO"):
        logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return logging.getLogger("talent_graph")

    def get_logger(name: str = "talent_graph"):
        return logging.getLogger(name)

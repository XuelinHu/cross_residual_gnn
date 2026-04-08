"""日志配置工具。

该模块目前主要被 `geomatric.dingtalk_util` 使用，用来统一文件日志和控制台日志格式。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


def setup_logger(log_file: str = "app.log", log_level: int = logging.INFO) -> logging.Logger:
    """创建并返回项目级 logger。

    默认行为：
    - 日志文件名：`app.log`
    - 文件滚动大小：10 MB
    - 最多保留：5 个历史文件
    - 控制台输出级别：`DEBUG`
    """

    logger = logging.getLogger("app")
    logger.setLevel(log_level)

    # 避免模块被重复导入时重复添加 handler。
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s P %(process)d T %(thread)d %(levelname)s "
            "%(filename)s %(funcName)s %(lineno)d %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 模块导入时直接暴露一个默认 logger，便于小工具脚本直接使用。
logger = setup_logger()

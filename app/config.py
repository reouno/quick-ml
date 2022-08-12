"""Configurations"""
from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Global settings"""
    YOLOV5_PARENT_DIR = Path('./libs/vision/yolov5/runs/detect')


settings = Settings()

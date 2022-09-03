"""Configurations"""
import os
from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Global settings"""
    YOLOV5_PARENT_DIR = Path('./libs/vision/yolov5/runs/detect')

    # Redis
    REDIS_HOST = os.environ.get('REDIS_HOST')
    REDIS_PORT = os.environ.get('REDIS_PORT')

    # Local workspace
    WORKSPACE_DIR = Path('/tmp_workspace')

    # GCS
    GCP_SA_JSON = os.environ.get('GCP_SA_JSON')
    GCS_BUCKET = os.environ.get('GCS_BUCKET')


settings = Settings()

"""Sample tasks"""
import time

from config import settings
from libs.utils.redis_handler import get_redis_handler


def sleep_seconds(sec: int):
    time.sleep(sec)
    job_lock = get_redis_handler(settings)
    job_lock.delete_job_info()

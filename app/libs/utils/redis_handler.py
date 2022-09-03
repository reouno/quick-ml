"""Redis handler"""
import pickle
from typing import Any, Tuple

import redis

from config import Settings
from libs.exceptions import BusyError
from libs.utils.singleton import Singleton

JobStatus = str
JobTmpId = str
JobInfo = Tuple[JobStatus, JobTmpId]

JOB_BUSY = 'busy'
JOB_ERROR = 'error'


def to_job_info(status: JobStatus, job_id: JobTmpId) -> JobInfo:
    return status, job_id


class RedisHandler(Singleton):
    """Redis handler"""

    __JOB_INFO = 'job-info'

    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.__client = redis.Redis(host=host, port=port)

    def _set(self, key: str, value: Any) -> bool:
        return self.__client.set(key, pickle.dumps(value))

    def _get(self, key: str) -> Any:
        return pickle.loads(self.__client.get(key))

    def _delete(self, key: str):
        return self.__client.delete(key)

    def _exists(self, key: str):
        return self.__client.exists(key)

    def is_free(self):
        return not self._exists(self.__JOB_INFO)

    def register_job_info(self, job_id: JobTmpId) -> bool:
        if not self.is_free():
            raise BusyError()

        return self._set(self.__JOB_INFO, to_job_info(JOB_BUSY, job_id))

    def delete_job_info(self):
        return self._delete(self.__JOB_INFO)


def get_redis_handler(settings: Settings):
    return RedisHandler(settings.REDIS_HOST, settings.REDIS_PORT)

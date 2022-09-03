"""API router for debug"""
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import Response

from config import settings
from libs.tasks.sample_tasks import sleep_seconds
from libs.utils.redis_handler import get_redis_handler

logger = logging.getLogger('uvicorn')

router = APIRouter()


@router.post('/sleep-job/')
async def do_sleep_job(seconds: int, background_tasks: BackgroundTasks):
    job_lock = get_redis_handler(settings)

    if not job_lock.is_free():
        raise HTTPException(status_code=423, detail='Server is busy right now.')

    job_lock.register_job_info('dummy-job-id-string')

    background_tasks.add_task(sleep_seconds, seconds)
    logger.info('Sleep job started. ID=(dummy-job-id-string)')

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.delete('/force-delete-job/')
def delete_job():
    job_lock = get_redis_handler(settings)
    job_lock.delete_job_info()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

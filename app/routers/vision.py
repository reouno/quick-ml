"""API router for image/video analysis"""
import logging
import os
import shutil
import tempfile
import uuid
from datetime import time, datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, UploadFile, HTTPException, \
    status
from fastapi.responses import FileResponse, JSONResponse, Response, \
    StreamingResponse

from config import settings
from libs.exceptions import QMLError
from libs.response import to_error_response
from libs.tasks.image_classification_fine_tuning import \
    fine_tune_image_classifier, ImageClassifierFineTuningParams
from libs.tasks.ocr_batch import OcrBatchParams, ocr_batch_task
from libs.utils.redis_handler import get_redis_handler
from libs.vision import run_yolov5
from libs.vision.detectron2 import detectron2_detector
from libs.vision.image.classification import \
    ImageClassificationInferenceParams, image_classification

logger = logging.getLogger('uvicorn')

router = APIRouter()


def yolov5_detect(file: UploadFile, background_tasks: BackgroundTasks,
                  is_image: bool):
    """Tokenization"""
    dir_name = uuid.uuid4().hex
    background_tasks.add_task(shutil.rmtree,
                              settings.YOLOV5_PARENT_DIR / dir_name)
    if file:
        _, ext = os.path.splitext(file.filename)
        with tempfile.NamedTemporaryFile(delete=True, dir='',
                                         suffix=ext) as fp:
            shutil.copyfileobj(file.file, fp)
            fp.seek(0)  # important!!
            try:
                return FileResponse(
                    run_yolov5.run(Path(fp.name), Path(dir_name), is_image))
            except QMLError as err:
                logger.error(f'{err}')
                return JSONResponse(
                    content=to_error_response(str(err), details=err.details),
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
    else:
        return JSONResponse(
            content=to_error_response('No file uploaded')
        )


@router.post('/yolov5-detect-video/')
def yolov5_detect_video(file: UploadFile, background_tasks: BackgroundTasks):
    return yolov5_detect(file, background_tasks, False)


@router.post('/yolov5-detect-image/')
def yolov5_detect_image(file: UploadFile, background_tasks: BackgroundTasks):
    return yolov5_detect(file, background_tasks, True)


@router.post('/vision/image/detectron2/')
async def detectron2_det_image(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    out_img = detectron2_detector.detect(image)
    file_name = Path(file.filename)
    is_success, buffer = cv2.imencode('.jpg', out_img)
    if not is_success:
        raise Exception('Could not encode output image to buffer.')

    out_file_name = f'{file_name.stem[:100]}_out.jpg'
    headers = {
        "Content-Disposition": f"attachment; filename={out_file_name}"
    }

    return StreamingResponse(
        BytesIO(buffer.tobytes()), media_type=f'image/jpg', headers=headers)


@router.post('/vision/image/classification/fine-tuning/')
async def register_image_classifier_fine_tuning_job(
        params: ImageClassifierFineTuningParams,
        background_tasks: BackgroundTasks
):
    job_lock = get_redis_handler(settings)

    if not job_lock.is_free():
        raise HTTPException(status_code=423,
                            detail='Server is busy right now.')

    job_lock.register_job_info(params.job_id)

    background_tasks.add_task(fine_tune_image_classifier, params, settings)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.post('/vision/image/classification/inference/')
def classification_inference(remote_meta_json: str, remote_model_script: str,
                             file: UploadFile):
    params = ImageClassificationInferenceParams()
    params.file = file
    params.remote_meta_json = remote_meta_json
    params.remote_model_script = remote_model_script
    result = image_classification(params)

    return result


@router.post('/vision/image/ocr/batch/')
async def register_ocr_batch_task(
        params: OcrBatchParams,
        background_tasks: BackgroundTasks
):
    job_lock = get_redis_handler(settings)

    if not job_lock.is_free():
        raise HTTPException(status_code=423,
                            detail='Server is busy right now.')

    job_lock.register_job_info(params.job_id)

    background_tasks.add_task(ocr_batch_task, params, settings)

    return Response(status_code=status.HTTP_202_ACCEPTED)

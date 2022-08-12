"""API router for image/video analysis"""
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from libs.exceptions import QMLError
from libs.response import to_error_response
from libs.vision import run_yolov5

logger = logging.getLogger('uvicorn')

router = APIRouter()


@router.post('/yolov5-detect/')
def tokenize(file: UploadFile, background_tasks: BackgroundTasks):
    """Tokenization"""
    dir_name = uuid.uuid4().hex
    background_tasks.add_task(shutil.rmtree,  settings.YOLOV5_PARENT_DIR / dir_name)
    if file:
        _, ext = os.path.splitext(file.filename)
        with tempfile.NamedTemporaryFile(delete=True, dir='', suffix=ext) as fp:
            shutil.copyfileobj(file.file, fp)
            fp.seek(0)  # important!!
            try:
                return FileResponse(run_yolov5.run(Path(fp.name), dir_name))
            except QMLError as err:
                logger.error(f'{err}')
                return JSONResponse(content=to_error_response(str(err), details=err.details), status_code=500)
    else:
        return JSONResponse(
            content=to_error_response('No file uploaded')
        )

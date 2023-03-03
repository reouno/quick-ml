"""GCP Vision API OCR batch"""
import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from config import Settings
from libs.services.data_store import GcsArchiveStore
from libs.utils.gcp_util import get_gcs_and_bucket, get_vision_client
from libs.utils.redis_handler import get_redis_handler
from libs.vision.image.ocr_gcp_vis import do_ocr

logger = logging.getLogger('uvicorn')


class OcrBatchParams(BaseModel):
    """Request body"""
    job_id: str
    remote_workspace_dir: str
    gcs_archive_file: str


def do_ocr_batch(gcs_path: Path, result_gcs_path: Path, settings: Settings):
    """Do OCR for all images in the specified GCS archive"""

    """
    1. download
    2. extractall
    3. do OCR for all
    4. archive results
    4. upload archive
    """

    gcs, bucket = get_gcs_and_bucket(settings)
    client = get_vision_client(settings.GCP_SA_JSON)

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        gcs_store = GcsArchiveStore(work_dir, bucket)
        data_dir = work_dir / gcs_path.stem
        gcs_store.load(gcs_path, data_dir)
        result_dir = work_dir / f'{data_dir.name}_result'
        os.mkdir(result_dir)
        for path in data_dir.rglob('*'):
            logger.debug(f'{path}')
            if path.is_dir():
                os.mkdir(result_dir / path.relative_to(data_dir))
            elif path.is_file():
                p_dir = result_dir / path.parent.relative_to(data_dir)
                output_path = p_dir / f'{path.stem}_ocr.jpeg'
                result_json = p_dir / f'{path.stem}_ocr.json'
                do_ocr(path, output_path, result_json, client)

        gcs_store.save(result_dir, result_gcs_path)


def ocr_batch_task(params: OcrBatchParams, settings: Settings):
    try:
        gcs_archive_file = Path(params.gcs_archive_file)
        remote_output_dir = Path(params.remote_workspace_dir) / 'output'
        result_gcs_path = remote_output_dir / gcs_archive_file.name
        do_ocr_batch(gcs_archive_file, result_gcs_path, settings)
    except Exception as err:
        import traceback
        traceback.print_exc()
        logger.error(err)
    finally:
        job_lock = get_redis_handler(settings)
        job_lock.delete_job_info()

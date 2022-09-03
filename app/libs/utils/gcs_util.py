"""GCS util functions"""
import json
from pathlib import Path
from typing import Union, Optional, Tuple

from google.cloud.storage import Bucket, Blob, Client

from config import Settings


def get_gcs_and_bucket(settings: Settings) -> Tuple[Client, Bucket]:
    # Create a Cloud Storage client.
    gcs = Client.from_service_account_info(json.loads(settings.GCP_SA_JSON))

    bucket = gcs.get_bucket(settings.GCS_BUCKET)

    return gcs, bucket


def upload_string_to_gcs(bucket: Bucket, path: str, data: str, content_type: str = 'text/plain; charset=utf-8') -> Blob:
    """Upload string to gcs"""
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return blob


def upload_file_to_gcs(bucket: Bucket, path: str, file_path: Union[str, Path],
                       content_type: Optional[str] = None) -> Blob:
    """Upload file to gcs"""
    blob = bucket.blob(path)
    blob.upload_from_filename(file_path, content_type=content_type)
    return blob


def upload_file_obj_to_gcs(bucket: Bucket, path: str, file_obj, content_type: str, **kwargs) -> Blob:
    """Upload file to gcs"""
    blob = bucket.blob(path)
    blob.upload_from_file(file_obj, content_type=content_type, **kwargs)
    return blob

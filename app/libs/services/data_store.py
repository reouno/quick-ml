import logging
import shutil
from abc import abstractmethod
from pathlib import Path
from zipfile import ZipFile

from google.cloud.storage import Bucket

from libs.utils.gcp_util import upload_file_to_gcs

logger = logging.getLogger('uvicorn')


class DataStoreBase:
    """Base class of Data store"""

    @abstractmethod
    def load(self, path: Path, path_to: Path):
        pass

    @abstractmethod
    def save(self, path: Path, path_to: Path):
        pass


class GcsArchiveStore(DataStoreBase):
    """GCS Archive store"""

    def __init__(self, work_dir: Path, bucket: Bucket):
        self.__work_dir = work_dir
        self.__bucket = bucket

    def load(self, path: Path, path_to: Path):
        """Download and extract archive file from GCS

        allowed format: zip
        """
        blob = self.__bucket.blob(str(path))
        archive = self.__work_dir / path.name
        blob.download_to_filename(str(archive))

        try:
            with ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(path_to)
        except Exception as err:
            logger.error(f'Failed extract archive "{archive}"')
            raise err

    def save(self, path: Path, path_to: Path):
        """Create zip and upload to GCS"""
        if not path.is_dir():
            raise RuntimeError(
                f'path must be an existing directory, but got "{path}"'
            )

        archive = self.__work_dir / f'{path.name}.zip'
        shutil.make_archive(str(path), 'zip', path)
        upload_file_to_gcs(self.__bucket, str(path_to), archive,
                           content_type='application/zip')

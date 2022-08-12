"""Execute yolov5 detector"""
import logging
import subprocess
from pathlib import Path

from libs.exceptions import QMLError

logger = logging.getLogger('uvicorn')


def run(file_path: Path, save_dir_name: Path) -> Path:
    """Execute yolov5"""
    cmd = f'cd libs/vision/yolov5 && python detect.py --source {file_path} --name {save_dir_name}'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    save_dir = Path('./libs/vision/yolov5/runs/detect') / save_dir_name
    saved_file = save_dir / file_path.name

    if not saved_file.exists():
        error_details = {
            'cmd': cmd,
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8'),
            'not_saved_file': str(saved_file),
        }
        logger.error(f'{error_details}')
        raise QMLError('YOLOv5 detector failed.', error_details)

    return saved_file

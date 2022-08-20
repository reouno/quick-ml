"""Execute yolov5 detector"""
import ffmpeg
import logging
import subprocess
from pathlib import Path

from config import settings
from libs.exceptions import QMLError

logger = logging.getLogger('uvicorn')


def run(file_path: Path, save_dir_name: Path) -> Path:
    """Execute yolov5"""
    if file_path.suffix.upper() == '.MOV':
        mp4_path = file_path.with_suffix('.mp4')
        ffmpeg.input(str(file_path)).output(str(mp4_path)).run(overwrite_output=True, cmd='/usr/bin/ffmpeg')
        file_path = mp4_path
    cmd = f'cd libs/vision/yolov5 && python detect.py --source {file_path} --name {save_dir_name}'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    save_dir = settings.YOLOV5_PARENT_DIR / save_dir_name
    saved_file = save_dir / file_path.name
    saved_file = saved_file.with_suffix('.mp4')

    result_info = {
        'cmd': cmd,
        'stdout': result.stdout.decode('utf-8'),
        'stderr': result.stderr.decode('utf-8'),
        'saved_file': str(saved_file),
    }
    logger.debug(f'{result_info}')

    if not saved_file.exists():
        logger.error(f'{result_info}')
        raise QMLError('YOLOv5 detector failed.', result_info)

    return saved_file

"""Image classification"""
import json
import logging
import os
import shutil
import tempfile
from typing import Optional, Tuple, List

import torch.jit
from PIL import Image
from fastapi import UploadFile
from google.cloud.storage import Blob
from torchvision import transforms

from config import settings
from libs.utils.gcs_util import get_gcs_and_bucket

logger = logging.getLogger('uvicorn')


class ImageClassificationInferenceParams:
    file: UploadFile
    remote_meta_json: str
    remote_model_script: Optional[str] = None


def image_classification(params: ImageClassificationInferenceParams) -> List[Tuple[str, float]]:
    """Image classification inference"""

    gcs, bucket = get_gcs_and_bucket(settings)

    # Download meta json
    meta_json_blob = Blob.from_string(params.remote_meta_json, client=gcs)
    meta_json = json.loads(meta_json_blob.download_as_string().decode('utf-8'))
    class_labels: list[str] = meta_json['classes']
    logger.debug(f'CLASS LABELS: {class_labels}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix='./') as tmpdir:
        img_file = os.path.join(tmpdir, params.file.filename)
        with open(img_file, 'wb') as fp:
            shutil.copyfileobj(params.file.file, fp)

        img = Image.open(img_file)
        logger.debug(f'READ IMAGE SIZE: {img.size}')

        model_script_blob = Blob.from_string(params.remote_model_script, client=gcs)
        model_script_file = os.path.join(tmpdir, 'model_script.pt')
        with open(model_script_file, 'wb') as fp:
            model_script_blob.download_to_file(fp, client=gcs)

        model = torch.jit.load(model_script_file).to(device)
        model.eval()
        logger.debug(f'LOADED MODEL: {model.eval()}')

        common_transforms = transforms.Compose([
            transforms.Resize(int(meta_json['input_size'])),
            transforms.ToTensor(),
        ])
        input_ = common_transforms(img).unsqueeze(0).to(device)
        pred = model(input_).squeeze(0).softmax(0).tolist()
        logger.debug(f'PREDICTED RESULT: {pred}')

    return list(zip(class_labels, pred))

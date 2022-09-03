"""Image classification fine-tuning"""
import datetime
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud.storage import Bucket
from pydantic import BaseModel
from torchvision import datasets, transforms
from ulid import new as ulid_new

from config import Settings
from libs.exceptions import UnexpectedFileTypeError, QMLError
from libs.utils.gcs_util import get_gcs_and_bucket, upload_file_to_gcs, upload_string_to_gcs
from libs.utils.image_classification_libs import initialize_model, train_model
from libs.utils.redis_handler import get_redis_handler

logger = logging.getLogger('uvicorn')


def extract_dataset(file_path: Path, extract_dir: Path, ext: str):
    """Extract dataset"""
    if ext.lower() == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise UnexpectedFileTypeError(f'The file is {file_path}, ext: {ext}')


def download_and_extract_dataset(
        remote_workspace_dir: str,
        dataset_parent_dir: Path,
        bucket: Bucket
) -> Path:
    remote_dataset_dir = os.path.join(remote_workspace_dir, 'dataset')
    blobs = bucket.list_blobs(prefix=remote_dataset_dir)

    # NOTE: There must be only one directory and one archive file.
    for blob in blobs:
        _, _ext = os.path.splitext(blob.name)
        if len(_ext):  # The archive file
            file_path = dataset_parent_dir / Path(blob.name).name
            extract_dir, _ = os.path.split(file_path)
            blob.download_to_filename(file_path)

            # Extract archive file
            extract_dataset(file_path, extract_dir, _ext)

            # Find the extracted directory
            listed = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir)]
            dirs = [d for d in listed if os.path.isdir(d)]
            if len(dirs) != 1:
                raise QMLError(
                    'Failed to find extracted dataset directory',
                    f'Found directories: {dirs}'
                )

            return Path(dirs[0])


class ImageClassifierFineTuningParams(BaseModel):
    """Request body"""
    job_id: str
    remote_workspace_dir: str
    model_name: str = 'squeezenet'
    batch_size: int = 8
    num_epochs: int = 10
    feature_extract: bool = True

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "remote_workspace_dir": self.remote_workspace_dir,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "feature_extract": self.feature_extract,
        }


def upload_trained_model(model: nn.Module, checkpoint: dict, meta_dict: dict, model_dir: Path, remote_model_dir: Path,
                         params: ImageClassifierFineTuningParams, bucket: Bucket):
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    checkpoint_file_name = f'{params.model_name}_checkpoint_{now}.pt'
    checkpoint_file = model_dir / checkpoint_file_name
    remote_checkpoint_file = remote_model_dir / checkpoint_file_name
    torch_script_file_name = f'{params.model_name}_scripted_{now}.pt'
    torch_script_file = model_dir / torch_script_file_name
    remote_torch_script_file = remote_model_dir / torch_script_file_name
    torch.save(checkpoint, checkpoint_file)
    torch.jit.script(model).save(torch_script_file)
    upload_file_to_gcs(bucket, str(remote_checkpoint_file), checkpoint_file)
    upload_file_to_gcs(bucket, str(remote_torch_script_file), torch_script_file)
    remote_meta_file = str(remote_model_dir / f'meta_{now}.json')
    upload_string_to_gcs(
        bucket, remote_meta_file, json.dumps(meta_dict, indent=2), content_type='application/json; charset=utf-8')


def fine_tune_image_classifier(params: ImageClassifierFineTuningParams, settings: Settings):
    """Fine tune and save model to GCS"""
    tmp_dir = settings.WORKSPACE_DIR / Path(ulid_new().str)
    try:
        os.mkdir(tmp_dir)
        dataset_parent_dir = tmp_dir / 'dataset'
        os.mkdir(dataset_parent_dir)
        model_dir = tmp_dir / 'model'
        os.mkdir(model_dir)

        _, bucket = get_gcs_and_bucket(settings)

        dataset_dir = download_and_extract_dataset(
            params.remote_workspace_dir, dataset_parent_dir, bucket)

        # Determine the number of classes
        num_classes = len(list((dataset_dir / 'train').glob('*')))

        # Initialize the model for this run
        model_ft, input_size = initialize_model(
            params.model_name, num_classes, params.feature_extract, use_pretrained=True)

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create training and validation datasets
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params.batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        if params.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Set up the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist, last_epoch_loss, last_epoch_acc, train_logs = train_model(
            model_ft, dataloaders_dict, criterion, optimizer_ft, device,
            num_epochs=params.num_epochs, is_inception=(params.model_name == "inception"))

        remote_model_dir = Path(params.remote_workspace_dir) / 'model'
        checkpoint = {
            'epoch': params.num_epochs,
            'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'loss': last_epoch_loss,
        }
        meta_dict = {
            "params": params.to_dict(),
            "results": {
                "epoch": params.num_epochs,
                "last_epoch_val_loss": float(last_epoch_loss),
                "last_epoch_val_acc": float(last_epoch_acc),
            },
            "train_logs": train_logs,
        }
        upload_trained_model(model_ft, checkpoint, meta_dict, model_dir, remote_model_dir, params, bucket)

        logger.info('Fine-tuning is successful!')
    except Exception as err:
        import traceback
        traceback.print_exc()
        logger.error(err)
    finally:
        shutil.rmtree(tmp_dir)
        job_lock = get_redis_handler(settings)
        job_lock.delete_job_info()

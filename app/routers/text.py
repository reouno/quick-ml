"""API router for text analysis"""
import shutil
import tempfile

from fastapi import APIRouter, UploadFile

from libs.text import tokenize as text_tokenize

router = APIRouter()


@router.post('/tokenize/')
def tokenize(file: UploadFile):
    """Tokenization"""
    if file:
        with tempfile.NamedTemporaryFile(delete=True, dir='', suffix='.txt') as fp:
            shutil.copyfileobj(file.file, fp)
            fp.seek(0)  # important!!
            return text_tokenize.tokenize(fp.name)

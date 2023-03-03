import io
import logging
from pathlib import Path

import numpy as np
from PIL import ImageFont, ImageDraw, Image
from google.cloud import vision
from google.protobuf import json_format

logger = logging.getLogger(__name__)


def detect_text(
        file_obj: bytes,
        client: vision.ImageAnnotatorClient
) -> vision.AnnotateImageResponse:
    """Detects text in the file."""
    image = vision.Image(content=file_obj)

    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return response


def white_mask(img: np.ndarray, percent: float):
    """percent: [0,1]"""
    img[:, :, :] = img[:, :, :] + (255 - img[:, :, :]) * percent


def put_polygon(pil_img, full_annot: vision.TextAnnotation):
    draw = ImageDraw.Draw(pil_img)
    for page in full_annot.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for symbol in word.symbols:
                        vs = symbol.bounding_box.vertices
                        ws = [v.x for v in vs]
                        w = max(ws) - min(ws)
                        hs = [v.y for v in vs]
                        h = max(hs) - min(hs)
                        size = 90 + int((min([w, h]) - 90) * 0.7)
                        font = ImageFont.truetype(
                            'data/fonts/mplus-1m-regular.ttf',
                            size)
                        draw.text((vs[0].x, vs[0].y), symbol.text,
                                  fill=(0, 0, 0), font=font)
                        ps = []
                        for v in vs:
                            ps.append((v.x, v.y))
                        draw.polygon(ps, outline=(255, 0, 0))


def output_ocr_image(
        file_obj: bytes,
        ocr: vision.AnnotateImageResponse
) -> io.BytesIO:
    img = Image.open(io.BytesIO(file_obj))

    img_arr = np.array(img)
    white_mask(img_arr, 0.7)
    img = Image.fromarray(img_arr)

    texts = ocr.text_annotations
    if len(texts) == 0:
        logger.warning(f'No OCR result')
        return img.tobytes()

    put_polygon(img, ocr.full_text_annotation)

    bio = io.BytesIO()
    img.save(bio, 'jpeg')
    img.close()
    return bio


def do_ocr(f_path: Path, output_path: Path,
           result_json: Path, client: vision.ImageAnnotatorClient):
    with io.open(f_path, 'rb') as image_file:
        content = image_file.read()

    result = detect_text(content, client)
    output_img_bytes = output_ocr_image(content, result)

    with open(result_json, 'w') as fp:
        response_json = json_format.MessageToJson(
            result._pb, ensure_ascii=False, indent=4)
        fp.write(response_json)

    with open(output_path, 'wb') as fp:
        fp.write(output_img_bytes.getbuffer())

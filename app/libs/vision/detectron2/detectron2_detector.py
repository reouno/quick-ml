from pathlib import Path
from typing import Union

from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def detect(input_img: Union[Path, np.ndarray]):
    if isinstance(input_img, Path):
        im = cv2.imread(str(input_img))
    else:
        im = input_img

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def main():
    img_file = Path('IMG_4020.JPG')
    img_file = Path('IMG_3596.jpg')
    img_file = Path('people.png')
    out = detect(img_file)
    cv2.imshow('Detected', out)
    cv2.waitKey(0)
    cv2.imwrite(img_file.stem + '_det' + img_file.suffix, out)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

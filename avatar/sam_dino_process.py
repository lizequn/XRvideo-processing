import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import cv2
import json
import pickle

import supervision as sv

import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

GROUNDING_DINO_CONFIG_PATH = os.environ['GROUNDING_DINO_CONFIG_PATH']
GROUNDING_DINO_CHECKPOINT_PATH = os.environ['GROUNDING_DINO_CHECKPOINT_PATH']
SAM_CHECKPOINT_PATH = os.environ['SAM_CHECKPOINT_PATH']
SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running with {DEVICE}")

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", type=str)
parser.add_argument("--dest",type=str)
parser.add_argument("--dry_run",action="store_true")
args = parser.parse_args()

DRY_RUN = args.dry_run
FILE_PATH = Path(args.source)
frame_folders =[f for f in FILE_PATH.glob("*") if f.is_dir()]
print(f"processing # {len(frame_folders)} videos")

OUT_PATH = Path(args.dest)
OUT_PATH.mkdir(exist_ok=True)

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']

CLASSES = ['full body avatar']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
if __name__ == "__main__":
    for frame_folder in tqdm(frame_folders):
        video_name = frame_folder.name
        output_path = OUT_PATH/ f'{video_name}_anno.pkl'
        # output_path.mkdir(exist_ok=True)
        #images={}
        annotations={}
        image_paths = sv.list_files_with_extensions(
            directory=str(frame_folder), 
            extensions=IMAGES_EXTENSIONS)
        for image_path in tqdm(image_paths):
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=CLASSES,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            detections = detections[detections.class_id != None]
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            #images[image_name] = image
            annotations[image_name] = detections
            del image
            if DRY_RUN:
                break
        with open(output_path,'wb') as f:
            pickle.dump(annotations,f)
        if DRY_RUN:
            break

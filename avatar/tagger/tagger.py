from typing import List
import argparse
import functools

import pandas as pd
import numpy as np
import onnxruntime as rt
from PIL import Image
from tqdm import tqdm
from dbimutils import *
from pathlib import Path
import json

MODEL_FILENAME = "./tagger/model.onnx"
LABEL_FILENAME = "./tagger/selected_tags.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder","-i", type=str)
    parser.add_argument("--pattern",type=str,default="*_mask.*")
    return parser.parse_args()

def load_model(path: str) -> rt.InferenceSession:
    model = rt.InferenceSession(path,providers=['CUDAExecutionProvider'])
    return model

def load_labels(path):

    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def predict(
    image: Image,
    model: rt.InferenceSession,
    general_threshold: float,
    character_threshold: float,
    tag_names: List[str],
    rating_indexes: List[np.int64],
    general_indexes: List[np.int64],
    character_indexes: List[np.int64],
):

    rawimage = image
    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]
    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))

    return a, c, rating, character_res, general_res

def main():
    args = parse_args()
    general_threshold = 0.2
    character_threshold = 0.2
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels(LABEL_FILENAME)
    model = load_model(MODEL_FILENAME)
    func = functools.partial(
        predict,
        model=model,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        tag_names=tag_names,
        rating_indexes=rating_indexes,
        general_indexes=general_indexes,
        character_indexes=character_indexes
    )
    img_files = list(Path(args.image_folder).glob(args.pattern))
    print(f"# of image to predict {len(img_files)}")
    result_dict = {}
    for img_file in tqdm(img_files):
        img = Image.open(str(img_file))
        a, c, rating, character_res, general_res = func(img)
        result_dict[img_file.name] = {'rating':rating,
                                      'character': character_res,
                                      'general': general_res}
    with open(f'./{Path(args.image_folder).name}.json','w') as f:
        json.dump(result_dict,f)

main()
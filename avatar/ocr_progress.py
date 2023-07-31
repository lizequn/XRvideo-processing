import pandas as pd
from mmocr.apis import MMOCRInferencer
import cv2
import numpy as np
# from sklearn.cluster import KMeans
from pathlib import Path
from tqdm.auto import tqdm
from numpy import ones, vstack
from numpy.linalg import lstsq
import json

ocr = MMOCRInferencer(det='dbnetpp', rec='abinet')

menu_dict = {
    'body': ['body'],
    'skin_tone': ['tone', 'skin'],
    'hair': ['hair'],
    'face_shape': ['shape'],
    'face_markings': ['markings'],
    'face_lines': ['lines'],
    'eyes': ['eyes'],
    'eyebrows': ['eyebrows'],
    'eyelashes': ['eyelashes'],
    'nose': ['nose'],
    'mouth': ['mouth'],
    'facial_hair': ['facial'],
}
menu_dict2 = {
    'outfit': ['outfit'],
    'eyewear': ['eyewear'],
    'headwear': ['headwear'],
    'bindi': ['bindi'],
    'ear_piercings': ['ear'],
    'nose_piercings': ['nose'],
    'hearing_devices': ['devices']
}



def match_ocr(ocr_result):
    match_dict = {}
    match_dict2 = {}
    for idx, text in enumerate(ocr_result['predictions'][0]['rec_texts']):

        for k, v in menu_dict.items():
            for match_value in v:
                if match_value in text.lower():
                    if k not in match_dict:  # first match
                        match_dict.setdefault(k, {})
                        match_dict[k].update({match_value: text, 'idx': idx})
                    else:
                        if match_value == text.lower():
                            match_dict[k].update({match_value: text, 'idx': idx})

    for idx, text in enumerate(ocr_result['predictions'][0]['rec_texts']):

        for k, v in menu_dict2.items():
            for match_value in v:
                if match_value in text.lower():
                    if k not in match_dict2:  # first match
                        match_dict2.setdefault(k, {})
                        match_dict2[k].update({match_value: text, 'idx': idx})
                    else:
                        if match_value == text.lower():
                            match_dict2[k].update({match_value: text, 'idx': idx})
    if abs(len(match_dict) - len(menu_dict)) <= 0:
        return match_dict, 0
    if abs(len(match_dict2) - len(menu_dict2)) <= 1:
        return match_dict2, 1
    return match_dict, -1


def get_bbox(key, match_dict, ocr_result):
    if key in match_dict:
        idx = match_dict[key]['idx']
        bbox = ocr_result['predictions'][0]['det_polygons'][idx]
        return bbox
    return None


def get_line_eq(points):
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    def line(x):
        return m * x + c

    return line


def get_croped_by_bbox(img, bbox, mul=1):
    x_list = []
    y_list = []
    bbox = [int(p) for p in bbox]
    for i in range(0, len(bbox), 2):
        x_list.append(bbox[i])
        y_list.append(bbox[i + 1])
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    croped = img[y_min:y_max, x_min:x_max].copy()
    return croped


def build_hist_X(img, menu_dict, match_dict, ocr_result):
    hist_label = []
    hist_X = []
    for key in menu_dict.keys():
        bbox = get_bbox(key, match_dict, ocr_result)
        if bbox is None: continue
        croped = get_croped_by_bbox(img, bbox)[:, :, ::-1]
        color = ('b', 'g', 'r')
        color_hist = []
        for ii, col in enumerate(color):
            if ii != 0: continue
            histr = cv2.calcHist([croped], [ii], None, [256], [0, 256])
            histr /= histr.sum()
            color_hist.append(histr.squeeze())
        color_hist = np.array(color_hist)
        hist_label.append(key)
        hist_X.append(color_hist.flatten())
    hist_X = np.array(hist_X)
    return hist_X, hist_label


# def predict_selection(hist_X, hist_label):
#     y_pred = KMeans(n_clusters=2).fit_predict(hist_X)
#     counts = np.bincount(y_pred)
#     most_freq = np.argmax(counts)
#     return [label for label, y in zip(hist_label, y_pred) if y != most_freq]


def predict_selection_grey_bg(hist_X, hist_label):
    idx = np.argmax(np.argmax(hist_X, axis=1))
    return hist_label[idx]

def predict_selection_blue_bg(hist_X,hist_label):
  idx = np.argmin(np.argmax(hist_X,axis=1))
  return hist_label[idx]

output_dir = Path('./output')
frames_dir = Path('./frames')


if __name__=='__main__':
    frame_folders = [f for f in frames_dir.glob("*") if f.is_dir()]
    with open('./bg.txt','r') as f:
        grey_bg_ids = f.readlines()
    grey_bg_ids = [line.strip() for line in grey_bg_ids]
    for vid in tqdm(frame_folders):
        selection_func = predict_selection_blue_bg
        for grey_bg_id in grey_bg_ids:
            if grey_bg_id in vid.name:
                selection_func = predict_selection_grey_bg
        images = list(vid.glob("*.png"))
        image_ids = []
        selections = []
        for img_file in tqdm(images):
            img = cv2.imread(str(img_file))
            ocr_result = ocr(str(img_file), save_pred=False)  # save_vis=True
            match_dict, stat = match_ocr(ocr_result)
            if stat == -1:
                # result_dict[img_file.name] = []
                continue
            menu = menu_dict if stat == 0 else menu_dict2
            hist_X, hist_label = build_hist_X(img, menu, match_dict, ocr_result)
            selection = selection_func(hist_X, hist_label)
            # result_dict[img_file.name] = selection
            image_ids.append(img_file.name)
            selections.append(selection)
        df = pd.DataFrame(data={'frame':image_ids,'selection':selections})
        df.to_csv(output_dir/f'{vid.name}_selection.csv',index=False)



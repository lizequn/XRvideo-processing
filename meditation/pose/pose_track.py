from pathlib import Path
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm
DATA_FOLDER = Path("./data/pose")

videos = list(DATA_FOLDER.glob("*.mp4"))

inferencer = MMPoseInferencer('wholebody',device='cuda')

for video_file in tqdm(videos):
    generator = inferencer(str(video_file),pred_out_dir='./data/pred_out',vis_out_dir='./data/vis_out')
    for i in generator:
        pass



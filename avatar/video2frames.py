import argparse
from pathlib import Path
import invoke
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", type=str)
parser.add_argument("--fps",type=str)
args = parser.parse_args()
FILE_PATH = Path(args.source)
assert FILE_PATH.exists() or "Source folder do not exists"
SAVE_PATH = Path("./frames")
SAVE_PATH.mkdir(exist_ok=True)

extensions = ['.mov', '.mp4','.mkv']

def process_video(video_file:Path, dest:Path,fps:int):
    assert fps>=1
    assert video_file.exists() or "video not exist"
    file_name = video_file.stem
    output_folder = dest/ file_name
    output_folder.mkdir(exist_ok=True)
    cmd = f'ffmpeg -i "{str(video_file)}" -vf fps={fps} "{str(output_folder)}/{file_name}_%04d.png"'
    print(cmd)
    result = invoke.run(cmd, hide=False, warn=True)
    return result.ok

files = [f for f in FILE_PATH.rglob("*.*") if f.is_file() and f.suffix.lower() in extensions]

for video_file in tqdm(files):
    process_video(video_file,SAVE_PATH,int(args.fps))


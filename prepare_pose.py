from pathlib import Path
import invoke
from tqdm.auto import tqdm

SOURCE_FOLDER = Path('./meditation')
OUTPUT_FOLDER = Path('./meditation/pose')
OUTPUT_FOLDER.mkdir(exist_ok=True)

video_files = [f for f in SOURCE_FOLDER.glob("*.mp4")]
print(f"# of videos {len(video_files)}")

cmd = 'ffmpeg -i input.mp4 -vn -acodec libmp3lame output.mp3'


def run():
    for video in tqdm(video_files):
        file_name = video.stem
        output_name = file_name + ".mp4"
        cmd = f'ffmpeg -y -vsync 0 -i "{str(video)}" -c:v hevc_nvenc -cq 24 -filter:v fps=fps=10 "{str(OUTPUT_FOLDER / output_name)}"'
        invoke.run(cmd, hide=True)


if __name__ == "__main__":
    run()

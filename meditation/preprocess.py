import argparse
from pathlib import Path
import invoke
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--rotate", "-r", action='store_true')
parser.add_argument("--encode","-r",action='store_true')
args = parser.parse_args()
if not args.rotate:
    FILE_PATH = Path("E:\\OneDrive - Bournemouth University\\Aspire study videos\\University Recordings")
else:
    FILE_PATH = Path("./data_h264")
    with open('./rotate_list.txt','r') as f:
        rotate_list = f.readlines()
        print(f"# of file to rotate {len(rotate_list)}")
SAVE_PATH = Path("./data_h264_rotated")
SAVE_PATH.mkdir(exist_ok=True)
LOG_PATH = Path("./preprocess.log")
assert FILE_PATH.exists()
extensions = ['.mov', '.mp4']




def process_video(source: Path, dest: Path, rotate: bool = False, vformat: str = 'h264'):
    file_name = source.stem
    if rotate:
        file_name = f"{file_name}_rotated"
    else:
        file_name = f"{file_name}_h264"
    file_name += ".mp4"
    output_path = dest / file_name
    if output_path.exists():
        return
    if vformat == 'h264':
        encoder = 'h264_nvenc'
    else:
        encoder = 'libx265'
    if rotate:
        cmd = f'ffmpeg -i "{str(source)}" -c:v {encoder} -vf "transpose=1" -c:a copy "{str(output_path)}" '
    else:
        if vformat == 'h264':
            cmd = f'ffmpeg -y -vsync 0 -i "{str(source)}" -c:a copy -c:v {encoder} -b:v 5M -r 30 "{str(output_path)}"'
        else:
            cmd = f'ffmpeg -i "{str(source)}" -c:v {encoder} -preset fast -crf 28 -filter:v fps=fps=30 -c:a copy "{str(output_path)}"'
    result = invoke.run(cmd, hide=True, warn=True)
    if result.ok:
        with open(LOG_PATH, 'a') as f:
            f.write(f"filename: {file_name}\n")
            f.write(result.stdout)
            f.write("\n")


def run():
    files = [f for f in FILE_PATH.rglob("*.*") if f.is_file() and f.suffix.lower() in extensions]
    files_med = []
    for f in files:
        if 'swim' not in str(f).lower():
            files_med.append(f)
    print(f"# of files {len(files_med)}")
    for file in tqdm(files_med):
        process_video(file, SAVE_PATH, rotate=False)


def run_rotate():
    files = [f for f in FILE_PATH.rglob("*.*") if f.is_file() and f.suffix.lower() in extensions]
    to_rotate =[]
    for file in rotate_list:
        file_name = file.split('\t')[0].strip()
        for path in files:
            if file_name in path.name.lower() and 'rotate' not in path.name.lower():
                to_rotate.append(path)
    print(len(to_rotate))
    for path in tqdm(to_rotate):
        process_video(path,SAVE_PATH,rotate=True)


if __name__ == "__main__":
    if args.rotate:
        run_rotate()
    if args.encode:
        run()

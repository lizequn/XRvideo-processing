import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--compress","-c",type=bool)
args = parser.parse_args()
FILE_PATH = Path("E:\\OneDrive - Bournemouth University\\Aspire study videos\\University Recordings")
SAVE_PATH = Path("./data")
assert FILE_PATH.exists()
extensions = ['.mov','.mp4']
files = [ f for f in FILE_PATH.rglob("*.*") if f.is_file() and f.suffix.lower() in extensions]
files_med = []
for f in files:
    if 'swim' not in str(f).lower():
        files_med.append(f)
print(f"# of files {len(files_med)}")


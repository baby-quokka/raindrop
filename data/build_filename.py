import os
from pathlib import Path
from typing import Dict, List, Iterable
from tqdm import tqdm

def list2txt(paths: List[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open('w', encoding="utf-8") as f:
        for p in paths:
            f.write(p + '\n')

def build_filename(
    dataset_map: Dict[str, str],
    out_txt: Path,
    filters: Iterable[str] = (".png", ".jpg", ".jpeg"),
    per_dataset: bool = True    
):
    filt = {e.lower() if e.startswith(".") else "." + e.lower() for e in filters}
    total_paths: List[str] = []
    per_counts: Dict[str, int] = {}

    for ds_name, ds_dirs in dataset_map.items():
        ds_paths: List[str] = []
        dp = Path(ds_dirs).resolve()
        if not dp.exists():
            print(f"[Warning] Dataset path {dp} does not exist.")
            continue
        for cur_dir, _, files in os.walk(dp):
            for file in tqdm(files, desc=f"{ds_name}"):
                p = Path(cur_dir) / file
                if p.is_file() and p.suffix.lower() in filt:
                    ds_paths.append(str(p.resolve()))
        per_counts[ds_name] = len(ds_paths)
        if per_dataset:
            list2txt(ds_paths, out_txt / f"{ds_name}.txt")
        total_paths.extend(ds_paths)
    if not per_dataset:
        list2txt(total_paths, out_txt / "filename.txt")
    return per_counts, len(total_paths)

if __name__ == "__main__":
    try:
        DEFAULT_SAVE_DIR = Path(__file__).resolve().parent
    except NameError:
        DEFAULT_SAVE_DIR = Path.cwd()
    ROOT_DIR = "/root/dataset/ImageRestoration/RaindropClarity_NTIRE/"
    #ROOT_DIR = "/root/dataset/ImageRestoration/Dehazing/"

    TRAIN_DATASET = {
        "DayRainDrop_Train": os.path.join(ROOT_DIR, "DayRainDrop_Train/DayRainDrop_Train/Drop"),
        "NightRainDrop_Train": os.path.join(ROOT_DIR, "NightRainDrop_Train/NightRainDrop_Train/Drop")
        #"NH-HAZE": os.path.join(ROOT_DIR, "NH-HAZE/train/input")
    }

    per_counts, total = build_filename(
        dataset_map=TRAIN_DATASET,
        out_txt=DEFAULT_SAVE_DIR / 'train',
        filters=(".png", ".jpg", ".jpeg"),
        per_dataset=False,
    )
    print(f"[info] | {per_counts} | total={total}")


    TEST_DATASET = {
        "RainDrop_Valid" : os.path.join(ROOT_DIR, "RainDrop_Valid/RainDrop_Valid"),
        #"NightRainDrop_Test" : os.path.join(ROOT_DIR, "NightRainDrop_Test/input")
        #"NH-HAZE": os.path.join(ROOT_DIR, "NH-HAZE/valid/input")
    }

    per_counts, total = build_filename(
        dataset_map=TEST_DATASET,
        out_txt=DEFAULT_SAVE_DIR / 'test',
        filters=(".png", ".jpg", ".jpeg"),
        per_dataset=False,
    )
    print(f"[info] | {per_counts} | total={total}")
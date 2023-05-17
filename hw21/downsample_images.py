# %%
from pathlib import Path

import cv2
import albumentations as A


# %%
def rescale_sample(image, mppxl_current, mppxl_target):
    if not mppxl_current == mppxl_target:
        scale_factor = mppxl_current / mppxl_target
        aug = A.RandomScale(
            scale_limit=[scale_factor - 1.0, scale_factor - 1.0],
            interpolation=cv2.INTER_AREA,
            p=1,
        )
        image = aug(image=image)["image"]
    return image


def load_image(fn):
    image = cv2.imread(str(fn))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(array, fn="./unnamed.png", dtype="uint8"):
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR).astype(dtype)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    cv2.imwrite(str(fn), image, params)


def main(path_in: Path, path_out: Path):
    path_in.mkdir(parents=True, exist_ok=True)
    path_out.mkdir(parents=True, exist_ok=True)
    for fn_in in path_in.iterdir():
        fn_out = path_out / fn_in.name
        img_in = load_image(fn_in)
        img_out = rescale_sample(img_in, 1 / 1024, 1 / 256)
        save_image(img_out, fn_out)


# %%
if __name__ == "__main__":
    path_root = Path("/data/data.sets/free/")
    path_in = path_root / "faces_dataset_small_1024x1024"
    path_out = path_root / "faces_dataset_small_128x128"
    path_out = path_root / "faces_dataset_small_256x256"
    main(path_in, path_out)

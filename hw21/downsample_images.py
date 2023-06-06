# %%
from pathlib import Path

import cv2
import albumentations as A


# %%
def rescale_image(image, mppxl_current, mppxl_target):
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


def main(path_in, path_out, mppxl_current=1., mppxl_target=1.):
    path_out.mkdir(parents=True, exist_ok=True)
    for fn_in in path_in.rglob(pattern='*.png'):
        fn_out = path_out / fn_in.relative_to(path_in)
        (fn_out.parent).mkdir(parents=True, exist_ok=True)
        img_in = load_image(fn_in)
        img_out = rescale_image(img_in, mppxl_current, mppxl_target)
        save_image(img_out, fn_out)


# %%
if __name__ == "__main__":
    path_root = Path("/data/data.sets.raw/GANs-ffhq-Faces/")
    H_current, W_current = 1024, 1024
    H_target, W_target = 256, 256
    mppxl_current = 1.0 / H_current
    mppxl_target = 1.0 / H_target
    path_in = path_root / f"images{H_current}x{W_current}"
    path_out = path_root / f"images{H_target}x{W_target}"
    main(path_in, path_out, mppxl_current, mppxl_target)

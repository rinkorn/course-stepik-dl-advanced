# %%
import pandas as pd
from pathlib import Path
import torch  # noqa: F401
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_wider_face_train_bbx_gt(fn):
    with open(str(fn)) as f:
        lines = [line.rstrip(" \n") for line in f.readlines()]

    name_indices = [i for i, line in enumerate(lines) if "--" in line]

    wider_list = []
    for idx in name_indices:
        name = lines[idx]
        amount = int(lines[idx + 1])
        amount = 1 if amount == 0 else amount
        ground_truth = lines[idx + 2 : idx + 2 + amount]
        for gt in ground_truth:
            row = [name, *[int(item) for item in gt.split(" ")]]
            wider_list.append(row)

    wider_df = pd.DataFrame(
        wider_list,
        columns=[
            "filename",
            "x1",
            "y1",
            "w",
            "h",
            "blur",
            "expression",
            "illumination",
            "invalid",
            "occlusion",
            "pose",
        ],
    )
    wider_df.set_index("filename", inplace=True)
    wider_df.sort_index(inplace=True)
    return wider_df


class WiderDataset(Dataset):
    def __init__(self, path_images, path_labels="wider_face_train_bbx_gt.txt"):
        self.filenames = list(Path(path_images).rglob("*.jpg"))
        self.labels = parse_wider_face_train_bbx_gt(path_labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fn = self.filenames[index]
        fn_key = f"{fn.parent.name}/{fn.name}"
        image = Image.open(fn)
        values = self.labels.loc[[fn_key]].values
        return image, values

    def imshow_item(self, index):
        image, values = self[index]
        fig, ax = plt.subplots()
        ax.imshow(image)
        for item in values:
            x, y, w, h = item[:4]
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()


# %%
if __name__ == "__main__":
    path_data = Path("/data/data.sets/free/ObjectDetection_Faces_WIDER")
    path_images = Path(path_data / "WIDER_train/images/")
    path_labels = Path(path_data / "wider_face_split/wider_face_train_bbx_gt.txt")
    ds = WiderDataset(path_images, path_labels)
    img, values = ds[1001]
    print(len(ds.filenames))
    print(len(ds.labels.index.unique()))
    print(ds.labels.iloc[:10, :])
    print(ds.labels.loc[["0--Parade/0_Parade_marchingband_1_799.jpg"]])
    ds.imshow_item(1001)

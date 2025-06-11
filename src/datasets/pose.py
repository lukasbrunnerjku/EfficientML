from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from typing import Optional


def is_visible(kp, img_shape):
    x, y, v = kp
    h, w = img_shape[:2]
    return v if (0 <= x < w and 0 <= y < h) else 0


class PoseDataset(Dataset):
    def __init__(
        self,
        input_folder: Path,
        meta_folder: Path,
        kpt_shape: tuple[int, int],
        kpt_labels: list[str],
        file_ext: str = "png",
        meta_ext: str = "txt",
        transforms: Optional[list[A.BasicTransform]] = None,
    ):
        self.input_files = sorted(input_folder.glob(f"*.{file_ext}"))
        self.meta_files = sorted(meta_folder.glob(f"*.{meta_ext}"))
        assert len(self.input_files) == len(self.meta_files), "Lenght of input and meta files has to match."
        input_file_names = set((x.stem for x in self.input_files))
        assert input_file_names == set((x.stem for x in self.meta_files)), "File names of input and meta have to match."
        self.kpt_shape = kpt_shape
        assert self.kpt_shape[1] in (2, 3), "Keypoints with either xy or xyv expected."
        self.kpt_labels = kpt_labels
        assert len(self.kpt_labels) == self.kpt_shape[0], "Keypoint labels do not match shape."
        self.transform = A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )
        
    def __len__(self):
        return len(self.input_files)
    
    def _load_image(self, file) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
    
    def _load_target(self, file, sep: str = " "):
        """
        In case of xy keypoint annotations we would have:
        <px1> <py1> <px2> <py2> ... <pxn> <pyn>
        or in case of xyv annotations: 
        <px1> <py1> <pv1> <px2> <py2> <pv2> ... <pxn> <pyn> <pvn>
        and each object in the image is annotated in a seperate line.
        """
        xyv = []
        with open(file, "r") as fp:
            objects = fp.readlines()
            for object in objects:
                values = object.split(sep)
                xyv.append(values)
                
        xyv = np.array(xyv).reshape(-1, self.kpt_shape[1])
        return xyv
    
    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        meta_file = self.meta_files[idx]
        image = self._load_image(input_file)
        xyv = self._load_target(meta_file)
        t = self.transform(image=image, keypoints=xyv[:, :2])
        image = t["image"]
        xyv[:, :2] = t["keypoints"]
        xyv[:, 2] = [is_visible(kp, image.shape) for kp in xyv]
        return {
            "image": image,
            "xyv": xyv,
        }
    
  
if __name__ == "__main__":
    pass

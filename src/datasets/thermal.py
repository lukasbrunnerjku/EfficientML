from pathlib import Path
import cv2
import torch
from torch import Tensor

from .image_to_image import ImageToImageBase


def get_meta(meta_file: Path):
    """
    Read temperature min/max values to transform *.png
    tone mapped thermal images to raw temperature values in Kelvin.
    
    Returns:
        min_input, max_input, min_output, max_output
    """
    with open(meta_file, "r") as fp:
        line = fp.read()
        values = line.split(",")
        
    min_input, max_input, min_output, max_output = [float(t) for t in values[:4]]
    env_temp = int(values[4])
    return min_input, max_input, min_output, max_output, env_temp


def load_image(file: Path, min_temp: float, max_temp: float) -> Tensor:
    x = torch.from_numpy(cv2.imread(str(file), -1)[:, :, 0])  # HxW; [0, 255]
    x = ((max_temp - min_temp) * (x / 255)) + min_temp  # in Kelvin
    return x


class ThermalDataset(ImageToImageBase):
    
    def load_from_files(self, input_file: Path, output_file: Path, meta_file: Path):
        min_input, max_input, min_output, max_output, env_temp = get_meta(meta_file)
        input_tensor = load_image(input_file, min_input, max_input)
        output_tensor = load_image(output_file, min_output, max_output)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "condition": env_temp,
        }
        
        

if __name__ == "__main__":
    # ds = ThermalDataset()
    
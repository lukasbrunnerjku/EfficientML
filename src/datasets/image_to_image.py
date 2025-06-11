from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Any
from abc import ABC, abstractmethod


class ImageToImageBase(Dataset, ABC):
    def __init__(
        self,
        input_folder: Path,
        output_folder: Path,
        meta_folder: Optional[Path] = None,
        file_ext: str = "png",
        meta_ext: str = "txt",
    ):
        self.input_files = sorted(input_folder.glob(f"*.{file_ext}"))
        self.output_files = sorted(output_folder.glob(f"*.{file_ext}"))
        
        assert len(self.input_files) == len(self.output_files), "Lenght of input and output files has to match."
        input_file_names = set((x.stem for x in self.input_files))
        assert input_file_names == set((x.stem for x in self.output_files)), "File names of input and output have to match."
        
        self.meta_files = None
        if meta_folder is not None:
            self.meta_files = sorted(meta_folder.glob(f"*.{meta_ext}"))
            assert len(self.input_files) == len(self.meta_files), "Lenght of input and meta files has to match."
            
    def __len__(self):
        return len(self.input_files)
    
    @abstractmethod
    def load_from_files(self, input_file: Path, output_file: Path, meta_file: Optional[Path]) -> dict[str, Any]:
        pass  # TODO: Return a dictionary ie. {"input": image_tensor, "output": image_tensor,... }
    
    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        output_file = self.output_files[idx]
        meta_file = None
        if self.meta_files is not None:
            meta_file = self.meta_files[idx]
        return self.load_from_files(input_file, output_file, meta_file)  
    

if __name__ == "__main__":
    ds = ImageToImageBase()
    
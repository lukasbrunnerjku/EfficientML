import json
from pathlib import Path


class ConvertCOCO:
    
    def __init__(
        self,
        coco_file: Path,
        category_id: int,
        meta_path: Path,
    ):
        # coco_dir = coco_file.parent
        with open(coco_file, "r") as fp:
            coco = json.load(fp)
            
        images = coco["images"]
        annotations = coco["annotations"]
        for image in images:
            file_name = image["file_name"]
            image_id = image["id"]
            
            _annotations = []
            for annotation in annotations:
                if category_id != annotation["category_id"]:
                    continue
                if image_id == annotation["image_id"]:
                    _annotations.append(annotation)
                    
            objs = []
            for annotation in _annotations:
                xyv = annotation["keypoints"]
                obj = " ".join(f"{x:.2f} {y:.2f} {int(v > 0)}" for x, y, v in xyv[::3])
                objs.append(obj)
            
            meta_file = (meta_path / file_name).with_suffix("txt")
            with open(meta_file, "w") as fp:
                fp.writelines(objs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_file", type=Path)
    parser.add_argument("meta_path", type=Path)
    parser.add_argument("--category_id", type=int, default=1)
    args = parser.parse_args()
    
    cc = ConvertCOCO(args.coco_file, args.category_id, args.meta_path)
    
from nnaf._types import *

def build_voc07_wds(root: PathOrStr):
    from .database import write_wds_data, NodupHash
    import xml.etree.ElementTree as ET
    import json

    root = Path(root)
    if (root / "VOCdevkit").exists():
        root = root / "VOCdevkit" / "VOC2007"            
    
    hash_func = NodupHash()

    def process_func(args: tuple[Path, Path]):
        img_path, ann_path = args
        img = img_path.read_bytes()
        
        anns = []
        for ann in ET.parse(str(ann_path)).getroot().findall("object"):
            difficult = int(ann.find("difficult").text)
            label = ann.find("name").text
            bbox = ann.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            anns.append(dict(
                label=label,
                bbox=[xmin, ymin, xmax, ymax],
                bbox_format="xyxy",
                difficult=difficult
            ))
        anns = json.dumps(anns)

        nonlocal hash_func
        key = hash_func(img_path.resolve().as_uri())
        return {
            "__key__": key,
            "jpeg": img,
            "annotations": anns,
        }
    
    for part in (root / "ImageSets" / "Layout").iterdir():
        part_name = part.with_suffix("").name

        def source(part: Path):
            with part.open("r") as f:
                while True:
                    idx = f.readline().strip()
                    if not idx:
                        break
                    img_path = (root / "JPEGImages" / idx).with_suffix(".jpg")
                    ann_path = (root / "Annotations" / idx).with_suffix(".xml")
                    if img_path.is_file() and ann_path.is_file():
                        yield (img_path, ann_path)
                    else:
                        print(f"Image or Annotation for {idx} not found, skipping.")
                    yield img_path, ann_path

        write_wds_data(
            source=source(part),
            target_path=root / "webdataset" / part_name,
            process_func=process_func,
            parallel_njobs=1,
            parallel_backend="threading",
            refresh=True,
        )
        hash_func.dump_hashed_keys(root / "webdataset" / part_name / "keys.json")

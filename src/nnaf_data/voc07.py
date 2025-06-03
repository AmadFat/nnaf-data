from nnaf_utils.parallel import JoblibConfig
from nnaf_utils.pytype import *

from .basic import TarConfig


def _search_voc07(root: StrPath) -> Path:
    root = Path(root)

    if (root / "VOCdevkit").exists():
        return root / "VOCdevkit" / "VOC2007"

    if (root / "VOC2007").exists() and (root / "VOC2007" / "VOCdevkit").exists():
        return root / "VOC2007" / "VOCdevkit" / "VOC2007"
    
    return root


def build_voc07_wds(
    root: StrPath,
    refresh_target: bool = False,
    tar_config: TarConfig = None,
    joblib_config: JoblibConfig = None,
    success_callback: Callable[[str], None] = None,
    error_callback: Callable[[str], None] = None,
):
    """Build VOC07 WebDataset."""
    try:
        import json
        import os
        import xml.etree.ElementTree as ET

        from nnaf_utils.encoding import NodupRehashPool
        from xxhash import xxh64_hexdigest

        from .basic import write_wds_data

        root = _search_voc07(root)
        hash_pool = NodupRehashPool()

        def hash_func(x: str) -> str:
            return xxh64_hexdigest(x.encode())
        
        def rehash_func(x: str) -> str:
            return xxh64_hexdigest(x.encode() + os.urandom(16))

        def fn(args: tuple[Path, Path]):
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

            key = hash_pool(
                img_path.resolve().as_uri(),
                hash_func=hash_func,
                rehash_func=rehash_func,
            )

            return {
                "__key__": key,
                "jpeg": img,
                "annotations": anns,
            }
        
        for part in (root / "ImageSets" / "Layout").iterdir():
            part_name = part.with_suffix("").name

            def source(part: Path):
                with part.open() as f:
                    while True:
                        idx = f.readline().strip()
                        if not idx:
                            break
                        img_path = (root / "JPEGImages" / idx).with_suffix(".jpg")
                        ann_path = (root / "Annotations" / idx).with_suffix(".xml")
                        if img_path.is_file() and ann_path.is_file():
                            yield (img_path, ann_path)
                        else:
                            msg = f"Image or Annotation for {idx} not found."
                            if error_callback:
                                error_callback(msg, "Skipping.")
                            else:
                                raise FileNotFoundError(" ".join([msg, "Exiting."]))
                        yield img_path, ann_path

            write_wds_data(
                source=source(part),
                target_path=root / "webdataset" / part_name,
                refresh_target=refresh_target,
                fn=fn,
                tar_config=tar_config or TarConfig(),
                joblib_config=joblib_config or JoblibConfig(),
                success_callback=success_callback,
                error_callback=error_callback,
            )
            hash_pool.dump(root / "webdataset" / part_name / "keys.json")
        
        if success_callback:
            success_callback(f"VOC2007 WebDataset built: {root.as_posix()}.")

    except Exception as e:
        if error_callback:
            error_callback(e)
        else:
            raise e

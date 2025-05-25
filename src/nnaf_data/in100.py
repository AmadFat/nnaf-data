from nnaf._types import *


def build_in100_wds(root: PathOrStr):
    from .database import write_wds_data, NodupHash

    hash_func = NodupHash()

    def process_func(img_path: Path):
        img = img_path.read_bytes()

        label = img_path.parent.name

        nonlocal hash_func
        key = hash_func(img_path.resolve().as_uri())

        return {
            "__key__": key,
            "jpeg": img,
            "label": label,
        }

    for part in ["train", "val"]:
        target_path = Path(root) / "webdataset" / part
        write_wds_data(
            source=Path(root).glob(f"{part}*/*/*.JPEG"),
            target_path=target_path,
            process_func=process_func,
            parallel_njobs=2,
            parallel_backend="threading",
            refresh=True,
        )
        hash_func.dump_hashed_keys(target_path / "keys.json")

from nnaf_utils.parallel import JoblibConfig
from nnaf_utils.pytype import *

from .basic import TarConfig


def build_in100_wds(
    root: StrPath,
    refresh_target: bool = False,
    tar_config: TarConfig = None,
    joblib_config: JoblibConfig = None,
    success_callback: Callable[[str], None] = None,
    error_callback: Callable[[str], None] = None,
):
    """Build ImageNet-100 WebDataset."""
    try:
        import os

        from nnaf_utils.encoding import NodupRehashPool
        from xxhash import xxh64_hexdigest

        from .basic import write_wds_data

        root = Path(root)
        hash_pool = NodupRehashPool()

        def hash_func(x: str) -> str:
            return xxh64_hexdigest(x.encode())

        def rehash_func(x: str) -> str:
            return xxh64_hexdigest(x.encode() + os.urandom(16))
        
        def fn(imgpath: Path):
            img = imgpath.read_bytes()

            label = imgpath.parent.name

            key = hash_pool(
                imgpath.resolve().as_uri(),
                hash_func=hash_func,
                rehash_func=rehash_func,
            )

            return {
                "__key__": key,
                "jpeg": img,
                "label": label,
            }

        for part in ["train", "val"]:
            target_path = root / "webdataset" / part
            write_wds_data(
                source=root.glob(f"{part}.*/*/*.JPEG"),
                target_path=target_path,
                refresh_target=refresh_target,
                fn=fn,
                tar_config=tar_config or TarConfig(),
                joblib_config=joblib_config or JoblibConfig(),
                success_callback=success_callback,
                error_callback=error_callback,
            )
            hash_pool.dump(root / "webdataset" / part / "keys.json")
        
        if success_callback:
            success_callback(f"ImageNet-100 WebDataset built: {root.as_posix()}.")

    except Exception as e:
        if error_callback:
            error_callback(e)
        else:
            raise e

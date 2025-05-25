from nnaf._types import *
from nnaf.miscs import refresh_dir
import os, pwd, xxhash


class NodupHash:
    def __init__(
        self,
        existed_hashes = None,
    ):
        self.hash_set = set(existed_hashes or [])
        self.hash_func = xxhash.xxh64_hexdigest
    
    def __call__(self, key_to_hash: str) -> str:
        key = self.hash_func(key_to_hash.encode())
        while key in self.hash_set:
            key = self.hash_func(key.encode() + b"omh")
        self.hash_set.add(key)
        return key
    
    def dump_hashed_keys(self, path: PathOrStr):
        import json
        with Path(path).open("w") as f:
            json.dump(list(self.hash_set), f)


def write_wds_data(
    source: Iterable,
    target_path: PathOrStr,
    process_func: Callable[..., dict[str, str | bytes]] = None,
    parallel_njobs: int = 2,
    parallel_backend: str = "threading",
    shard_pattern: str = "%08d",
    shard_maxcount: int = 1000,
    shard_maxsize: int = 1 << 28,
    shard_user: str = pwd.getpwuid(os.getuid()).pw_name,
    shard_group: str = "nnaf",
    refresh: bool = False,
):
    import webdataset as wds
    from alive_progress import alive_bar
    from joblib import Parallel, delayed

    if refresh:
        refresh_dir(target_path)

    sink = wds.ShardWriter(
        pattern=(Path(target_path) / shard_pattern).with_suffix(".tar").resolve().as_posix(),
        maxcount=shard_maxcount,
        maxsize=shard_maxsize,
        user=shard_user,
        group=shard_group,
        keep_meta=False,
    )

    if process_func is not None:
        parallel = Parallel(n_jobs=parallel_njobs, return_as="generator", backend=parallel_backend)
        pool = parallel(delayed(process_func)(src) for src in source)
    else:
        pool = source

    with alive_bar() as bar:
        for i, d in enumerate(pool, 1):
            sink.write(d)
            bar()
    
    sink.close()
    pool.close()

    try:
        import nvidia.dali
        import subprocess, multiprocessing
        from joblib import Parallel, delayed

        shard_paths = Path(target_path).glob("*.tar")
        parallel = Parallel(n_jobs=multiprocessing.cpu_count(), backend="loky", return_as="generator")
        pool = parallel(delayed(subprocess.run)(["wds2idx", path.resolve().as_posix()]) for path in shard_paths)
        for result in pool:
            if result.returncode != 0:
                raise RuntimeError(f"wds2idx failed with error code {result.returncode}")
    except ImportError:
        import warnings
        warnings.warn("DALI not installed, skipping DALI pipeline generation.")

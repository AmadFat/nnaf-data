from dataclasses import dataclass

import webdataset as wds
from nnaf_utils.parallel import JoblibConfig
from nnaf_utils.pytype import *


@dataclass
class TarConfig:
    """Configuration for writing webdataset tar file(s).

    Attributes:
        user (str): User name. Default: :attr:`pwd.getpwuid(os.getuid()).pw_name`.
        group (str): Group name. Default: ``"nnaf"``.
        mode (int): File mode. Default: ``0o0444``.
        pattern (str): Shard pattern. Default: ``"%08d"``.
        maxcount (int): Maximum counts per shard. Default: ``None``.
        maxsize (int): Maximum bytes size per shard. Default: ``None``.

    Notes:
        If :args:`maxcount` or :args:`maxsize` is provided, use
        :class:`webdataset.ShardWriter`. If not, use :class:`webdataset.TarWriter`.

    """

    import os
    import pwd

    user: str = pwd.getpwuid(os.getuid()).pw_name
    group: str = "nnaf"
    mode: int = 0o0444
    pattern: str = "%08d"
    maxcount: int = None
    maxsize: int = None


def _create_wds_sink(
    target_path: StrPath,
    user: str,
    group: str,
    mode: int = 0o0444,
    pattern: str = "%08d",
    maxcount: int = None,
    maxsize: int = None,
) -> wds.TarWriter | wds.ShardWriter:
    match [maxcount is not None, maxsize is not None]:
        case [True, _] | [_, True]:
            return wds.ShardWriter(
                (Path(target_path) / pattern).with_suffix(".tar").resolve().as_posix(),
                maxcount=maxcount or float("inf"),
                maxsize=maxsize or float("inf"),
                user=user,
                group=group,
                mode=mode,
            )

        case [False, False]:
            return wds.TarWriter(
                (Path(target_path) / (pattern % 1)).with_suffix(".tar").resolve().as_posix(),
                user=user,
                group=group,
                mode=mode,
            )


def write_wds_data(
    source: Iterable,
    target_path: StrPath,
    refresh_target: bool = False,
    fn: Callable[..., dict[str, str | bytes]] = None,
    tar_config: TarConfig = None,
    joblib_config: JoblibConfig = None,
    success_callback: Callable[[Any], None] = None,
    error_callback: Callable[[Any], None] = None,
):
    try:
        import subprocess
        from dataclasses import asdict

        from alive_progress import alive_bar

        match refresh_target:
            case True:
                from nnaf_utils.filesystem import refresh_obj

                refresh_obj(target_path, success_callback=success_callback, error_callback=error_callback)

            case False:
                if not Path(target_path).is_dir():
                    raise NotADirectoryError(f"Impossible path: {target_path} is not a directory.")
                if any(Path(target_path).iterdir()):
                    raise FileExistsError(f"Impossible directory: {target_path} is not empty.")

        Path(target_path).mkdir(parents=True, exist_ok=True)
        sink = _create_wds_sink(target_path, **asdict(tar_config or TarConfig()))

        if fn:
            from nnaf_utils.parallel import create_parallel_executor, delayed

            executor = create_parallel_executor(**asdict(joblib_config or JoblibConfig()))
            pool = executor(delayed(fn)(data) for data in source)
        else:
            executor = None
            pool = source

        with alive_bar() as bar:
            for data in pool:
                sink.write(data)
                bar()

        sink.close()
        pool.close()

    except Exception as e:
        match error_callback is not None:
            case True:
                error_callback(e)
            case False:
                raise e

    try:
        import nvidia.dali

    except (ImportError, ModuleNotFoundError):
        import warnings

        warnings.warn("Importing DALI fails, skipping DALI pipeline generation.", ImportWarning, stacklevel=2)
        return

    try:
        shardpaths = Path(target_path).glob("*.tar")
        if executor is None:
            executor = create_parallel_executor(**asdict(joblib_config or JoblibConfig()))
        pool = executor(delayed(subprocess.run)(["wds2idx", path.as_posix()]) for path in shardpaths)
        for result in pool:
            result: subprocess.CompletedProcess
            if result.returncode != 0:
                raise RuntimeError

    except Exception as e:
        match error_callback is not None:
            case True:
                error_callback(e)
            case False:
                raise e

    if success_callback:
        success_callback(f"Data written successfully to {target_path}.")

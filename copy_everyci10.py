#!/usr/bin/env python3
"""逐个、可恢复地复制 everyci10 数据文件。

文件先复制为隐藏的 .part 文件，确认写入完成后再原子改名。只有原子改名
成功的文件才会写入完成清单。
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path("/data/dn/FRTP_revision1/imagecls/recons_data/everyci10")
DEFAULT_DESTINATION = Path("/mnt/newdisk/FRPT0726/recons_data/everyci10")
MANIFEST_NAME = ".copy_completed.tsv"
LOCK_NAME = ".copy_everyci10.lock"
RESERVE_NAME = ".copy_log_reserve"
RESERVE_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 16 * 1024 * 1024


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐个复制文件，记录完成项，并在目标盘空间不足时清理半成品后停止。"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="复制后重新读取目标文件并核对 SHA-256（更可靠，但耗时更久）",
    )
    return parser.parse_args()


def source_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def load_completed(manifest: Path) -> dict[str, tuple[int, str]]:
    completed: dict[str, tuple[int, str]] = {}
    if not manifest.exists():
        return completed
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 4 or columns[0] == "completed_utc":
                continue
            _, relative_name, size_text, digest = columns
            try:
                completed[relative_name] = (int(size_text), digest)
            except ValueError:
                print(f"警告：忽略完成清单中的无效行：{line.rstrip()}", file=sys.stderr)
    return completed


def append_completed(
    manifest: Path, relative_name: str, size: int, digest: str
) -> None:
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", encoding="utf-8") as stream:
        if new_file:
            stream.write("completed_utc\trelative_path\tsize_bytes\tsha256\n")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stream.write(f"{timestamp}\t{relative_name}\t{size}\t{digest}\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_completed_with_reserve(
    manifest: Path,
    reserve: Path,
    relative_name: str,
    size: int,
    digest: str,
) -> None:
    try:
        append_completed(manifest, relative_name, size, digest)
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        release_log_space(reserve)
        append_completed(manifest, relative_name, size, digest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def reserve_log_space(path: Path) -> None:
    try:
        with path.open("wb") as stream:
            os.posix_fallocate(stream.fileno(), 0, RESERVE_BYTES)
    except (AttributeError, OSError):
        path.unlink(missing_ok=True)


def release_log_space(path: Path) -> None:
    path.unlink(missing_ok=True)


def remove_partial(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        print(f"警告：无法删除未完成文件 {path}: {error}", file=sys.stderr)


def copy_one(source: Path, partial: Path) -> str:
    digest = hashlib.sha256()
    partial.parent.mkdir(parents=True, exist_ok=True)
    remove_partial(partial)
    with source.open("rb", buffering=0) as reader, partial.open(
        "xb", buffering=0
    ) as writer:
        while chunk := reader.read(CHUNK_BYTES):
            remaining = memoryview(chunk)
            while remaining:
                written = writer.write(remaining)
                if written is None or written == 0:
                    raise OSError(errno.EIO, "写入目标文件时没有进展")
                remaining = remaining[written:]
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if partial.stat().st_size != source.stat().st_size:
        raise OSError(errno.EIO, "复制后的文件大小与源文件不一致")
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    source_root = args.source.resolve()
    destination_root = args.destination.resolve()

    if not source_root.is_dir():
        print(f"错误：源目录不存在：{source_root}", file=sys.stderr)
        return 2

    destination_root.mkdir(parents=True, exist_ok=True)
    manifest = destination_root / MANIFEST_NAME
    reserve = destination_root / RESERVE_NAME
    lock_path = destination_root / LOCK_NAME

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("错误：已有一个复制进程正在使用该目标目录。", file=sys.stderr)
            return 2

        completed = load_completed(manifest)
        files = source_files(source_root)
        reserve_log_space(reserve)
        print(f"共发现 {len(files)} 个文件；完成清单：{manifest}")

        for index, source in enumerate(files, start=1):
            relative = source.relative_to(source_root)
            relative_name = relative.as_posix()
            destination = destination_root / relative
            partial = destination.with_name(f".{destination.name}.part")
            source_size = source.stat().st_size

            record = completed.get(relative_name)
            if destination.exists():
                if not destination.is_file():
                    print(
                        f"错误：目标路径已存在但不是普通文件：{destination}",
                        file=sys.stderr,
                    )
                    release_log_space(reserve)
                    return 2

                print(
                    f"[{index}/{len(files)}] 目标文件已存在，正在校验 MD5："
                    f"{relative_name}"
                )
                source_md5 = md5_file(source)
                destination_md5 = md5_file(destination)
                if destination_md5 == source_md5:
                    if record is None or record[0] != source_size:
                        source_digest = sha256_file(source)
                        append_completed_with_reserve(
                            manifest,
                            reserve,
                            relative_name,
                            source_size,
                            source_digest,
                        )
                        completed[relative_name] = (source_size, source_digest)
                    print(
                        f"[{index}/{len(files)}] MD5 一致，跳过：{relative_name}"
                    )
                    continue

                print(
                    f"[{index}/{len(files)}] MD5 不一致，将覆盖：{relative_name}"
                )

            free_bytes = shutil.disk_usage(destination_root).free
            required_bytes = source_size + RESERVE_BYTES
            if free_bytes < required_bytes:
                release_log_space(reserve)
                print(
                    f"空间不足，停止复制：{relative_name} 需要约 "
                    f"{source_size / 2**30:.2f} GiB，当前可用 "
                    f"{free_bytes / 2**30:.2f} GiB。",
                    file=sys.stderr,
                )
                return 3

            print(
                f"[{index}/{len(files)}] 开始复制：{relative_name} "
                f"({source_size / 2**30:.2f} GiB)"
            )
            try:
                digest = copy_one(source, partial)
                if args.verify_sha256:
                    print(f"[{index}/{len(files)}] 正在校验 SHA-256：{relative_name}")
                    if sha256_file(partial) != digest:
                        raise OSError(errno.EIO, "目标文件 SHA-256 校验失败")
                os.replace(partial, destination)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                append_completed_with_reserve(
                    manifest, reserve, relative_name, source_size, digest
                )
                completed[relative_name] = (source_size, digest)
                print(f"[{index}/{len(files)}] 完成：{relative_name}")
            except BaseException as error:
                release_log_space(reserve)
                remove_partial(partial)
                if isinstance(error, KeyboardInterrupt):
                    print("\n收到中断信号，已删除当前未完成文件。", file=sys.stderr)
                    return 130
                if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                    print(
                        "目标硬盘空间已满，已删除当前未完成文件并停止。",
                        file=sys.stderr,
                    )
                    return 3
                print(
                    f"复制失败，已删除当前未完成文件：{relative_name}: {error}",
                    file=sys.stderr,
                )
                return 1

        release_log_space(reserve)
        print("全部文件均已复制完成。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

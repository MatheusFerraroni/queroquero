"""Bounded ZIP central-directory reader; no archive extraction or corpus cache."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import struct
import zlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

from .datasets.base import stable_hash

THREAD = re.compile(r"clear_threads/([0-9]+)(?:_[^/]+)?\.tsv\Z")


def exact_read(handle, size):
    value = handle.read(size)
    if len(value) != size:
        raise BadZipFile("truncated ZIP metadata")
    return value


def central_directory(path: Path):
    """Yield ZipInfo records in archive order, including ZIP64 sizes/offsets.

    Only ordinary single-disk ZIP archives are supported (no prepended data).
    Selected members are subsequently read by stdlib ZipFile, including CRC checks.
    """
    size = path.stat().st_size
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        handle.seek(max(0, size - 65557))
        tail = handle.read()
        end = tail.rfind(b"PK\x05\x06")
        if end < 0 or len(tail) - end < 22:
            raise BadZipFile("missing ZIP end record")
        record = struct.unpack_from("<4s4H2LH", tail, end)
        if record[1] or record[2] or len(tail) - end != 22 + record[-1]:
            raise BadZipFile("unsupported ZIP layout")
        entries, directory_size, offset = record[4:7]
        if entries == 65535 or offset == 0xFFFFFFFF or directory_size == 0xFFFFFFFF:
            if end < 20:
                raise BadZipFile("missing ZIP64 locator")
            locator = struct.unpack_from("<4sLQL", tail, end - 20)
            if locator[0] != b"PK\x06\x07" or locator[1] or locator[3] != 1:
                raise BadZipFile("unsupported ZIP64 layout")
            handle.seek(locator[2])
            extended = struct.unpack("<4sQ2H2L4Q", exact_read(handle, 56))
            if extended[0] != b"PK\x06\x06" or extended[4] or extended[5]:
                raise BadZipFile("invalid ZIP64 end record")
            entries, directory_size, offset = extended[7:10]
        if offset + directory_size > size:
            raise BadZipFile("ZIP directory outside archive")
        handle.seek(offset)
        for index in range(entries):
            raw = exact_read(handle, 46)
            if raw[:4] != b"PK\x01\x02":
                raise BadZipFile("invalid ZIP directory record")
            flags, method = struct.unpack_from("<HH", raw, 8)
            crc, compressed, uncompressed = struct.unpack_from("<LLL", raw, 16)
            name_size, extra_size, comment_size = struct.unpack_from("<HHH", raw, 28)
            disk = struct.unpack_from("<H", raw, 34)[0]
            if disk:
                raise BadZipFile("multi-disk ZIP is unsupported")
            local_offset = struct.unpack_from("<L", raw, 42)[0]
            raw_name = exact_read(handle, name_size)
            name = raw_name.decode("utf-8" if flags & 2048 else "cp437")
            if "\x00" in name:
                raise BadZipFile("NUL in ZIP member name")
            info = ZipInfo(name)
            info.flag_bits, info.compress_type = flags, method
            info.CRC, info.compress_size, info.file_size = crc, compressed, uncompressed
            info.header_offset = local_offset
            info.extra = exact_read(handle, extra_size)
            exact_read(handle, comment_size)
            info._decodeExtra(
                zlib.crc32(raw_name)
            )  # Python 3.12.13 ZIP64/unicode path handling
            info._end_offset = offset
            info._raw_time = struct.unpack_from("<H", raw, 12)[0]
            if info.header_offset >= offset or flags & 1:
                raise BadZipFile("unsafe/encrypted ZIP member")
            yield index, entries, info
        if handle.tell() != offset + directory_size:
            raise BadZipFile("ZIP directory length mismatch")


class IndexedZip(ZipFile):
    """Read only supplied, previously verified ZipInfo; never load the directory."""

    def _RealGetContents(self):
        self.start_dir = 0


def info_record(info):
    return {
        key: getattr(info, key)
        for key in (
            "filename",
            "flag_bits",
            "compress_type",
            "CRC",
            "compress_size",
            "file_size",
            "header_offset",
            "_end_offset",
            "_raw_time",
        )
    }


def record_info(record):
    info = ZipInfo(record["filename"])
    for key, value in record.items():
        setattr(info, key, value)
    return info


def stat_identity(path):
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_members(
    path, source_hashes, expected, *, seed, limit, max_member_bytes, log
):
    """One pass: resolve every CPT source, retain a bounded hash-ordered thread pool.

    Exclusions apply AFTER resolving all source files, including siblings occurring
    later in the archive. One lowest-hash eligible flow per retained thread.
    """
    before = stat_identity(path)
    digest = hashlib.sha256(before["size_bytes"].to_bytes(16, "big"))
    mapping, retained, heap = {}, {}, []
    eligible = 0
    total = 0
    for index, total, info in central_directory(path):
        raw = json.dumps(
            [info.filename, info.CRC, info.compress_size, info.file_size],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        is_member = (
            info.filename.startswith("clear_threads/")
            and info.filename.endswith(".tsv")
            and info.filename.count("/") == 1
        )
        if is_member:
            eligible += 1
            source = hashlib.sha256(
                f"adrenaline/{path.name}/{info.filename}".encode()
            ).hexdigest()
            match = THREAD.fullmatch(info.filename)
            if source in source_hashes:
                if match is None or source in mapping:
                    raise RuntimeError(
                        "CPT source cannot be mapped uniquely to a thread"
                    )
                mapping[source] = stable_hash(
                    "adrenaline-classification-thread/v1", match[1]
                )
            if match is not None and 0 < info.file_size <= max_member_bytes:
                thread = stable_hash("adrenaline-classification-thread/v1", match[1])
                rank = int(stable_hash("intrinsic-holdout-thread/v1", seed, thread), 16)
                member_rank = stable_hash("intrinsic-holdout-flow/v1", seed, source)
                if thread not in retained and (len(heap) < limit or rank < -heap[0][0]):
                    if len(heap) == limit:
                        _, evicted = heapq.heappop(heap)
                        del retained[evicted]
                    heapq.heappush(heap, (-rank, thread))
                    retained[thread] = (member_rank, source, info_record(info), rank)
                if thread in retained and member_rank < retained[thread][0]:
                    retained[thread] = (member_rank, source, info_record(info), rank)
        if (index + 1) % 100000 == 0:
            log(
                f"Índice ZIP: {index + 1:,}/{total:,}; {len(mapping):,} origens resolvidas"
            )
    actual = {
        "archive_size_bytes": before["size_bytes"],
        "central_directory_entries": total,
        "eligible_member_count": eligible,
        "sha256": digest.hexdigest(),
    }
    if any(actual[k] != expected[k] for k in actual) or before != stat_identity(path):
        raise RuntimeError("archive fingerprint changed during holdout preparation")
    if set(mapping) != source_hashes:
        raise RuntimeError("CPT source crosswalk is incomplete")
    excluded = set(mapping.values())
    selected = [
        {"thread_sha256": thread, "source_sha256": value[1], "info": value[2]}
        for thread, value in sorted(
            retained.items(), key=lambda item: (item[1][3], item[0])
        )
        if thread not in excluded
    ]
    return {
        "source_stat": before,
        "fingerprint": actual,
        "excluded_threads": len(excluded),
        "excluded_thread_hashes": sorted(excluded),
        "matched_sources": len(mapping),
        "members": selected,
    }

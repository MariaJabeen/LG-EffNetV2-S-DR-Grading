#!/usr/bin/env python3
"""Download ONLY the CFP subset of MMRDR from figshare and preprocess it.

MMRDR (figshare article 29423747, v2) is published as one ~18.6 GB zip archive
byte-split into nine parts (MMRDR.zip.001 ... .009). The zip central directory
lives at the very end (part 009), but the CFP images are stored first, inside
parts 001-004 (~7.5 GB). Instead of downloading all 18.6 GB, this script:

  1. reads the zip central directory with HTTP range requests (~3 MB),
  2. streams only the byte range that holds MMRDR-CFP/* entries,
  3. inflates each entry, verifies its CRC-32 against the central directory,
  4. optionally saves the original JPEG, and
  5. saves a preprocessed copy: fundus field-of-view cropped, padded to a
     square and resized to --size px (default 512, as in the MMRDR benchmark).

It is idempotent and resumable: already-processed images are skipped, and a
dropped connection is re-opened at the current byte offset.

Usage:
    python scripts/prepare_data.py --out data --size 512 --keep-original
Outputs:
    <out>/MMRDR-CFP/FP.csv                 official labels (unchanged)
    <out>/MMRDR-CFP/img/*.jpg              originals (only with --keep-original)
    <out>/cfp<size>/img/*.jpg              preprocessed images
    <out>/cfp<size>/prep_meta.csv          per-image crop / size metadata
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

ARTICLE_API = "https://api.figshare.com/v2/articles/29423747/versions/2"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"
SUBSET_PREFIX = "MMRDR-CFP/"
USER_AGENT = "mmrdr-cfp-prep/1.0"


# --------------------------------------------------------------------------- #
# Remote multi-part file access
# --------------------------------------------------------------------------- #
@dataclass
class Part:
    name: str
    file_id: int
    size: int
    start: int  # global byte offset of this part inside the concatenated zip


def list_parts() -> list[Part]:
    """Return the archive parts in order, with their global offsets."""
    req = urllib.request.Request(ARTICLE_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        files = json.load(r)["files"]
    parts = sorted((f for f in files if ".zip." in f["name"]), key=lambda f: f["name"])
    out, offset = [], 0
    for f in parts:
        out.append(Part(f["name"], f["id"], f["size"], offset))
        offset += f["size"]
    return out


def _open_range(part: Part, local_start: int, local_end: int | None = None):
    """Open an HTTP stream for bytes [local_start, local_end] of one part."""
    rng = f"bytes={local_start}-" + ("" if local_end is None else str(local_end))
    req = urllib.request.Request(
        DOWNLOAD_URL.format(file_id=part.file_id),
        headers={"Range": rng, "User-Agent": USER_AGENT},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    if resp.status != 206:  # a 200 would silently return the whole file
        raise RuntimeError(f"Server ignored Range header for {part.name} (HTTP {resp.status})")
    return resp


def read_range(parts: list[Part], start: int, length: int) -> bytes:
    """Read `length` bytes at global offset `start` (may span several parts)."""
    buf = bytearray()
    pos = start
    while len(buf) < length:
        part = next(p for p in parts if p.start <= pos < p.start + p.size)
        local = pos - part.start
        n = min(length - len(buf), part.size - local)
        with _open_range(part, local, local + n - 1) as r:
            chunk = r.read()
        if len(chunk) != n:
            raise IOError(f"Short read from {part.name}: {len(chunk)} != {n}")
        buf += chunk
        pos += n
    return bytes(buf)


class SequentialReader:
    """Forward-only reader over the concatenated parts with automatic reconnect."""

    def __init__(self, parts: list[Part], start: int, max_retries: int = 20):
        self.parts, self.pos, self.max_retries = parts, start, max_retries
        self._resp = None
        self._resp_part_end = 0
        self.bytes_read = 0

    def _connect(self):
        part = next(p for p in self.parts if p.start <= self.pos < p.start + p.size)
        self._resp = _open_range(part, self.pos - part.start)
        self._resp_part_end = part.start + part.size

    def _close(self):
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
        self._resp = None

    def read(self, n: int) -> bytes:
        out = bytearray()
        retries = 0
        while len(out) < n:
            try:
                if self._resp is None or self.pos >= self._resp_part_end:
                    self._close()
                    self._connect()
                want = min(n - len(out), self._resp_part_end - self.pos, 1 << 20)
                chunk = self._resp.read(want)
                if not chunk:
                    raise IOError("connection closed early")
                out += chunk
                self.pos += len(chunk)
                self.bytes_read += len(chunk)
                retries = 0
            except Exception as exc:  # network hiccup -> reconnect at self.pos
                retries += 1
                self._close()
                if retries > self.max_retries:
                    raise
                wait = min(60, 2 ** retries)
                print(f"\n[warn] {exc!r}; reconnecting at byte {self.pos} in {wait}s", file=sys.stderr)
                time.sleep(wait)
        return bytes(out)

    def skip_to(self, target: int):
        """Advance to `target`; reconnect instead of reading large gaps."""
        gap = target - self.pos
        if gap < 0:
            raise ValueError("SequentialReader cannot seek backwards")
        if gap > 8 << 20:
            self._close()
            self.pos = target
        elif gap:
            self.read(gap)

    def close(self):
        self._close()


# --------------------------------------------------------------------------- #
# Zip central-directory parsing (ZIP64 aware)
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    name: str
    method: int
    crc: int
    comp_size: int
    size: int
    header_offset: int


def read_central_directory(parts: list[Part]) -> list[Entry]:
    total = parts[-1].start + parts[-1].size
    tail_len = 1 << 16
    tail = read_range(parts, total - tail_len, tail_len)
    i64 = tail.rfind(b"PK\x06\x06")
    if i64 >= 0:  # ZIP64 end of central directory record
        _, _, _, _, _, _, _, n_entries, cd_size, cd_offset = struct.unpack("<IQHHIIQQQQ", tail[i64:i64 + 56])
    else:
        i = tail.rfind(b"PK\x05\x06")
        _, _, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    cd = read_range(parts, cd_offset, cd_size)

    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, crc, csize, usize, nlen, elen, clen,
         _, _, _, hoff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):  # ZIP64 extended information extra field
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, k = extra[q + 4:q + 4 + hsz], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if hoff == 0xFFFFFFFF:
                    hoff = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
            q += 4 + hsz
        entries.append(Entry(name, method, crc, csize, usize, hoff))
        p += 46 + nlen + elen + clen
    if len(entries) != n_entries:
        raise RuntimeError(f"Central directory parse mismatch: {len(entries)} vs {n_entries}")
    return entries


# --------------------------------------------------------------------------- #
# Image preprocessing (also used, verbatim, by the notebook)
# --------------------------------------------------------------------------- #
def fundus_bbox(rgb: np.ndarray, thresh: int = 15, min_frac: float = 0.01):
    """Bounding box (y0, y1, x0, x1) of the circular fundus field of view.

    A pixel belongs to the FOV if its maximum channel exceeds `thresh`; rows /
    columns are kept if more than `min_frac` of their pixels are FOV, which
    ignores thin text overlays and sensor noise in the black border.
    Falls back to the full image if nothing is detected.
    """
    mask = rgb.max(axis=2) > thresh
    rows = np.where(mask.mean(axis=1) > min_frac)[0]
    cols = np.where(mask.mean(axis=0) > min_frac)[0]
    if rows.size < 10 or cols.size < 10:
        return 0, rgb.shape[0], 0, rgb.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def preprocess_fundus(img: Image.Image, size: int = 512) -> tuple[Image.Image, dict]:
    """Crop to the FOV, pad to a centred square (black) and resize to `size`."""
    img = img.convert("RGB")
    rgb = np.asarray(img)
    y0, y1, x0, x1 = fundus_bbox(rgb)
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    out = Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
    meta = dict(orig_w=rgb.shape[1], orig_h=rgb.shape[0], crop_y0=y0, crop_y1=y1,
                crop_x0=x0, crop_x1=x1, fov_h=h, fov_w=w)
    return out, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output root directory")
    ap.add_argument("--size", type=int, default=512, help="preprocessed image side length")
    ap.add_argument("--keep-original", action="store_true", help="also save original JPEGs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (debug)")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "MMRDR-CFP")
    raw_img = os.path.join(raw_dir, "img")
    prep_dir = os.path.join(args.out, f"cfp{args.size}")
    prep_img = os.path.join(prep_dir, "img")
    for d in (raw_dir, prep_img) + ((raw_img,) if args.keep_original else ()):
        os.makedirs(d, exist_ok=True)

    parts = list_parts()
    entries = read_central_directory(parts)
    cfp = sorted((e for e in entries if e.name.startswith(SUBSET_PREFIX) and not e.name.endswith("/")),
                 key=lambda e: e.header_offset)
    images = [e for e in cfp if e.name.lower().endswith(".jpg")]
    others = [e for e in cfp if not e.name.lower().endswith(".jpg")]
    if args.limit:
        images = images[:args.limit]
    print(f"Archive: {len(parts)} parts, {len(entries)} entries; CFP: {len(images)} images + {[e.name for e in others]}")

    def done(e: Entry) -> bool:
        base = os.path.basename(e.name)
        ok = os.path.exists(os.path.join(prep_img, base))
        if args.keep_original:
            ok = ok and os.path.exists(os.path.join(raw_img, base))
        return ok

    todo = [e for e in others] + [e for e in images if not done(e)]
    todo.sort(key=lambda e: e.header_offset)
    print(f"{len(images) - sum(1 for e in todo if e in images)} images already done; {len(todo)} entries to fetch")

    meta_path = os.path.join(prep_dir, "prep_meta.csv")
    meta_lock = threading.Lock()
    meta_new = not os.path.exists(meta_path)
    meta_f = open(meta_path, "a", newline="")
    meta_w = csv.DictWriter(meta_f, fieldnames=["image", "orig_w", "orig_h", "crop_y0", "crop_y1",
                                                "crop_x0", "crop_x1", "fov_h", "fov_w"])
    if meta_new:
        meta_w.writeheader()

    def process(e: Entry, data: bytes):
        base = os.path.basename(e.name)
        if args.keep_original:
            with open(os.path.join(raw_img, base), "wb") as f:
                f.write(data)
        img, meta = preprocess_fundus(Image.open(io.BytesIO(data)), args.size)
        tmp = os.path.join(prep_img, base + ".tmp")
        img.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, os.path.join(prep_img, base))  # atomic: no half-written files
        with meta_lock:
            meta_w.writerow({"image": f"img/{base}", **meta})
            meta_f.flush()

    t0 = time.time()
    reader = SequentialReader(parts, todo[0].header_offset) if todo else None
    pool = ThreadPoolExecutor(max_workers=args.workers)
    inflight, n_done = [], 0
    try:
        for e in todo:
            reader.skip_to(e.header_offset)
            hdr = reader.read(30)
            if hdr[:4] != b"PK\x03\x04":
                raise RuntimeError(f"Bad local header for {e.name}")
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            reader.read(nlen + elen)
            comp = reader.read(e.comp_size)
            data = zlib.decompress(comp, -15) if e.method == 8 else comp
            if e.method not in (0, 8):
                raise RuntimeError(f"Unsupported compression {e.method} for {e.name}")
            if zlib.crc32(data) & 0xFFFFFFFF != e.crc or len(data) != e.size:
                raise RuntimeError(f"CRC/size mismatch for {e.name}")
            if not e.name.lower().endswith(".jpg"):  # label file(s): save as is
                with open(os.path.join(raw_dir, os.path.basename(e.name)), "wb") as f:
                    f.write(data)
                continue
            inflight.append(pool.submit(process, e, data))
            if len(inflight) >= 4 * args.workers:  # bound memory use
                inflight.pop(0).result()
            n_done += 1
            if n_done % 100 == 0 or n_done == len(todo) - len(others):
                mb = reader.bytes_read / 1e6
                el = time.time() - t0
                print(f"\r{n_done} images | {mb:,.0f} MB | {mb / max(el, 1e-6):.1f} MB/s | {el / 60:.1f} min",
                      end="", flush=True)
        for fut in inflight:
            fut.result()
    finally:
        pool.shutdown(wait=True)
        meta_f.close()
        if reader:
            reader.close()

    # Labels are needed next to the preprocessed images too.
    fp = os.path.join(raw_dir, "FP.csv")
    if os.path.exists(fp):
        with open(fp, "rb") as src, open(os.path.join(prep_dir, "FP.csv"), "wb") as dst:
            dst.write(src.read())
    n_prep = len([f for f in os.listdir(prep_img) if f.endswith(".jpg")])
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Preprocessed images on disk: {n_prep}")


if __name__ == "__main__":
    main()

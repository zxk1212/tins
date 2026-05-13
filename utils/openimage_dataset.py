import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch


DEFAULT_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _iter_image_paths(root: str, exts: Sequence[str]) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in exts):
                yield os.path.join(dirpath, name)


def _read_manifest_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _write_manifest(path: str, paths: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for p in paths:
            handle.write(p)
            handle.write("\n")


@dataclass(frozen=True)
class OpenImagePaths:
    root: str
    paths: Tuple[str, ...]


def build_openimage_paths(
    root: str,
    *,
    manifest_path: Optional[str] = None,
    exts: Sequence[str] = DEFAULT_IMG_EXTS,
    max_images: Optional[int] = None,
) -> OpenImagePaths:
    root_path = str(Path(root).resolve())
    if manifest_path is not None and os.path.isfile(manifest_path):
        paths = _read_manifest_lines(manifest_path)
        paths = [p if os.path.isabs(p) else os.path.join(root_path, p) for p in paths]
    else:
        paths = list(_iter_image_paths(root_path, exts=exts))
        paths.sort()
        if manifest_path is not None:
            rel = [os.path.relpath(p, root_path) for p in paths]
            _write_manifest(manifest_path, rel)

    if max_images is not None and max_images > 0:
        paths = paths[: max_images]

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found under '{root_path}'. "
            f"Tried extensions: {list(exts)}. If data exists, check permissions/mount."
        )

    return OpenImagePaths(root=root_path, paths=tuple(paths))


class OpenImageOutlierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable] = None,
        manifest_path: Optional[str] = None,
        exts: Sequence[str] = DEFAULT_IMG_EXTS,
        max_images: Optional[int] = None,
    ) -> None:
        self.transform = transform
        built = build_openimage_paths(
            root=root,
            manifest_path=manifest_path,
            exts=exts,
            max_images=max_images,
        )
        self.root = built.root
        self.paths = list(built.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


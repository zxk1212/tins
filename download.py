import os
import sys
import time
from pathlib import Path

import gdown

FILES = {
    "places": "1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3",
    "sun": "1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
}

OUT_DIR = Path("datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def human_size(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(n)
    for u in units:
        if n < 1024 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024


def main():
    results = []

    for name, file_id in FILES.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        out_path = OUT_DIR / name

        print(f"[{name}] 状态: 开始下载")
        start = time.time()

        try:
            final_path = gdown.download(
                url=url,
                output=str(out_path),
                quiet=False,
                fuzzy=True,
            )

            if final_path is None or not os.path.exists(final_path):
                raise RuntimeError("gdown 未返回有效文件路径")

            size = os.path.getsize(final_path)
            elapsed = max(time.time() - start, 1e-6)

            print(
                f"[{name}] 状态: 下载完成 -> {final_path} "
                f"({human_size(size)}, {human_size(size / elapsed)}/s)"
            )

            results.append((name, "SUCCESS", final_path, human_size(size), ""))
        except Exception as e:
            print(f"[{name}] 状态: 下载失败")
            print(f"[{name}] 错误: {e}")
            results.append((name, "FAILED", "-", "-", str(e)))

    print("\n===== 下载汇总 =====")
    for name, status, path, size, err in results:
        print(f"{name:12s} | {status:7s} | {size:>10s} | {path}")
        if err:
            print(f"  error: {err}")


if __name__ == "__main__":
    main()
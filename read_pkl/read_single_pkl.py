import pickle
import sys
import numpy as np
import torch


def describe(val, indent=0):
    pad = "  " * indent
    if isinstance(val, dict):
        lines = [f"{pad}dict({len(val)} keys)"]
        for k, v in val.items():
            lines.append(f"{pad}  [{k!r}]:")
            lines.extend(describe(v, indent + 2).splitlines())
        return "\n".join(lines)
    elif isinstance(val, (list, tuple)):
        t = type(val).__name__
        if len(val) == 0:
            return f"{pad}{t}(empty)"
        lines = [f"{pad}{t}(len={len(val)})"]
        for i, v in enumerate(val[:3]):
            lines.append(f"{pad}  [{i}]:")
            lines.extend(describe(v, indent + 2).splitlines())
        if len(val) > 3:
            lines.append(f"{pad}  ... ({len(val) - 3} more)")
        return "\n".join(lines)
    elif isinstance(val, np.ndarray):
        return f"{pad}ndarray shape={val.shape} dtype={val.dtype} min={val.min():.4g} max={val.max():.4g}"
    elif isinstance(val, torch.Tensor):
        return f"{pad}Tensor shape={tuple(val.shape)} dtype={val.dtype} min={val.min():.4g} max={val.max():.4g}"
    else:
        return f"{pad}{type(val).__name__}: {val!r}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_single_pkl.py <path_to_pkl> [output.txt]")
        sys.exit(1)

    pkl_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else pkl_path + "_contents.txt"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    output = f"File: {pkl_path}\n{'=' * 60}\n" + describe(data) + "\n"

    with open(out_path, "w") as f:
        f.write(output)

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

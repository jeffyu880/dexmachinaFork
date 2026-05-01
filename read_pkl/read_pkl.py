import pickle
import pprint
import sys
from pathlib import Path

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_pkl_files(pkl1, pkl2, output_file='diff.txt'):
    """Compare two pickle files and write differences to output file."""
    
    # Load both files
    with open(pkl1, 'rb') as f:
        data1 = pickle.load(f)
    
    with open(pkl2, 'rb') as f:
        data2 = pickle.load(f)
    
    # Flatten for easier comparison
    flat1 = flatten_dict(data1) if isinstance(data1, dict) else data1
    flat2 = flatten_dict(data2) if isinstance(data2, dict) else data2
    
    differences = []
    differences.append(f"Comparing: {pkl1} vs {pkl2}\n")
    differences.append("=" * 80 + "\n\n")
    
    # Find keys only in first file
    if isinstance(flat1, dict) and isinstance(flat2, dict):
        keys_only_in_1 = set(flat1.keys()) - set(flat2.keys())
        if keys_only_in_1:
            differences.append("Keys only in first file:\n")
            for key in sorted(keys_only_in_1):
                differences.append(f"  - {key}: {flat1[key]}\n")
            differences.append("\n")
        
        # Find keys only in second file
        keys_only_in_2 = set(flat2.keys()) - set(flat1.keys())
        if keys_only_in_2:
            differences.append("Keys only in second file:\n")
            for key in sorted(keys_only_in_2):
                differences.append(f"  + {key}: {flat2[key]}\n")
            differences.append("\n")
        
        # Find different values for common keys
        different_values = []
        for key in sorted(set(flat1.keys()) & set(flat2.keys())):
            if flat1[key] != flat2[key]:
                different_values.append((key, flat1[key], flat2[key]))
        
        if different_values:
            differences.append("Different values:\n")
            for key, val1, val2 in different_values:
                differences.append(f"  {key}:\n")
                differences.append(f"    File 1: {val1}\n")
                differences.append(f"    File 2: {val2}\n")
            differences.append("\n")
        
        if not keys_only_in_1 and not keys_only_in_2 and not different_values:
            differences.append("No differences found - files are identical!\n")
    else:
        differences.append("File structures are not dictionaries, cannot compare in detail.\n")
        differences.append(f"File 1 type: {type(flat1)}\n")
        differences.append(f"File 2 type: {type(flat2)}\n")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.writelines(differences)
    
    print(f"Differences written to {output_file}")
    return differences

def read_pkl_file(pkl_path, output_file=None):
    """Read a single pickle file and write its contents to a text file."""
    if output_file is None:
        output_file = str(pkl_path) + "_contents.txt"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    lines = [f"File: {pkl_path}\n", "=" * 80 + "\n\n"]

    def describe(val, indent=0):
        pad = "  " * indent
        import numpy as np
        try:
            import torch
            is_tensor = isinstance(val, torch.Tensor)
        except ImportError:
            is_tensor = False

        if isinstance(val, dict):
            out = [f"{pad}dict({len(val)} keys)\n"]
            for k, v in val.items():
                out.append(f"{pad}  [{k!r}]:\n")
                out.extend(describe(v, indent + 2))
            return out
        elif isinstance(val, (list, tuple)):
            t = type(val).__name__
            if len(val) == 0:
                return [f"{pad}{t}(empty)\n"]
            out = [f"{pad}{t}(len={len(val)})\n"]
            for i, v in enumerate(val[:3]):
                out.append(f"{pad}  [{i}]:\n")
                out.extend(describe(v, indent + 2))
            if len(val) > 3:
                out.append(f"{pad}  ... ({len(val) - 3} more)\n")
            return out
        elif isinstance(val, np.ndarray):
            return [f"{pad}ndarray shape={val.shape} dtype={val.dtype} min={val.min():.4g} max={val.max():.4g}\n"]
        elif is_tensor:
            return [f"{pad}Tensor shape={tuple(val.shape)} dtype={val.dtype} min={val.min():.4g} max={val.max():.4g}\n"]
        else:
            return [f"{pad}{type(val).__name__}: {val!r}\n"]

    lines.extend(describe(data))

    with open(output_file, "w") as f:
        f.writelines(lines)

    print(f"Written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pkl.py <pkl_file> [output_file]")
        print("       python read_pkl.py <pkl_file_1> <pkl_file_2> [output_file]")
        sys.exit(1)

    # Single file mode
    if len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2].endswith(".txt")):
        pkl = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) == 3 else None
        if not Path(pkl).exists():
            print(f"Error: {pkl} not found")
            sys.exit(1)
        read_pkl_file(pkl, output)
        sys.exit(0)

    # Diff mode
    pkl1 = sys.argv[1]
    pkl2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else 'diff.txt'

    if not Path(pkl1).exists():
        print(f"Error: {pkl1} not found")
        sys.exit(1)
    if not Path(pkl2).exists():
        print(f"Error: {pkl2} not found")
        sys.exit(1)

    differences = compare_pkl_files(pkl1, pkl2, output)
    print('\n'.join(differences))
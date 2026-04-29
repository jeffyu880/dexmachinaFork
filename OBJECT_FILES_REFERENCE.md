# Object Definition Reference: Files and Paths

## Asset Location Map

### Base Directory
```
dexmachina/assets/arctic/
```

### Object Directories

```
arctic/
├── box/                      ← Cardboard box (2-part articulated)
├── ketchup/                  ← Ketchup bottle (single or 2-part)
├── laptop/                   ← Laptop (2-part: body + screen)
├── mixer/                    ← Electric mixer (2-part: body + beaters)
├── notebook/                 ← Spiral notebook (2-part: cover + pages)
├── waffleiron/              ← Waffle iron (2-part: bottom + top)
└── processed/               ← Intermediate processing files
```

---

## File Structure by Object

### Example: Box Object

```
dexmachina/assets/arctic/box/
│
├── COLLISION MODELS (for physics simulation)
│   ├── box.urdf                          ← Standard URDF with all collision geometry
│   └── decomp/
│       └── box_decomp.urdf               ← Decomposed into convex pieces
│
├── MESH FILES FOR ADD METRIC ⭐ (USE THESE)
│   ├── bottom_watertight_tiny.obj        ← ASCII format, ~750 vertices
│   ├── bottom_watertight_tiny.stl        ← Binary format, same geometry
│   ├── top_watertight_tiny.obj           ← ASCII format, ~800 vertices
│   └── top_watertight_tiny.stl           ← Binary format, same geometry
│
├── VISUALIZATION MESHES (for rendering)
│   ├── bottom_textured.obj               ← High quality visualization
│   ├── top_textured.obj                  ← High quality visualization
│   ├── bottom_texture.jpg                ← Texture image for bottom
│   ├── top_texture.jpg                   ← Texture image for top
│   ├── bottom_watertight_tiny_box.obj    ← Alternative naming
│   └── top_watertight_tiny_box.obj       ← Alternative naming
│
└── MATERIAL FILES
    ├── bottom_material.mtl               ← Material definition for OBJ
    └── top_material.mtl                  ← Material definition for OBJ
```

---

## File Naming Convention

### Understanding the Names

**`{part}_{quality}_{object}.{format}`**

#### Part:
- `bottom` - Lower/fixed part of articulated object
- `top` - Upper/movable part of articulated object

#### Quality/Type:
- `watertight_tiny` - **BEST FOR ADD METRIC**
  - Closed surface (no holes)
  - Simplified/downsampled vertices
  - Maintains geometric accuracy
  - ~500-1500 vertices per part
  
- `textured` - **FOR VISUALIZATION ONLY**
  - High quality mesh
  - Includes texture coordinates
  - Too complex for metric calculation
  
- `(no suffix)` - Original URDF collision geometry
  - For physics simulation
  - Not typically used directly for ADD

#### Format:
- `.obj` - ASCII text format
  - Human readable
  - Larger file size
  - Can include texture coordinates
  - Good for debugging

- `.stl` - Binary format
  - Compact file size
  - Faster to load
  - No texture data
  - Pure geometry only
  
- `.urdf` - URDF XML format
  - Describes rigid body structure
  - Includes collision geometry
  - For physics engine

---

## How to Access Files in Code

### Method 1: Via `get_arctic_object_cfg()`

```python
from dexmachina.envs.object import get_arctic_object_cfg

obj_cfg = get_arctic_object_cfg(name="box")

# Access mesh file paths:
print(obj_cfg['bottom_mesh_fname'])  # Path to bottom watertight mesh
print(obj_cfg['top_mesh_fname'])     # Path to top watertight mesh
print(obj_cfg['urdf_path'])          # Path to collision URDF

# Example output:
# /home/.../assets/arctic/box/bottom_watertight_tiny.obj
# /home/.../assets/arctic/box/top_watertight_tiny.obj
# /home/.../assets/arctic/box/box.urdf
```

### Method 2: Direct Path Construction

```python
from pathlib import Path
from dexmachina.asset_utils import get_asset_path

asset_path = get_asset_path("arctic")
obj_name = "box"

bottom_mesh = Path(asset_path) / obj_name / "bottom_watertight_tiny.obj"
top_mesh = Path(asset_path) / obj_name / "top_watertight_tiny.obj"
```

### Method 3: Relative to Installation

```python
import os
from pathlib import Path

# Find dexmachina package location
dexmachina_root = Path(__file__).parent.parent  # From any dexmachina file
assets_dir = dexmachina_root / "assets" / "arctic"

mesh_path = assets_dir / "box" / "bottom_watertight_tiny.obj"
```

---

## File Format Details

### OBJ Format Example

```
# box/bottom_watertight_tiny.obj
# Vertices (v x y z)
v -0.05 -0.02 0.0
v -0.05 -0.02 0.1
v 0.05 -0.02 0.0
v 0.05 -0.02 0.1
...
# Faces (f v1 v2 v3)
f 1 2 3
f 2 4 3
...
```

**Reading with trimesh:**
```python
import trimesh
mesh = trimesh.load("bottom_watertight_tiny.obj")
vertices = mesh.vertices  # (N, 3) numpy array
```

### STL Format

- Binary format (not human readable)
- Contains vertex and face data
- No texture coordinates
- Faster to parse

**Reading with trimesh:**
```python
import trimesh
mesh = trimesh.load("bottom_watertight_tiny.stl")
vertices = mesh.vertices  # (N, 3) numpy array
```

### URDF Format Example

```xml
<!-- box.urdf -->
<robot name="box">
  <link name="bottom">
    <collision>
      <geometry>
        <mesh filename="path/to/mesh.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
  <link name="top">
    <collision>
      <geometry>
        <mesh filename="path/to/mesh.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="bottom"/>
    <child link="top"/>
    <limit lower="0" upper="3.14159"/>
  </joint>
</robot>
```

---

## Vertex Count Reference

### Typical Vertex Counts (watertight_tiny meshes)

| Object | Bottom Vertices | Top Vertices | Total | Notes |
|--------|-----------------|--------------|-------|-------|
| box | 754 | 812 | 1566 | Simple box shape |
| ketchup | 428 | - | 428 | Single piece (no joint) |
| laptop | 621 | 834 | 1455 | Screen folds open |
| mixer | 512 | 289 | 801 | Rotating beaters |
| notebook | 756 | 298 | 1054 | Spiraling pages |
| waffleiron | 682 | 756 | 1438 | Clamshell design |

**Implications:**
- More vertices = more accurate ADD metric
- But also more computation
- ~1000 vertices is a good balance
- Computation: O(T × N) where T=frames, N=vertices

---

## Configuration: How Objects Are Defined

### Default Configuration for Each Object

**box** (cardboard box, ~30cm):
```python
{
    "base_init_pos": [0.0597, -0.2476, 1.0354],
    "base_init_quat": [-0.6413, 0.2875, 0.6467, -0.2964],
}
```

**ketchup** (bottle):
```python
{
    "base_init_pos": [0.0, 0.0, 0.3],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
}
```

**laptop** (15-inch):
```python
{
    "base_init_pos": [0.0, 0.0, 0.3],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
}
```

All values in:
- Position: meters (x, y, z)
- Quaternion: normalized (w, x, y, z)

---

## Object State Representation

### How Objects Are Tracked During Episodes

Each frame stores **8-element state**:

```python
state[t] = [
    obj_pos[0], obj_pos[1], obj_pos[2],        # Position (m)
    obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3],  # Quaternion (w, x, y, z)
    obj_arti[0]                                # Joint angle (rad)
]
```

**Stored as:**
- **Demo states**: Ground truth from ARCTIC dataset
  - Shape: (T, 8) where T = frames in clip
  - Loaded from demonstration recording
  
- **Agent states**: RL agent's predictions
  - Shape: (T, 8) same T as demo
  - Output of trained policy network

**Processing:**
```python
# Extract data from state vector for ADD metric
position = state[:, :3]       # (T, 3)
quaternion = state[:, 3:7]    # (T, 4) in wxyz format
articulation = state[:, 7:8]  # (T, 1)

# For ADD metric, use position + quaternion (ignore articulation)
# The articulation angle doesn't affect rigid mesh transformation
```

---

## Troubleshooting: Finding the Right Files

### Problem: "bottom_watertight_tiny.obj not found"

**Check:**
1. Object name is lowercase and exact: `box`, not `Box` or `BOX`
2. Asset files are installed in package
3. Using correct path separator for OS (auto-handled by pathlib)

**Fix:**
```python
from dexmachina.envs.object import get_arctic_object_cfg
obj_cfg = get_arctic_object_cfg(name="box")
# This will fail with exact error if files don't exist
```

### Problem: Different vertex counts than expected

**Possible causes:**
1. Loading different file (textured vs. watertight_tiny)
2. OBJ file includes duplicate vertices
3. Loading both parts instead of one

**Fix:**
```python
import trimesh
mesh = trimesh.load("bottom_watertight_tiny.obj")
print(f"Vertices: {mesh.vertices.shape[0]}")
print(f"Faces: {mesh.faces.shape[0]}")
print(f"File: bottom_watertight_tiny.obj")  # Verify filename
```

### Problem: Slow mesh loading

**Expected:**
- First load of trimesh: 1-2 seconds per mesh
- Subsequent loads from cache: instant

**Solution:**
```python
# Cache loaded meshes across multiple evaluations
mesh_cache = {}

def load_once(obj_name):
    if obj_name not in mesh_cache:
        # load and cache
        mesh_cache[obj_name] = {...}
    return mesh_cache[obj_name]
```

---

## Integration: Using in ADD Metric Code

### Complete File Path Usage

```python
import trimesh
from dexmachina.envs.object import get_arctic_object_cfg

# Get configuration
obj_cfg = get_arctic_object_cfg(name="box")

# Load meshes using paths from config
bottom = trimesh.load(obj_cfg['bottom_mesh_fname'])
top = trimesh.load(obj_cfg['top_mesh_fname'])

# Combine vertices
vertices = np.vstack([
    bottom.vertices,   # Shape: (N_bottom, 3)
    top.vertices       # Shape: (N_top, 3)
])  # Result: (N_total, 3)

# Pass to ADD metric
object_models = {'vertices': vertices}
metrics = compute_auc_add3(
    agent_states, demo_states,
    object_models=object_models,
    max_distance=0.1
)
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Base directory** | `dexmachina/assets/arctic/` |
| **Available objects** | box, ketchup, laptop, mixer, notebook, waffleiron |
| **Mesh files for ADD** | `{part}_watertight_tiny.{obj\|stl}` |
| **Access in code** | `obj_cfg['bottom_mesh_fname']`, `obj_cfg['top_mesh_fname']` |
| **Load library** | `trimesh.load(path)` |
| **Vertex array shape** | `(N, 3)` where N is 500-1500 |
| **State format** | `(T, 8)` = [pos(3), quat(4), arti(1)] |
| **Metric calculation** | Compare transformed vertices at each timestep |
| **Thresholds** | 3cm, 5cm, 10cm (standard for 6D pose) |
| **Output** | AUC-ADD3 score 0.0-1.0 (higher is better) |

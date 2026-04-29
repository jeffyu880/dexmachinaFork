# Object Loading and Mesh Definition Guide for ADD Metrics

## Overview

Objects in dexmachina are ARCTIC dataset objects (Box, Ketchup, Laptop, Mixer, Notebook, Waffleiron) that are loaded through a configuration system. This guide explains how objects are loaded into demonstrations and how to extract mesh vertices for ADD (Average Distance of Model Points) metric calculations.

---

## 1. Object Configuration: `get_arctic_object_cfg()`

**Location:** `dexmachina/envs/object.py` (lines 13-62)

### Basic Configuration

Objects are defined via `get_arctic_object_cfg()` function:

```python
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg

# Get configuration for an object
obj_cfg = get_arctic_object_cfg(
    name="box",           # Object name: box, ketchup, laptop, mixer, notebook, waffleiron
    convexify=True,       # Convexify collision geometry
    decomp=True,          # Use decomposed URDF
    texture_mesh=False    # Load texture meshes for visualization
)
```

### Configuration Structure

The returned `obj_cfg` dictionary contains:

```python
{
    "name": str,                           # Object name
    "base_init_pos": [x, y, z],           # Initial position
    "base_init_quat": [w, x, y, z],       # Initial quaternion
    "base_init_qpos": [0.0],              # Joint angle at rest
    "num_sample_vertics": 300,            # Number of sample vertices for collision
    "convexify": bool,                     # Convexify collision meshes
    "fixed": bool,                         # Fixed or free body
    "actuated": bool,                      # Whether to control object joint
    "kp": float,                           # Joint proportional gain
    "kv": float,                           # Joint derivative gain
    "force_range": float,                  # Joint force limit
    "collect_data": bool,                  # Collect data during episode
    "offset_pos": [x, y, z],              # Visualization offset
    "color": None,                         # Object color
    "show_link_frame": bool,               # Show link frames
    
    # File paths (automatically set):
    "urdf_path": str,                      # Path to URDF file
    "bottom_mesh_fname": str,              # Path to bottom watertight mesh
    "top_mesh_fname": str,                 # Path to top watertight mesh
    
    # Optional texture meshes:
    "texture_meshes": {
        "top": {"fname": str, "tex": str},
        "bottom": {"fname": str, "tex": str}
    }
}
```

---

## 2. Object Storage Location

**Base Path:** `dexmachina/assets/arctic/{object_name}/`

### Available Objects

- **box/** - Cardboard box
- **ketchup/** - Ketchup bottle
- **laptop/** - Laptop
- **mixer/** - Electric mixer
- **notebook/** - Notebook/notepad
- **waffleiron/** - Waffle iron
- **processed/** - Processed objects (intermediate format)

### Mesh Files per Object

Each object folder contains:

```
{object_name}/
├── {object_name}.urdf                    # Main URDF (collision model)
├── decomp/{object_name}_decomp.urdf      # Decomposed URDF (convex pieces)
│
├── bottom_watertight_tiny.stl            # Bottom part - STL format (binary)
├── bottom_watertight_tiny.obj            # Bottom part - OBJ format (ASCII)
├── bottom_textured.obj                   # Bottom part - textured visualization
├── bottom_texture.jpg                    # Bottom texture image
│
├── top_watertight_tiny.stl               # Top part - STL format (binary)
├── top_watertight_tiny.obj               # Top part - OBJ format (ASCII)
├── top_textured.obj                      # Top part - textured visualization
├── top_texture.jpg                       # Top texture image
│
└── Material files (.mtl)                 # Material definitions
```

### Mesh File Types

1. **Collision URDF** (`*.urdf`)
   - Defines collision geometry for physics simulation
   - Primary model used in Genesis physics engine
   - Contains link definitions and joint information

2. **Watertight Meshes** (`*_watertight_tiny.*`)
   - High-quality closed surface meshes
   - **Best for ADD metric calculations** ✓
   - Available in both STL (binary) and OBJ (ASCII) formats
   - `tiny` suffix indicates simplified/downsampled version
   - Represents the actual object shape accurately

3. **Textured Visualization Meshes** (`*_textured.obj`)
   - For visual rendering with textures
   - Lower fidelity than watertight meshes
   - Not suitable for precise metric calculations

---

## 3. Loading Objects into Demonstrations

### The `ArticulatedObject` Class

**Location:** `dexmachina/envs/object.py` (lines 65-561)

Objects are instantiated as `ArticulatedObject` in RL training and evaluation:

```python
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg

obj_cfg = get_arctic_object_cfg(name="box", texture_mesh=True)
demo_data = get_demo_data(
    obj_name="box",
    frame_start=10,
    frame_end=510,
    hand_name='inspire_hand',
    load_retarget_contact=False,
)

obj = ArticulatedObject(
    obj_cfg,
    device=device,
    scene=scene,
    num_envs=num_envs,
    demo_data=demo_data  # ← Demonstration trajectory
)
```

### Object State Representation

Object state is stored as **8-element tensor** per frame:

```python
obj_state = [
    pos[0], pos[1], pos[2],     # Position (x, y, z)
    quat[0], quat[1], quat[2], quat[3],  # Quaternion (w, x, y, z)
    arti[0]                     # Articulation (joint angle)
]  # Total: 8D state vector
```

This representation is used in:
- `demo_states`: Ground truth states from ARCTIC dataset
- `obj_state`: Agent's predicted states during evaluation
- Stored in NumPy arrays of shape `(T, 8)` where T is number of frames

---

## 4. Extracting Mesh Vertices for ADD Metrics

### Step 1: Load Mesh Vertices from Files

Use `trimesh` library to load mesh vertices:

```python
import trimesh
import numpy as np
from pathlib import Path

def load_object_model(obj_name, asset_dir="dexmachina/assets/arctic"):
    """
    Load mesh vertices for an object.
    
    Returns:
        dict with 'vertices' key containing (N, 3) array of vertex positions
    """
    # Path to watertight mesh (best quality)
    mesh_path = Path(asset_dir) / obj_name / f"{obj_name}_watertight_tiny.obj"
    # Fallback to STL if OBJ not available
    if not mesh_path.exists():
        mesh_path = Path(asset_dir) / obj_name / f"{obj_name}_watertight_tiny.stl"
    
    if not mesh_path.exists():
        # Try bottom and top separately
        bottom_path = Path(asset_dir) / obj_name / "bottom_watertight_tiny.obj"
        top_path = Path(asset_dir) / obj_name / "top_watertight_tiny.obj"
        
        bottom_mesh = trimesh.load(str(bottom_path))
        top_mesh = trimesh.load(str(top_path))
        
        # Combine vertices from both parts
        vertices = np.vstack([bottom_mesh.vertices, top_mesh.vertices])
    else:
        # Single mesh file
        mesh = trimesh.load(str(mesh_path))
        vertices = mesh.vertices
    
    return {
        'vertices': vertices,  # Shape: (N, 3) where N is number of vertices
        'name': obj_name,
        'path': str(mesh_path)
    }
```

### Step 2: Update `compute_auc_add3()` Function

**Location:** `dexmachina/rl/eval_rl_games.py` (lines 42-96)

The function signature already supports object models:

```python
def compute_auc_add3(obj_states, demo_states, object_models=None, max_distance=0.1):
    """
    Args:
        obj_states: (T, 8) tensor [pos, quat, arti]
        demo_states: (T, 8) tensor [pos, quat, arti]
        object_models: dict with 'vertices' key (N, 3) or None
        max_distance: threshold for ADD accuracy (default 0.1m)
    """
```

Current implementation uses **origin point only**. To use actual vertices:

```python
# Inside compute_auc_add3(), around line 73-74:
if object_models is not None and 'vertices' in object_models:
    vertices = torch.tensor(object_models['vertices'], dtype=torch.float32, device=device)
    # vertices shape: (N, 3)
    # Then transform vertices using the computed rotation matrix
    # and compare ADD error using all vertices
```

---

## 5. Using Object Models in Evaluation

### Current Usage in `eval_rl_games.py`

Around line 407-410:

```python
add3_metrics = compute_auc_add3(
    eval_data['obj_state'],           # Agent's predicted states
    eval_data['demo_state'],          # Ground truth states
    object_models=None,               # ← Can load here!
    max_distance=0.1
)
```

### Enhanced Usage with Mesh Vertices

```python
from dexmachina.envs.object import get_arctic_object_cfg
from pathlib import Path

def load_object_model_vertices(obj_name):
    """Helper to load object vertices for ADD metric."""
    import trimesh
    
    obj_cfg = get_arctic_object_cfg(name=obj_name)
    asset_base = Path(obj_cfg['bottom_mesh_fname']).parent.parent.parent
    
    # Load both parts
    bottom_mesh = trimesh.load(obj_cfg['bottom_mesh_fname'])
    top_mesh = trimesh.load(obj_cfg['top_mesh_fname'])
    
    # Combine vertices
    vertices = np.vstack([bottom_mesh.vertices, top_mesh.vertices])
    
    return {'vertices': vertices}

# In evaluation:
# Extract object name from config
object_models = load_object_model_vertices(args.object_name)

add3_metrics = compute_auc_add3(
    eval_data['obj_state'],
    eval_data['demo_state'],
    object_models=object_models,  # ← Now using actual mesh!
    max_distance=0.1
)
```

---

## 6. Object State Definition Summary

### State Vector Format: `(T, 8)` tensors

```
Frame t: [x, y, z, w, qx, qy, qz, arti]
         └─── Position ─┘ └─ Quaternion ─┘ └─ Joint ─┘
              (3D)           (wxyz format)   angle
                             (w=scalar)      (1D)
```

### Where Objects Come From

1. **ARCTIC Dataset**: Recorded human-object interactions
   - Real capture using OptiTrack or similar
   - Contact-rich object manipulation tasks
   - Multiple subjects, objects, and viewpoints

2. **Demonstration Data**: Loaded via `get_demo_data()`
   - **Location:** `dexmachina/envs/demo_data.py`
   - Returns dict: `{'obj_pos', 'obj_quat', 'obj_arti', ...}`
   - Shapes: (T, 3), (T, 4), (T, 1) → concatenated to (T, 8)

3. **Evaluation Data**: Stored from evaluation episodes
   - Agent's predictions: `eval_data['obj_state']`
   - Demonstration replay: `eval_data['demo_state']`
   - Both shape (T, 8)

---

## 7. Key Mesh Characteristics

| Property | Value | Note |
|----------|-------|------|
| Watertight | Yes | Closed surface without holes |
| Format | OBJ (ASCII) or STL (Binary) | Use watertight_tiny versions |
| Vertices per object | 100-2000 | Varies by object complexity |
| Quality for ADD | Optimal | Accurately represents object geometry |
| Scaling | Real-world units | Typically 0.1-0.3m objects |
| Center | Not at origin | Mesh coordinates relative to object frame |

---

## 8. Example: Complete ADD Metric Workflow

```python
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

# 1. Load object configuration
from dexmachina.envs.object import get_arctic_object_cfg
obj_cfg = get_arctic_object_cfg(name="box")

# 2. Load mesh vertices
bottom = trimesh.load(obj_cfg['bottom_mesh_fname'])
top = trimesh.load(obj_cfg['top_mesh_fname'])
vertices = np.vstack([bottom.vertices, top.vertices])  # (N, 3)

# 3. Get evaluation states (both shape (T, 8))
agent_states = np.load('eval_ep0.npy')['obj_state']      # Agent predictions
demo_states = np.load('eval_ep0.npy')['demo_state']     # Ground truth

# 4. Compute ADD3 with vertices
from dexmachina.rl.eval_rl_games import compute_auc_add3
object_models = {'vertices': vertices}

metrics = compute_auc_add3(
    torch.from_numpy(agent_states),
    torch.from_numpy(demo_states),
    object_models=object_models,
    max_distance=0.1
)

print(f"AUC-ADD3 Score: {metrics['auc_add3']:.4f}")
print(f"ADD Mean Error: {metrics['add_mean']:.6f} m")
```

---

## 9. Technical Details: Object Hierarchy

### Genesis Scene Structure

```
Scene
├── Physics Entities
│   ├── Support Box (ground reference)
│   └── Object (ArticulatedObject)
│       ├── Base Link
│       │   └── Collision geometry (from URDF)
│       └── Movable Link
│           └── Joint (REVOLUTE or PRISMATIC)
│
└── Visualization Entities
    ├── Texture Mesh (top part)
    └── Texture Mesh (bottom part)
```

### Why Two Meshes?

- Most ARCTIC objects have **two articulated parts** (e.g., box lid, laptop screen)
- Bottom part: Fixed relative to base link
- Top part: Moves with the joint
- For ADD metric: **Combine vertices from both parts** for complete object representation

---

## 10. Files to Modify for Full ADD3 Integration

To fully integrate object vertices into ADD metric calculations:

### 1. `eval_rl_games.py` - Add mesh loading
```python
# Add near line 380 (before eval loop)
def load_evaluation_object_model(obj_name):
    """Load object mesh for ADD metric evaluation."""
    from dexmachina.envs.object import get_arctic_object_cfg
    import trimesh
    
    obj_cfg = get_arctic_object_cfg(name=obj_name)
    bottom = trimesh.load(obj_cfg['bottom_mesh_fname'])
    top = trimesh.load(obj_cfg['top_mesh_fname'])
    vertices = np.vstack([bottom.vertices, top.vertices])
    return {'vertices': vertices}

# Then in main():
object_models = load_evaluation_object_model(obj_name)
for eps in range(args.eval_episodes):
    # ...
    add3_metrics = compute_auc_add3(
        eval_data['obj_state'],
        eval_data['demo_state'],
        object_models=object_models,  # ← Pass loaded vertices
        max_distance=0.1
    )
```

### 2. `compute_auc_add3()` - Already supports object_models
- Function at lines 42-96 already accepts `object_models` parameter
- Just needs vertices passed from eval loop

### 3. `analyze_eval.py` - Already has plot_add3_metric()
- Added in recent update (lines ~340-400)
- Visualizes ADD and ADD3 metrics automatically

---

## Summary

**Objects are loaded through:**
1. Configuration → `get_arctic_object_cfg(name="box")`
2. Creation → `ArticulatedObject(obj_cfg, ...)`
3. State tracking → 8D vectors `(pos, quat, arti)`

**Meshes are stored as:**
- Watertight STL/OBJ files in `assets/arctic/{object_name}/`
- Bottom and top parts combined for articulated objects
- Best files for ADD metrics: `*watertight_tiny.obj` or `.stl`

**For ADD metrics you need:**
- Object vertices from mesh files → `(N, 3)` arrays
- Agent states → `(T, 8)` tensors
- Demo states → `(T, 8)` tensors
- Pass to `compute_auc_add3(agent_states, demo_states, object_models={'vertices': vertices})`

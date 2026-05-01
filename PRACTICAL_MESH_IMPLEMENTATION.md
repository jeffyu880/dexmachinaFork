# Practical Implementation: Using Object Meshes in ADD Metric Calculation

## Quick Start

To use object mesh vertices in your ADD metric calculations:

### 1. Install Required Package (if not already installed)

```bash
pip install trimesh
```

### 2. Minimal Code Changes

Add this helper function to `eval_rl_games.py` (around line 380, before `main()`):

```python
def load_object_model_for_evaluation(obj_name):
    """
    Load object mesh vertices for ADD metric calculation.
    
    Args:
        obj_name: str, object name from ARCTIC dataset
        
    Returns:
        dict with 'vertices' key containing (N, 3) vertex positions
    """
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed. Install with: pip install trimesh")
        return None
    
    from dexmachina.envs.object import get_arctic_object_cfg
    from pathlib import Path
    
    try:
        obj_cfg = get_arctic_object_cfg(name=obj_name)
        
        # Load both parts of the articulated object
        bottom_path = obj_cfg['bottom_mesh_fname']
        top_path = obj_cfg['top_mesh_fname']
        
        # Load meshes
        bottom_mesh = trimesh.load(bottom_path)
        top_mesh = trimesh.load(top_path)
        
        # Combine vertices from both parts
        all_vertices = np.vstack([
            bottom_mesh.vertices,
            top_mesh.vertices
        ])
        
        print(f"[Object Mesh] Loaded {obj_name}")
        print(f"  Bottom vertices: {bottom_mesh.vertices.shape[0]}")
        print(f"  Top vertices: {top_mesh.vertices.shape[0]}")
        print(f"  Total vertices: {all_vertices.shape[0]}")
        
        return {'vertices': all_vertices}
        
    except Exception as e:
        print(f"Warning: Could not load object mesh: {e}")
        return None
```

### 3. Use in Evaluation Loop

In `main()` function, around line 395-410, modify to:

```python
def main():
    # ... existing setup code ...
    
    # Add this before the evaluation loop (around line 393):
    # Extract object name from env config
    env_cfg = get_all_env_cfg(args)
    obj_name = env_cfg.get('obj_name', 'box')
    
    # Load object mesh for ADD metric
    object_models = load_object_model_for_evaluation(obj_name)
    
    # ... run evaluation loop ...
    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference
        )
        
        # Compute AUC-ADD3 metric WITH mesh vertices
        print("\n" + "="*60)
        print("Computing AUC-ADD3 Metric")
        print("="*60)
        add3_metrics = compute_auc_add3(
            eval_data['obj_state'], 
            eval_data['demo_state'],
            object_models=object_models,  # ← Now passing loaded mesh!
            max_distance=0.1
        )
        
        # ... rest of code ...
```

---

## How It Works

### Object File Organization

```
dexmachina/assets/arctic/box/
├── bottom_watertight_tiny.obj     ← Load this
├── bottom_watertight_tiny.stl     ← Or this (binary format)
├── top_watertight_tiny.obj        ← Load this
└── top_watertight_tiny.stl        ← Or this
```

### What Gets Passed to compute_auc_add3()

```python
# Before (origin point only):
object_models = None
# Result: ADD calculated using only object center point
# Less accurate for detailed shape evaluation

# After (with mesh vertices):
object_models = {
    'vertices': array of shape (N, 3)  # e.g., (1500, 3)
                  where N = total vertices from both parts
}
# Result: ADD calculated using all N vertices
# Much more accurate representation of object shape
```

### What the Metric Actually Computes

For each frame in the trajectory:

1. **Extract poses from state vectors**:
   - Agent: position + quaternion from `obj_states[t, :7]`
   - Demo: position + quaternion from `demo_states[t, :7]`

2. **Transform vertices**:
   - Agent's transformed vertices = rotation_agent @ vertices + position_agent
   - Demo's transformed vertices = rotation_demo @ vertices + position_demo

3. **Compute distances**:
   - For each vertex: distance = ||transformed_agent - transformed_demo||₂
   - ADD error = mean(all vertex distances)

4. **Evaluate at thresholds**:
   - 3cm threshold: % vertices with distance < 0.03m
   - 5cm threshold: % vertices with distance < 0.05m
   - 10cm threshold: % vertices with distance < 0.1m

5. **Compute AUC-ADD3**:
   - Average of accuracy scores across 3 thresholds
   - Range: 0.0 (worst) to 1.0 (best)

---

## Example Output

With mesh vertices loaded:

```
============================================================
Computing AUC-ADD3 Metric
============================================================
[Object Mesh] Loaded box
  Bottom vertices: 754
  Top vertices: 812
  Total vertices: 1566

AUC-ADD3 Score: 0.854231
ADD Mean Error: 0.012456 m
ADD Max Error:  0.045123 m
ADD Min Error:  0.000012 m
============================================================
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'trimesh'"

**Solution:**
```bash
pip install trimesh
```

### Issue: "Could not load object mesh: File not found"

**Likely causes:**
1. Object name doesn't match available objects
2. Asset files not installed

**Available objects:**
- box
- ketchup
- laptop
- mixer
- notebook
- waffleiron

**Fix:** Make sure you're using a valid object name

### Issue: Mesh loading is slow

**Normal for first load** - trimesh needs to parse STL/OBJ files. The vertices are loaded once per evaluation run, not per frame.

**Optimization:** Cache the loaded meshes if running multiple evaluations:

```python
# Load once
mesh_cache = {}

def get_object_model(obj_name):
    if obj_name not in mesh_cache:
        mesh_cache[obj_name] = load_object_model_for_evaluation(obj_name)
    return mesh_cache[obj_name]

# Use in loop
object_models = get_object_model(obj_name)
```

---

## Validation: How to Verify It's Working

### Check 1: Verify Vertices Are Loaded

Add this debugging output to confirm:

```python
# In eval_rl_games.py, after loading object_models
if object_models is not None:
    num_vertices = object_models['vertices'].shape[0]
    print(f"✓ Object model loaded with {num_vertices} vertices")
else:
    print("✗ Using origin point only (no mesh)")
```

### Check 2: Compare Metrics With/Without Mesh

Run evaluation twice:
1. **Without mesh** (original code with `object_models=None`)
2. **With mesh** (modified code with `object_models=load_object_model_for_evaluation(obj_name)`)

Expected differences:
- WITH mesh: More accurate ADD error (should be smaller/more realistic)
- Better discrimination between good and bad policies
- More stable metrics (less dependent on pose calibration)

### Check 3: Visualize in Analysis Plot

The analysis script already has `plot_add3_metric()` which visualizes:
- ADD error over time
- ADD-3 accuracy at each threshold
- Summary statistics

Run analysis:
```bash
python dexmachina/rl/analyze_eval.py logs/path/to/eval_ep0.npy
```

You should see `add3_metric.png` with detailed visualizations.

---

## Comparison: Origin Point vs. Full Mesh

### Using Origin Point Only

```python
# Current compute_auc_add3() default behavior
vertices = np.array([[0, 0, 0]])  # Only center point
# ADD = distance between agent center and demo center
# Issues:
# - Doesn't capture shape differences
# - Sensitive to small rotation errors
# - May underestimate pose errors
```

### Using Full Mesh (1000+ vertices)

```python
# With loaded vertices
vertices = np.vstack([bottom_mesh.vertices, top_mesh.vertices])  # 1500+ points
# ADD = average distance across all surface points
# Benefits:
# - Captures complete shape alignment
# - Robust to local pose errors
# - Industry standard for 6D pose evaluation
```

---

## Technical Notes

### Vertex Ordering

Objects have **two parts with independent mesh files**:

1. **Bottom part** (usually fixed):
   - `bottom_watertight_tiny.obj/stl`
   - Fixed to the object's base link
   - E.g., box body, laptop keyboard

2. **Top part** (usually movable):
   - `top_watertight_tiny.obj/stl`
   - Moves with the articulated joint
   - E.g., box lid, laptop screen

The `compute_auc_add3()` function applies **the same rotation and translation** to all vertices, so combining them works because:
- During evaluation, the agent predicts a single pose for the entire object
- Both parts move together (they're rigidly attached in the physics simulation)
- Only the articulation angle differs, which isn't used for mesh transformation

### Mesh Scale

- Meshes are in **real-world units** (meters)
- Object dimensions typically 0.1-0.3m
- No additional scaling needed

### Quaternion Convention

The code uses **[w, x, y, z]** quaternion format:
- First element: w (scalar/real part)
- Elements 2-4: [x, y, z] (vector/imaginary part)

Example state vector:
```python
state = [x, y, z, w, qx, qy, qz, arti]
       = [0.1, 0.2, 0.3, 0.999, -0.01, 0.01, 0.002, 0.5]
```

---

## Next Steps

1. **Install trimesh**: `pip install trimesh`
2. **Add helper function** to `eval_rl_games.py`
3. **Modify eval loop** to load and use object models
4. **Run evaluation**: `python eval_rl_games.py --checkpoint model.pth`
5. **Analyze results**: `python analyze_eval.py logs/path/eval_ep0.npy`

The ADD3 metric with mesh vertices will now provide accurate 6D pose estimation accuracy!

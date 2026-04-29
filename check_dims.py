import numpy as np

fname = "dexmachina/assets/retargeter_results/allegro_hand/s01/ketchup_use_01_vector.npy"
loaded = np.load(fname, allow_pickle=True).item()

print(f"Keys in file: {loaded.keys()}")
print("\nDimensions:")
for key, val in loaded.items():
    if isinstance(val, dict):
        print(f"  {key}: dict with keys {val.keys()}")
        for subkey, subval in val.items():
            if isinstance(subval, np.ndarray):
                print(f"    {subkey}: {subval.shape} ({subval.dtype})")
            elif isinstance(subval, list):
                if subkey in ['actuated_dof_names']:
                    print(f"    {subkey}: {subval}")
                else:
                    print(f"    {subkey}: list of {len(subval)} items")
            else:
                print(f"    {subkey}: {type(subval)}")
    elif isinstance(val, np.ndarray):
        print(f"  {key}: {val.shape} ({val.dtype})")
    elif isinstance(val, list):
        print(f"  {key}: list of {len(val)} items")
    else:
        print(f"  {key}: {type(val)}")

# # Check if maybe it's the full demo length
# print("\n\nChecking processed ARCTIC data:")
# arctic_fname = "dexmachina/assets/arctic/processed/s01/ketchup_use_01.npy"
# try:
#     arctic_data = np.load(arctic_fname, allow_pickle=True).item()
#     if 'world_coord' in arctic_data:
#         world_coord = arctic_data['world_coord']
#         print(f"Full ARCTIC world_coord keys: {world_coord.keys()}")
#         for key, val in list(world_coord.items())[:2]:  # Just show first 2
#             if isinstance(val, np.ndarray):
#                 print(f"  {key}: {val.shape}")
# except Exception as e:
#     print(f"Could not load ARCTIC data: {e}")

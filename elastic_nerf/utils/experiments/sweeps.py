from typing import Dict, List

# Mapping from sweep name to sweep ID.
SWEEPS: Dict[str, str] = {
    "ngp_occ-blender-baseline": "bj5mkdex",
    "ngp_prop-mipnerf360-baseline": "0rn5ziwc",
}

# Validate that all sweep IDs are unique.
assert len(SWEEPS) == len(set(SWEEPS.values()))

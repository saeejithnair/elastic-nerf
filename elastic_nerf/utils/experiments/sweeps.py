from typing import Dict, List

# Mapping from sweep name to sweep ID.
SWEEPS: Dict[str, str] = {}

# Validate that all sweep IDs are unique.
assert len(SWEEPS) == len(set(SWEEPS.values()))

# %%
from elastic_nerf.utils import wandb_utils as wu
from pathlib import Path
import pandas as pd

sweep_mappings = {
    # "2uxektzo": "ngp_occ-mipnerf360-baseline",
    "kebumdc0": "ngp_occ-mipnerf360-baseline",
    # "xxjsfkbw": "ngp_prop-mipnerf360-baseline",
    "8w0wks0x": "ngp_prop-mipnerf360-baseline",
    "qfkjdvv2": "ngp_occ-mipnerf360-sampling_single",
    "hy03dx0e": "ngp_occ-mipnerf360-sampling",
    "wsxh6gjo": "ngp_prop-mipnerf360-sampling",
    "8ishbvau": "ngp_prop-mipnerf360-sampling_single",
    "b674pjcs": "ngp_occ-mipnerf360-baseline_head_depth1",
    "58hgroe5": "ngp_prop-mipnerf360-baseline_head_depth1",
    "c6g1mc5g": "ngp_occ-mipnerf360-baseline-mup",
    "ccrwhsr5": "ngp_prop-mipnerf360-baseline-mup",
}

tables = ["EvalResultsSummarytable"]
sweeps = sweep_mappings.keys()
results_cache_dir = Path("/home/user/shared/results/elastic-nerf")
sweep_results = {}

for sweep in sweeps:
    sweep_results[sweep] = wu.fetch_sweep_results(
        sweep=sweep,
        refresh_cache=False,
        download_history=True,
        tables=tables,
        results_cache_dir=results_cache_dir,
    )

# %%

all_history = []
# Create a dataframe with all the results
for sweep_name in sweep_results:
    for run in sweep_results[sweep_name]:
        # Flatten the config
        flat_config = wu.flatten_dict(run.config, sep=".")
        # Concatenate the config with each row of the history results
        # Note that history results are already a dataframe
        history = run.history
        history["sweep_id"] = sweep_name
        history["run_id"] = run.run_id
        history["model_type"] = (
            "ngp_prop" if "prop" in sweep_mappings[sweep_name] else "ngp_occ"
        )
        history["sweep_name"] = sweep_mappings[sweep_name]
        for key in flat_config:
            try:
                history[key] = str(flat_config[key])
            except:
                print(f"Failed to add {key} to history with value {flat_config[key]}")
                raise
        all_history.append(history)

# %%
# Concatenate all the history results into a single dataframe
final_df = pd.concat(all_history, ignore_index=True)


# %%
fp = f"{'_'.join(sweeps)}_all_results.csv"
final_df.to_csv(fp, index=False)
print(f"Saved results to {fp}")
# %%

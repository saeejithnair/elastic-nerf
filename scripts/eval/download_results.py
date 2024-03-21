# %%
from elastic_nerf.utils import wandb_utils as wu
from pathlib import Path
import pandas as pd

sweep_mappings = {
    "2uxektzo": "ngp_occ-mipnerf360-baseline",
    "xxjsfkbw": "ngp_prop-mipnerf360-baseline",
    "qfkjdvv2": "ngp_occ-mipnerf360-sampling_single",
    "hy03dx0e": "ngp_occ-mipnerf360-sampling",
}
tables = ["EvalResultsSummarytable"]
sweeps = ["2uxektzo", "xxjsfkbw", "qfkjdvv2", "hy03dx0e"]
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
final_df.to_csv(f"{'_'.join(sweeps)}_all_results.csv", index=False)
# %%

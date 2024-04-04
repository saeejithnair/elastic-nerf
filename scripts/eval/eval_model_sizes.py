# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Sequence, Dict
from elastic_nerf.nerfacc.trainers.ngp_prop import NGPPropTrainer, NGPPropTrainerConfig
from elastic_nerf.nerfacc.trainers.ngp_occ import NGPOccTrainer, NGPOccTrainerConfig
import torch


# %%
def count_params(model, modules_to_skip=["encoding", "estimator"]) -> int:
    sum = 0
    for n, p in model.named_parameters():
        skip_module = False
        if not p.requires_grad:
            continue
        for m in modules_to_skip:
            if m in n:
                skip_module = True
                break
        if not skip_module:
            print(n, p.numel())
            sum += p.numel()

    return sum


def extract_params(models_info):
    params_dict = {}
    for width, model_info in models_info.items():
        for module_name, module in model_info.items():
            print(module_name)
            if isinstance(module, Sequence):
                for i, m in enumerate(module):
                    param_count = count_params(m)
                    if param_count == 0:
                        continue
                    if module_name not in params_dict:
                        params_dict[module_name] = {}
                    params_dict[module_name][f"{width}_{i}"] = param_count
            else:
                param_count = count_params(module)
                if param_count == 0:
                    continue
                if module_name not in params_dict:
                    params_dict[module_name] = {}
                params_dict[module_name][width] = param_count
    return params_dict


def plot_module_params(params_dict):
    for module_name, width_params in params_dict.items():
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Absolute parameter counts
        widths = list(width_params.keys())
        param_counts = list(width_params.values())
        color = "tab:blue"
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Parameter Count", color=color)
        ax1.bar(
            widths,
            param_counts,
            color=color,
            alpha=0.6,
            label="Absolute Parameter Count",
        )
        ax1.tick_params(axis="y", labelcolor=color)

        # Relative parameter counts (percentage)
        ax2 = ax1.twinx()
        max_count = max(param_counts)
        relative_counts = [(count / max_count) * 100 for count in param_counts]
        color = "tab:red"
        ax2.set_ylabel("Relative Parameter Count (%)", color=color)
        ax2.plot(
            widths,
            relative_counts,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label="Relative Parameter Count",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Combined legend for both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")

        # Final plot adjustments
        plt.title(f"Module: {module_name}")
        plt.xticks(widths)
        plt.show()


def generate_param_tables(params_dict):
    for module_name, width_params in params_dict.items():
        print(f"Module: {module_name}")
        df = pd.DataFrame(
            list(width_params.items()), columns=["Width", "Parameter Count"]
        )
        print(df, "\n")


# %%
# Evaluate NGP-Prop and NGP-Occ models across specified widths
ngp_prop_config = NGPPropTrainerConfig(
    enable_logging=False,
    fused_eval=False,
    device="cpu",
    scene="counter",
    dataset_name="mipnerf360",
)
ngp_prop_config.radiance_field.use_elastic = True
ngp_prop_config.radiance_field.use_elastic_head = True
ngp_prop_config.density_field.use_elastic = True
ngp_prop_trainer = ngp_prop_config.setup()

ngp_occ_config = NGPOccTrainerConfig(
    enable_logging=False,
    fused_eval=False,
    device="cpu",
    scene="counter",
    dataset_name="mipnerf360",
)
ngp_occ_config.radiance_field.use_elastic = True
ngp_occ_trainer = ngp_occ_config.setup()

# %%
ngp_prop_config.seed = 42
ngp_prop_config.num_train_widths = 1
ngp_prop_config.num_eval_elastic_widths = 1
ngp_prop_config.hidden_dim = 8
ngp_prop_trainer = ngp_prop_config.setup()
modules = ngp_prop_trainer.get_modules_for_eval(ngp_prop_config.hidden_dim)
unpacked_modules = {}
for k, v in modules.items():
    if isinstance(v, Sequence):
        for i, m in enumerate(v):
            unpacked_modules[f"{k}_{i}"] = m
    else:
        unpacked_modules[k] = v
for module_name, module in unpacked_modules.items():
    for n, p in module.named_parameters():
        if "layer" not in n or "norm" in n:
            continue
        if p.ndim == 1:
            p = p.unsqueeze(-1)
        norm = torch.linalg.matrix_norm(p, ord=2).item()
        var = p.var().item()
        print(n, p.shape, f"Norm: {norm:.3f}", f"Var: {var:.3f}")

# %%
# widths = range(8, 65, 8)
widths = [8, 16, 32, 64]
ngp_occ_info = {width: ngp_occ_trainer.get_modules_for_eval(width) for width in widths}

ngp_prop_info = {
    width: ngp_prop_trainer.get_modules_for_eval(width) for width in widths
}

# Extract and organize parameters
ngp_occ_params = extract_params(ngp_occ_info)
ngp_prop_params = extract_params(ngp_prop_info)

# Plot parameters
plot_module_params(ngp_occ_params)
plot_module_params(ngp_prop_params)

# Generate tables
generate_param_tables(ngp_occ_params)
generate_param_tables(ngp_prop_params)
# %%

table_cols = [
    "Width",
    "Radiance Field",
    "Proposal Network 1",
    "Proposal Network 2",
    "Total",
]
table_data = []
base_width = 64
base_params = {}
for width in widths[::-1]:
    table_row = [width]
    total = 0
    for module_name, width_params in ngp_prop_params.items():
        if module_name == "radiance_field":
            if width == base_width:
                base_params[module_name] = width_params[base_width]
                table_row.append(width_params[base_width])
            else:
                reduction = width_params[width] / base_params[module_name]
                table_row.append(f"{width_params[width]} ({1/reduction:.2f}x)")
            total += width_params[width]
        elif module_name == "proposal_networks":
            if width == base_width:
                base_params[f"{width}_0"] = width_params[f"{base_width}_0"]
                base_params[f"{width}_1"] = width_params[f"{base_width}_1"]
                table_row.append(width_params[f"{base_width}_0"])
                table_row.append(width_params[f"{base_width}_1"])
            else:
                reduction_0 = (
                    width_params[f"{width}_0"] / base_params[f"{base_width}_0"]
                )
                reduction_1 = (
                    width_params[f"{width}_1"] / base_params[f"{base_width}_1"]
                )
                table_row.append(f"{width_params[f'{width}_0']} ({1/reduction_0:.2f}x)")
                table_row.append(f"{width_params[f'{width}_1']} ({1/reduction_1:.2f}x)")
            total += width_params[f"{width}_0"] + width_params[f"{width}_1"]

    if width == base_width:
        base_params["total"] = total
        table_row.append(total)
    else:
        table_row.append(f"{total} ({1/(total / base_params['total']):.2f}x)")

    table_data.append(table_row)

prop_df = pd.DataFrame(table_data, columns=table_cols)
print(prop_df.to_latex(index=False))
# %%
table_cols = [
    "Width",
    "Radiance Field",
    "Total",
]
table_data = []
base_width = 64
base_params = {}
for width in widths[::-1]:
    table_row = [width]
    total = 0
    for module_name, width_params in ngp_occ_params.items():
        if module_name == "radiance_field":
            if width == base_width:
                base_params[module_name] = width_params[base_width]
                table_row.append(width_params[base_width])
            else:
                reduction = width_params[width] / base_params[module_name]
                table_row.append(f"{width_params[width]} ({1/reduction:.2f}x)")
            total += width_params[width]
    if width == base_width:
        base_params["total"] = total
        table_row.append(total)
    else:
        table_row.append(f"{total} ({1/(total / base_params['total']):.2f}x)")

    table_data.append(table_row)

occ_df = pd.DataFrame(table_data, columns=table_cols)
print(occ_df.to_latex(index=False))

# %%
occ_cols = ["Radiance Field", "Total"]
prop_cols = ["Radiance Field", "Proposal Network 1", "Proposal Network 2", "Total"]
table_cols = ["Width"] + occ_cols + prop_cols

header_row1 = [
    "\multicolumn{1}{c}{} & \multicolumn{2}{c|}{NGP Occ} & \\multicolumn{4}{c}{NGP Prop} \\\\",
]
header_row2 = ["\\textbf{Width}"] + table_cols[1:]
header = (
    " & ".join(header_row1)
    + " \\midrule \n"
    + " & ".join(header_row2)
    + " \\\\ \\midrule"
)
table_data = []
for width in widths[::-1]:
    row = "\\textbf{" + f"{width}" + "}"
    for c in occ_cols:
        row += f" & {occ_df.query(f'Width == {width}')[c].iloc[0]}"
    for c in prop_cols:
        row += f" & {prop_df.query(f'Width == {width}')[c].iloc[0]}"
    row += " \\\\\n"
    table_data.append(row)
table_body = " \\\\n".join([" ".join([row for row in table_data])]) + " \\\\"
final_table = (
    "\\begin{table*}[h]\n\\centering\n\\small\n"
    "\\caption{Combined Baseline PSNR after 20k steps of training for NGP Occ and NGP Prop models at different widths across scenes from the MipNeRF-360 dataset}\n"
    "\\label{tab:combined_params}\n"
    "\\begin{tabular}{lcc|cccc}\n\\toprule\n"
    + header
    + "\n"
    + table_body
    + "\n\\bottomrule\n"
    "\\end{tabular}\n\\end{table*}"
)
print(final_table)
# %%

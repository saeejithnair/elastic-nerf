import io
import os
from pathlib import Path
from typing import Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from adjustText import adjust_text
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.ticker import ScalarFormatter
from PIL import Image


def interactive_2d_colorplot(
    df,
    x_col,
    y_col,
    z_cols,
    title,
    xaxis_title,
    yaxis_title,
    extra_hover_cols,
    baseline_values,
):
    # Create traces
    data = []
    baseline_data = []  # Separate list for baseline traces
    for i, z_col in enumerate(z_cols):
        hover_text = df[extra_hover_cols].apply(
            lambda row: "<br>".join(
                [
                    f"<b>{col}</b>: {val}"
                    for col, val in zip(row.index, row.values.astype(str))
                ]
            ),
            axis=1,
        )
        trace = go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                color=df[z_col],
                colorscale="Viridis",
                colorbar=dict(title=z_col),
            ),
            name=z_col,
            visible=i == 0,  # Only make the first trace visible
            text=hover_text,
            hovertemplate="<b>"
            + x_col
            + "</b>: %{x}<br>"
            + "<b>"
            + y_col
            + "</b>: %{y}<br>"
            + "<b>"
            + z_col
            + "</b>: %{marker.color}<br>"
            + "%{text}<extra></extra>",
        )
        data.append(trace)

    for i, z_col in enumerate(z_cols):
        # Add baseline trace for each metric
        baseline_trace = go.Scatter(
            x=[baseline_values[x_col]],
            y=[baseline_values[y_col]],
            mode="markers",
            marker=dict(
                color=[baseline_values[z_col]],
                colorscale="Viridis",
                symbol="star-dot",
                size=10,
                cmin=min(df[z_col]),
                cmax=max(df[z_col]),
            ),
            name="Baseline (" + z_col + ")",
            text=[f"<b>{z_col}</b>: {baseline_values[z_col]}"],
            hovertemplate="<b>Baseline Vanilla</b><br>"
            "<b>" + x_col + "</b>: %{x}<br>"
            "<b>" + y_col + "</b>: %{y}<br>"
            "%{text}<extra></extra>",
            visible=i == 0,  # Only make the first baseline trace visible
        )
        baseline_data.append(baseline_trace)

    # Combine original and baseline traces
    data.extend(baseline_data)

    # Define updatemenus
    buttons = []
    for i, z_col in enumerate(z_cols):
        visible_status = [j == i for j in range(len(z_cols))] + [
            j == i for j in range(len(z_cols))
        ]
        button = dict(
            label=z_col,
            method="update",
            args=[
                {"visible": visible_status},
                {"title": f"{title} ({z_col})"},
            ],
        )
        buttons.append(button)

    updatemenus = [
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.15,
            xanchor="right",
            y=1.2,
            yanchor="top",
        )
    ]

    # Define layout
    layout = dict(
        title=f"{title} ({z_cols[0]})",
        showlegend=False,
        updatemenus=updatemenus,
        title_x=0.5,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    fig = dict(data=data, layout=layout)
    return go.Figure(fig)


def find_best_run_weighted(
    df: pd.DataFrame,
    metric_name: str,
    target_metric: float,
    weight_coarse: float,
    weight_fine: float,
    maximize_metric: bool,
) -> pd.DataFrame:
    """
    This function finds the best run with a metric greater than or equal to the target metric and the smallest total
    weighted flop ratio for given weights of the coarse field and the fine field.
    """
    # Filter the runs with a metric greater than or equal to the target
    if maximize_metric:
        df_metric_ge_target = df[df[metric_name] >= target_metric].copy()
    else:
        df_metric_ge_target = df[df[metric_name] <= target_metric].copy()

    # Compute the weighted sum of Flop Ratio vs. Vanilla for both fields
    df_metric_ge_target["total_flop_ratio_weighted"] = (
        weight_coarse * df_metric_ge_target["Coarse Field Flop Ratio vs. Vanilla"]
        + weight_fine * df_metric_ge_target["Fine Field Flop Ratio vs. Vanilla"]
    )

    # Sort the runs by 'total_flop_ratio_weighted' in ascending order and the metric in descending order if maximize_metric is True, else ascending order
    df_metric_ge_target.sort_values(
        by=["total_flop_ratio_weighted", metric_name],
        ascending=[True, not maximize_metric],
        inplace=True,
    )

    # Return the run with the smallest 'total_flop_ratio_weighted' and the highest metric value among them if maximize_metric is True, else the lowest
    return df_metric_ge_target.head(1)


def find_best_runs_sorted(
    df: pd.DataFrame, metric_name: str, target_metric: float, maximize_metric: bool
) -> pd.DataFrame:
    """
    This function finds the best run with a metric greater than or equal to the target metric and the smallest "Overall Params Ratio vs. Vanilla".
    """
    # Filter the runs with a metric greater than or equal to the target
    if maximize_metric:
        df_metric_ge_target = df[df[metric_name] >= target_metric].copy()
    else:
        df_metric_ge_target = df[df[metric_name] <= target_metric].copy()

    # Sort the runs by 'Overall Params Ratio vs. Vanilla' in ascending order and the metric in descending order if maximize_metric is True, else ascending order
    df_metric_ge_target.sort_values(
        by=["Overall Params Ratio vs. Vanilla", metric_name],
        ascending=[True, not maximize_metric],
        inplace=True,
    )

    # Return all the sorted runs
    return df_metric_ge_target


def find_best_run(
    df: pd.DataFrame, metric_name: str, target_metric: float, maximize_metric: bool
) -> pd.DataFrame:
    # Return the run with the smallest 'Overall Params Ratio vs. Vanilla' and the highest metric value among them if maximize_metric is True, else the lowest
    return find_best_runs_sorted(df, metric_name, target_metric, maximize_metric).head(
        1
    )


def find_best_runs_for_range(
    df: pd.DataFrame,
    metric_name: str,
    metric_start: float,
    metric_end: float,
    metric_step: float,
    maximize_metric: bool,
    weights: Optional[Tuple[float, float]] = None,
    metric_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    This function finds the best runs for a range of metric values.
    """
    if metric_values is None:
        metric_values = np.arange(metric_start, metric_end, metric_step)
    best_runs = []

    for target_metric in metric_values:
        if weights:
            weight_coarse, weight_fine = weights
            best_run_df = find_best_run_weighted(
                df,
                metric_name,
                target_metric,
                weight_coarse,
                weight_fine,
                maximize_metric,
            )
        else:
            best_run_df = find_best_run(df, metric_name, target_metric, maximize_metric)

        if not best_run_df.empty:
            best_runs.append(best_run_df)

    # Concatenate all the dataframes
    best_runs_df = pd.concat(best_runs)

    # Add the target metric values to the DataFrame
    best_runs_df[metric_name + "_target"] = metric_values[: len(best_runs_df)]

    best_runs_df = best_runs_df.sort_values(
        by=["Overall Params Ratio vs. Vanilla"], ascending=False
    )

    arch_labels = ["NAS-NeRF S", "NAS-NeRF XS", "NAS-NeRF XXS"]
    best_runs_df["Optimal Architecture"] = arch_labels

    return best_runs_df


def annotate_points(ax, df, x_field, y_field, label_lambda):
    """
    Annotate a selection of points with their metric values.
    """
    texts = []
    for idx in range(len(df)):
        x_val = df.iloc[idx][x_field]
        y_val = df.iloc[idx][y_field]
        arch_label = df.iloc[idx]["Optimal Architecture"]
        label = f"{arch_label}\n{label_lambda(df.iloc[idx])}"

        annotation = ax.text(
            x_val,
            y_val,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )
        texts.append(annotation)
    return texts


def plot_optimal_models(
    df,
    metric_name,
    metric_start,
    metric_end,
    metric_step,
    weight_coarse,
    weight_fine,
):
    """
    This function plots optimal models versus all models for different metric
    values and weight combinations.
    """
    weights = (weight_coarse, weight_fine)
    maximize_metric = True
    metric_label_arrow = "↑"
    if metric_name == "LPIPS":
        maximize_metric = False
        metric_label_arrow = "↓"
    # Find the best runs for a range of metric values
    best_runs_df_range = find_best_runs_for_range(
        df,
        metric_name,
        metric_start,
        metric_end,
        metric_step,
        maximize_metric,
        weights=weights,
    )

    # Create the 2D plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the colormap
    cmap = plt.get_cmap("viridis_r")

    # Plot all runs as small coloured dots
    ax.scatter(
        df["Coarse Field Flop Ratio vs. Vanilla"],
        df["Fine Field Flop Ratio vs. Vanilla"],
        c=df[metric_name],
        cmap=cmap,
        s=10,  # Use small dots
        alpha=0.5,  # Make the dots semi-transparent
    )

    # Plot the best runs for each metric target as larger points
    scatter = ax.scatter(
        best_runs_df_range["Coarse Field Flop Ratio vs. Vanilla"],
        best_runs_df_range["Fine Field Flop Ratio vs. Vanilla"],
        c=best_runs_df_range[metric_name + "_target"],
        cmap=cmap,
        s=50,  # Use larger dots
        edgecolors="red",  # Add edgecolors for visibility
        linewidths=1,  # Width of the edgecolors
        label=f"Optimal Architecture",
        marker="o",  # Use circle marker
    )

    # # Annotate the best runs
    # texts = annotate_points(
    #     ax=ax,
    #     df=best_runs_df_range,
    #     x_field="Coarse Field Flop Ratio vs. Vanilla",
    #     y_field="Fine Field Flop Ratio vs. Vanilla",
    #     label_lambda=lambda row: f"{row[metric_name + '_target']:.2f}",
    # )

    # adjust_text(texts, force_text=0.5, expand_text=(1, 1), ax=ax)

    texts = annotate_points(
        ax,
        best_runs_df_range,
        x_field="Coarse Field Flop Ratio vs. Vanilla",
        y_field="Fine Field Flop Ratio vs. Vanilla",
        label_lambda=lambda row: f"{row[metric_name + '_target']:.2f}",
    )

    adjust_text(
        texts,
        ax=ax,
        force_points=0.3,
        force_text=0.3,
        expand_points=(1.2, 1.2),
        expand_text=(1, 1),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    )

    # Add labels and title
    ax.set_xlabel("Coarse Field Compression Ratio (flops)")
    ax.set_ylabel("Fine Field Compression Ratio (flops)")
    ax.set_title(
        f"Optimal Architectures for Target Metric: {metric_name} ($\lambda_{{\mathrm{{coarse}}}}={weight_coarse}, \lambda_{{\mathrm{{fine}}}}={weight_fine}$)"
    )
    # Define the normalization for the colorbar
    norm = plt.Normalize(df[metric_name].min(), df[metric_name].max())

    # Create a new ScalarMappable that uses the colormap and normalization
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # Create the colorbar using the ScalarMappable
    fig.colorbar(sm, ax=ax, label=metric_name)

    # Add a legend to show the weight combination
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.show()

    return best_runs_df_range


def plot_optimal_models_params(
    df,
    metric_name,
    ax,
    fig,
    metric_start: Optional[float] = None,
    metric_end: Optional[float] = None,
    metric_step: Optional[float] = None,
    metric_values=None,
    ratio: str = "flops",
    dataset: str = "lego",
):
    """
    This function plots optimal models versus all models for different metric values.
    """
    maximize_metric = True
    metric_label_arrow = "↑"
    if metric_name == "LPIPS":
        maximize_metric = False
        metric_label_arrow = "↓"

    # Find the best runs for a range of metric values
    best_runs_df_range = find_best_runs_for_range(
        df,
        metric_name,
        metric_start,
        metric_end,
        metric_step,
        maximize_metric,
        metric_values=metric_values,
    )

    # Define the colormap.
    cmap = plt.get_cmap("viridis")

    if ratio == "params":
        ratio_key = "Architecture Efficiency Ratio (Params)"
    elif ratio == "flops":
        ratio_key = "Architecture Efficiency Ratio (FLOPs)"
    else:
        raise ValueError(f"Invalid ratio: {ratio}")

    # colorbar_key = "Field FLOP Ratios Harmonic Mean"
    colorbar_key = "Field Tradeoff Ratio"
    # Define the normalization for the colorbar
    norm = plt.Normalize(
        df[colorbar_key].min(),
        df[colorbar_key].max(),
    )  # Change the normalization based on the FPS (normalized) value
    # norm = LogNorm(
    #     vmin=df[colorbar_key].min(),
    #     vmax=df[colorbar_key].max(),
    # )
    # norm = PowerNorm(
    #     gamma=0.5,
    #     vmin=df[colorbar_key].min(),
    #     vmax=df[colorbar_key].max(),
    # )

    # Plot all runs as small coloured dots
    bubble_size_multiplier = 300
    ax.scatter(
        df[ratio_key],
        df[metric_name],
        norm=norm,
        c=df[colorbar_key],  # Change the color based on the FPS (normalized) value
        cmap=cmap,
        s=df["Overall Params Ratio vs. Vanilla"]
        * bubble_size_multiplier,  # Use small dots
        alpha=0.5,  # Make the dots semi-transparent
    )

    # Plot the best runs for each metric target with red edge
    scatter = ax.scatter(
        best_runs_df_range[ratio_key],
        best_runs_df_range[metric_name],
        c=best_runs_df_range[
            colorbar_key
        ],  # Change the color based on the FPS (normalized) value
        cmap=cmap,
        norm=norm,
        s=best_runs_df_range["Overall Params Ratio vs. Vanilla"]
        * bubble_size_multiplier,
        edgecolors="red",  # Add edgecolors for visibility
        linewidths=1,  # Width of the edgecolors
        label=f"Generated Architectures",
        marker="o",  # Use circle marker
    )

    # Annotate the best runs
    texts = annotate_points(
        ax=ax,
        df=best_runs_df_range,
        x_field=ratio_key,
        y_field=metric_name,
        label_lambda=lambda row: f"FPS: {row['FPS (normalized)']:.2f}x",
    )
    adjust_text(
        texts,
        ax=ax,
        force_points=0.3,
        force_text=0.3,
        expand_points=(1.2, 1.2),
        expand_text=(1, 1),
    )

    # Add labels and title
    ax.set_xlabel(ratio_key)
    ax.set_ylabel(f"{metric_name} {metric_label_arrow}")
    ax.set_title(
        f"Subset of NAS-NeRF architecture space evaluated on Blender {dataset.capitalize()} scene"
    )

    # Create a new ScalarMappable that uses the colormap and normalization
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the colorbar using the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, label="Field FLOPs Tradeoff Ratio")
    # ticks = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200]
    # cbar.set_ticks(ticks)

    # # Set tick labels (you can customize this if needed)
    # cbar.set_ticklabels([str(t) for t in ticks])

    # Add a legend to show the weight combination
    ax.legend(loc="best")
    ax.grid(True)

    return best_runs_df_range


def extract_metrics_and_architecture(df):
    # Define the metrics and architecture related columns
    metrics_columns = [
        "SSIM",
        "PSNR",
        "LPIPS",
        "SSIM (Best)",
        "LPIPS (Best)",
        "PSNR (Best)",
        "FPS (subset)",
        "FPS (normalized)",
        "dataset",
    ]
    architecture_columns = [
        "Coarse Field Arch Style",
        "Fine Field Arch Style",
        "Coarse Field Target Ratio",
        "Fine Field Target Ratio",
        "Coarse Field Depth (Stage 1)",
        "Coarse Field Channels (Stage 1)",
        "Coarse Field Depth (Stage 2)",
        "Coarse Field Channels (Stage 2)",
        "Coarse Field Depth (Stage 3)",
        "Coarse Field Channels (Stage 3)",
        "Fine Field Depth (Stage 1)",
        "Fine Field Channels (Stage 1)",
        "Fine Field Depth (Stage 2)",
        "Fine Field Channels (Stage 2)",
        "Fine Field Depth (Stage 3)",
        "Fine Field Channels (Stage 3)",
        "Architecture Efficiency Ratio (Params)",
        "Architecture Efficiency Ratio (FLOPs)",
        "Coarse Field Flop Ratio vs. Vanilla",
        "Fine Field Flop Ratio vs. Vanilla",
        "Overall Params Ratio vs. Vanilla",
        "Overall FLOPs Ratio vs. Vanilla",
        "Overall Params",
        "Overall FLOPs",
        "Optimal Architecture",
    ]

    # Extract the desired columns from the dataframe
    metrics_df = df[metrics_columns]
    architecture_df = df[metrics_columns + architecture_columns]

    # Output the architecture in a readable format
    architecture_df = architecture_df.applymap(
        lambda x: f"Depth: {x[0]}, Channels: {x[1]}"
        if isinstance(x, list) and len(x) == 2
        else x
    )

    return metrics_df, architecture_df


def parse_architecture_from_df(data):
    architecture_info = {
        "Coarse Field Depths": [
            data["Coarse Field Depth (Stage 1)"],
            data["Coarse Field Depth (Stage 2)"],
            data["Coarse Field Depth (Stage 3)"],
        ],
        "Coarse Field Channels": [
            data["Coarse Field Channels (Stage 1)"],
            data["Coarse Field Channels (Stage 2)"],
            data["Coarse Field Channels (Stage 3)"],
        ],
        "Fine Field Depths": [
            data["Fine Field Depth (Stage 1)"],
            data["Fine Field Depth (Stage 2)"],
            data["Fine Field Depth (Stage 3)"],
        ],
        "Fine Field Channels": [
            data["Fine Field Channels (Stage 1)"],
            data["Fine Field Channels (Stage 2)"],
            data["Fine Field Channels (Stage 3)"],
        ],
        "FPS": data["FPS (normalized)"],
        "Parameters": data["Overall Params"],
        "FLOPs": data["Overall FLOPs"],
        "Architecture Efficiency Ratio (Params)": data[
            "Architecture Efficiency Ratio (Params)"
        ],
        "Architecture Efficiency Ratio (FLOPs)": data[
            "Architecture Efficiency Ratio (FLOPs)"
        ],
        "Coarse FR": data["Coarse Field Flop Ratio vs. Vanilla"],
        "Fine FR": data["Fine Field Flop Ratio vs. Vanilla"],
    }

    return architecture_info


def plot_models_params(
    df,
    metric_name,
    ax,
    fig,
    architecture_to_marker,
    architecture_to_color,
    ratio: str = "flops",
    legend_loc: str = "best",
):
    """
    This function plots optimal models versus all models for different metric values.
    """
    maximize_metric = True
    metric_label_arrow = "↑"
    metric_label = metric_name
    if metric_name == "LPIPS":
        maximize_metric = False
        metric_label_arrow = "↓"
    elif metric_name == "FPS (normalized)":
        metric_label = "Relative FPS"

    if ratio == "params":
        ratio_key = "Architecture Efficiency Ratio (Params)"
    elif ratio == "flops":
        ratio_key = "Architecture Efficiency Ratio (FLOPs)"
    else:
        raise ValueError(f"Invalid ratio: {ratio}")

    for architecture, marker in architecture_to_marker.items():
        subset_df = df[df["Optimal Architecture"] == architecture]
        color = architecture_to_color[architecture]

        if architecture == "NeRF":
            xval = [1 for _ in range(len(subset_df))]
        else:
            xval = subset_df[ratio_key]

        ax.scatter(
            xval,
            subset_df[metric_name],
            c=color,
            s=subset_df["Overall Params Ratio vs. Vanilla"] * 250,
            alpha=0.5,
            marker=marker,
            label=architecture,
        )

    texts = []
    for idx in range(len(df)):
        if (
            metric_name == "FPS (normalized)"
            and df["Optimal Architecture"].iloc[idx] == "NeRF"
        ):
            continue
        x_val = df.iloc[idx][ratio_key]
        y_val = df.iloc[idx][metric_name]
        label = df.iloc[idx]["dataset"].capitalize()

        annotation = ax.text(
            x_val,
            y_val,
            label,
            fontsize=8,
            ha="center",
            va="center",
        )
        texts.append(annotation)

    adjust_text(
        texts,
        ax=ax,
        force_points=0.3,
        force_text=0.3,
        expand_points=(1.2, 1.2),
        expand_text=(1, 1),
    )

    # Add labels and title
    ax.set_xlabel("Architecture Efficiency Ratio ↑")
    ax.set_ylabel(f"{metric_label} {metric_label_arrow}")
    ax.set_title(
        f"Baseline NeRF vs. generated NAS-NeRF architectures on Blender synthetic scenes"
    )

    if metric_name == "SSIM":
        ax.set_yscale("log")
        ticks = [0.7, 0.75, 0.8, 0.85, 0.9, 1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])

    # Add a legend to show the weight combination
    ax.legend(loc=legend_loc)
    ax.grid(True)


def video_to_gif(video_path, dest_dir, fps=30):
    """
    Convert a video file to a GIF and save it to the specified directory.

    Parameters:
    - video_path: str, path to the video file.
    - dest_dir: Path, directory to save the generated GIF.
    - fps: int, frames per second for the output GIF.
    """
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Determine the GIF filename
    gif_filename = Path(video_path).stem + ".gif"
    gif_path = dest_dir / gif_filename

    # Read video frames
    video = imageio.get_reader(video_path)
    frames = [im for im in video.iter_data()]

    # Write frames to GIF
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video converted to GIF and saved at {gif_path}")


def generate_mlp_heatmap(model: nn.Module, model_name: str):
    """
    Creates and logs heatmaps of weights and biases for each layer of the given MLP model,
    and a combined heatmap representing the mean of weights and biases of each layer.

    Args:
        model (nn.Module): The MLP model to visualize.

    Returns:
        A dictionary containing the buffers (in-memory files) for each layer's heatmap and the combined heatmap.
    """

    layer_tensors = {}
    layer_means = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Prepare weight matrix with bias appended
            weight_bias_matrix = torch.hstack(
                (module.weight.detach().cpu(), module.bias.detach().cpu().view(-1, 1)),
            )

            # Create heatmap for each layer
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(weight_bias_matrix.numpy(), cmap="viridis")
            ax.set_title(f"{name} Weights and Bias")
            ax.set_xlabel("Connections to Previous Layer")
            ax.set_ylabel("Neurons and Bias")

            # Convert the plot to an in-memory file
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            image = Image.open(buf)
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).float()
            layer_tensors[f"{model_name}/{name}"] = image_tensor

            plt.close()

            # Calculate the mean across the first dimension (mean of each neuron)
            layer_mean = weight_bias_matrix.mean(dim=1)
            layer_means.append(layer_mean)
            layer_names.append(name)

    # Find the maximum size
    max_size = max([mean.size(0) for mean in layer_means])

    # Pad each tensor in layer_means to max_size
    padded_means = [F.pad(mean, (0, max_size - mean.size(0))) for mean in layer_means]

    if len(padded_means) > 0:
        # Create combined heatmap using the padded means
        combined_matrix = torch.stack(padded_means, dim=1)
        plt.figure(figsize=(len(layer_means) * 2, 8))
        ax = sns.heatmap(combined_matrix.numpy(), cmap="viridis")
        ax.set_title("MLP Layer Means")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Mean Weights and Biases")
        ax.set_xticklabels(layer_names)

        # Convert the combined plot to an in-memory file
        buf_combined = io.BytesIO()
        plt.savefig(buf_combined, format="png")
        buf_combined.seek(0)

        image_combined = Image.open(buf_combined)
        image_combined_array = np.array(image_combined)
        image_combined_tensor = torch.from_numpy(image_combined_array).float()
        layer_tensors[f"{model_name}/combined"] = image_combined_tensor

        plt.close()

    return layer_tensors

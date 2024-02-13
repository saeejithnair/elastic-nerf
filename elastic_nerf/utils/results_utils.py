from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def filter_columns_by_substring(
    df: pd.DataFrame, substrings: List[str]
) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only columns whose labels contain any of the specified substrings.

    Args:
        df: The DataFrame to filter.
        substrings: A list of substrings to check in column labels.

    Returns:
        A DataFrame with only the columns that contain any of the specified substrings.
    """
    # Identify columns that contain any of the specified substrings
    columns_to_keep = [
        col for col in df.columns if any(sub in col for sub in substrings)
    ]

    # Filter the DataFrame to include only these columns
    filtered_df = df[columns_to_keep]

    return filtered_df


class DataTablePlotter:
    def __init__(self):
        self.dataframes = {}
        self.configs = {}
        self.current_dataframe = None
        self.current_config = None

    def add_dataframe(
        self,
        name,
        dataframe,
        column_mapping=None,
        format_mapping=None,
        color_mapping=None,
    ):
        self.dataframes[name] = dataframe
        if column_mapping is not None:
            dataframe.rename(columns=column_mapping, inplace=True)

            # Reorder columns based on the mapping order
            dataframe = dataframe[column_mapping.values()]
            self.dataframes[name] = dataframe

        self.configs[name] = self.create_table_config(
            dataframe, format_mapping, color_mapping
        )

    def get_remapped_dataframe(self, name):
        return self.dataframes[name]

    def create_table_config(self, dataframe, format_mapping=None, color_mapping=None):
        if format_mapping is None:
            format_mapping = {}
        if color_mapping is None:
            color_mapping = {}

        if "ALL_COLUMNS" in color_mapping:
            colors = [color_mapping["ALL_COLUMNS"] for col in dataframe.columns]
        else:
            colors = [color_mapping.get(col, "white") for col in dataframe.columns]

        config = {
            "header": {
                "values": list(dataframe.columns),
                "align": "left",
                "fill_color": "lightgrey",
                "font": dict(color="black", size=12),
            },
            "cells": {
                "values": [dataframe[col] for col in dataframe.columns],
                "align": "left",
                "font": dict(color="darkslategray", size=11),
                "format": [format_mapping.get(col, None) for col in dataframe.columns],
                "fill": {"color": colors},
            },
        }
        return config

    def prepare_figure(self):
        # Add the first table as default
        for config in self.configs.values():
            self.fig = go.Figure(data=[go.Table(**config)])

        # Prepare buttons for toggling
        buttons = []
        for name in self.configs.keys():
            buttons.append(
                dict(
                    label=name,
                    method="update",
                    args=[self.configs[name]],
                )
            )

        # Update layout with buttons
        self.fig.update_layout(
            width=1000,
            showlegend=True,
            minreducedheight=200,
            height=200,
            colorscale=go.layout.Colorscale(diverging="rdylgn"),
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "buttons": buttons,
                    "active": 1,
                    "pad": {"r": 0, "t": 0},
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 1.195,
                    "yanchor": "top",
                }
            ],
        )

    def show(self):
        self.prepare_figure()
        self.fig.show()

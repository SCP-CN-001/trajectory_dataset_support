##!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: argoverse.py
# @Description: Some visualization functions for Argoverse 2 Dataset.
# @Time: 2024/01/09
# @Author: Yueyuan Li

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "Dejavu Serif",
        "font.size": 10,
        "font.stretch": "semi-expanded",
    }
)

categories = [
    "vehicle",
    "bus",
    "motorcycle",
    "bicycle",
    "riderless_bicycle",
    "pedestrian",
    "static",
    "background",
    "construction",
    "unknown",
]


def plot_class_proportion(folder, sub_folders):
    df_proportion = pd.DataFrame(columns=categories, index=sub_folders)
    df_proportion.fillna(0, inplace=True)

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder, sub_folder)
        sub_sub_folders = os.listdir(sub_folder_path)
        for sub_sub_folder in sub_sub_folders:
            sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
            file_name = [
                file_name
                for file_name in os.listdir(sub_sub_folder_path)
                if ".parquet" in file_name
            ][0]
            file_path = os.path.join(sub_sub_folder_path, file_name)

            df_trajectory = pd.read_parquet(file_path, engine="fastparquet")
            for category in categories:
                df_proportion.loc[sub_folder, category] += len(
                    df_trajectory[df_trajectory["object_type"] == category]
                )

    print(df_proportion)
    df_proportion = df_proportion.div(df_proportion.sum(axis=1), axis=0)
    df_proportion.plot.barh(stacked=True, colormap="Set3", rot=1)
    plt.gcf().set_figwidth(6)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


def plot_trajectory(folder, city):
    locations = []

    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder, sub_folder)
        file_name = [
            file_name for file_name in os.listdir(sub_folder_path) if ".parquet" in file_name
        ][0]
        file_path = os.path.join(sub_folder_path, file_name)

        df_trajectory = pd.read_parquet(file_path, engine="fastparquet")
        if df_trajectory["city"][0] != city:
            continue

        for _, line in df_trajectory.iterrows():
            locations.append([line["position_x"], line["position_y"], line["object_type"]])

    df = pd.DataFrame(locations, columns=["x", "y", "class"])
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    sns.scatterplot(
        df,
        x="x",
        y="y",
        hue="class",
        s=0.05,
        palette="husl",
        hue_order=categories,
        legend=True,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", markerscale=1)
    plt.show()


def plot_speed_distribution(folder, city, map_range):
    sub_folders = os.listdir(folder)
    x_min, x_max, y_min, y_max = map_range
    matrix_x = int((x_max - x_min) / 10)
    matrix_y = int((y_max - y_min) / 10)
    speed_map = np.zeros([matrix_y, matrix_x, 2])

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder, sub_folder)
        file_name = [
            file_name for file_name in os.listdir(sub_folder_path) if ".parquet" in file_name
        ][0]
        file_path = os.path.join(sub_folder_path, file_name)

        df_trajectory = pd.read_parquet(file_path, engine="fastparquet")
        if df_trajectory["city"][0] != city:
            continue

        for _, line in df_trajectory.iterrows():
            if line["object_type"] != "vehicle":
                continue

            x = int((line["position_x"] - x_min) / 10)
            y = int((line["position_y"] - y_min) / 10)
            if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                continue

            speed_map[y, x, 0] += np.sqrt(line["velocity_x"] ** 2 + line["velocity_y"] ** 2)
            speed_map[y, x, 1] += 1

    speed_map[:, :, 0] /= speed_map[:, :, 1]
    speed_map = np.flip(speed_map, axis=0)

    im = plt.imshow(speed_map[:, :, 0], cmap="cool", vmin=0)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.gcf().set_figwidth(6)
    cax = plt.gcf().add_axes(
        [
            plt.gca().get_position().x1 + 0.01,
            plt.gca().get_position().y0,
            0.02,
            plt.gca().get_position().height,
        ]
    )
    plt.gca().set_title(city)
    plt.gca().set_aspect("equal")
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_mean_speed_distribution(folder):
    speeds = []

    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder, sub_folder)
        file_name = [
            file_name for file_name in os.listdir(sub_folder_path) if ".parquet" in file_name
        ][0]
        file_path = os.path.join(sub_folder_path, file_name)

        df_trajectory = pd.read_parquet(file_path, engine="fastparquet")
        city = df_trajectory["city"][0]

        df_trajectory = pd.read_parquet(file_path, engine="fastparquet")
        df_trajectory["speed"] = np.sqrt(
            df_trajectory["velocity_x"] ** 2 + df_trajectory["velocity_y"] ** 2
        )
        df_trajectory_group = df_trajectory.groupby("track_id")

        for _, group in df_trajectory_group:
            speeds.append([group["speed"].mean(), group["object_type"].iloc[0], city])

    df = pd.DataFrame(speeds, columns=["meanSpeed", "class", "city"])
    plot = sns.FacetGrid(
        df,
        col="city",
        col_wrap=2,
        sharex=False,
        sharey=True,
        height=2,
        aspect=1.5,
        hue="class",
        palette="husl",
        hue_order=categories,
    )
    plot.map(sns.histplot, "meanSpeed", stat="percent", element="step", kde=True)
    plot.add_legend()
    plt.show()

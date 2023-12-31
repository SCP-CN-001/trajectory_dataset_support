#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: waymo_motion.py
# @Description:
# @Time: 2023/12/01
# @Author: Yueyuan Li

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


from waymo_open_dataset.protos import scenario_pb2


mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "Dejavu Serif",
        "font.size": 10,
        "font.stretch": "semi-expanded",
    }
)

categories = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}

category_order = ["vehicle", "cyclist", "pedestrian", "other", "unset"]


def plot_trajectory(folder_name, trajectory_file_path_list, proportion=None):
    x = []
    y = []
    category_col = []

    if proportion is not None:
        trajectory_subset = np.random.choice(
            trajectory_file_path_list, int(len(trajectory_file_path_list) * proportion)
        )
    else:
        trajectory_subset = trajectory_file_path_list

    dataset = tf.data.TFRecordDataset(trajectory_subset, compression_type="")
    for data in dataset:
        proto_string = data.numpy()
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)
        for track in proto.tracks:
            category = categories[track.object_type]
            for state in track.states:
                x.append(state.center_x)
                y.append(state.center_y)
                category_col.append(category)

    fig, ax = plt.subplots()
    df = pd.DataFrame({"x": x, "y": y, "category": category_col})
    sns.scatterplot(
        df,
        x="x",
        y="y",
        hue="category",
        s=0.05,
        palette="husl",
        hue_order=category_order,
        legend=True,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_xlim([-25000, 25000])
    ax.set_ylim([-20000, 20000])
    plt.title(folder_name)
    plt.legend(markerscale=5)
    plt.show()


def plot_class_proportion(trajectory_folders, trajectory_file_path_lists):
    df_proportion = pd.DataFrame(0, columns=category_order, index=trajectory_folders)

    for i, trajectory_file_path in enumerate(trajectory_file_path_lists):
        dataset = tf.data.TFRecordDataset(trajectory_file_path, compression_type="")

        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            for track in proto.tracks:
                df_proportion.loc[trajectory_folders[i]][categories[track.object_type]] += 1

    print(df_proportion.head(6))
    df_proportion = df_proportion.div(df_proportion.sum(axis=1), axis=0)
    df_proportion.plot.barh(stacked=True, colormap="Set2", rot=1)
    plt.gcf().set_figwidth(6)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


def plot_mean_speed_distribution(trajectory_folders, trajectory_file_path_lists, proportion=None):
    speeds = []

    for i, trajectory_file_path in enumerate(trajectory_file_path_lists):
        if proportion is not None:
            trajectory_file_path_subset = np.random.choice(
                trajectory_file_path, int(len(trajectory_file_path) * proportion)
            )
        else:
            trajectory_file_path_subset = trajectory_file_path
        dataset = tf.data.TFRecordDataset(trajectory_file_path_subset, compression_type="")
        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            for track in proto.tracks:
                category = categories[track.object_type]
                speed = []
                for state in track.states:
                    speed.append(np.sqrt(state.velocity_x**2 + state.velocity_y**2))
                speeds.append([trajectory_folders[i], category, np.mean(speed)])

    df = pd.DataFrame(speeds, columns=["dataset_folder", "class", "meanSpeed"])
    plot = sns.FacetGrid(
        df,
        col="dataset_folder",
        col_wrap=2,
        sharex=False,
        sharey=False,
        hue="class",
        palette="husl",
        hue_order=category_order,
    )
    plot.map(sns.histplot, "meanSpeed", stat="percent", element="step", kde=True)
    plot.add_legend()
    plt.show()


def plot_delta_angle_distribution(trajectory_folders, trajectory_file_path_lists, proportion=None):
    delta_angles = []

    for i, trajectory_file_path in enumerate(trajectory_file_path_lists):
        if proportion is not None:
            trajectory_file_path_subset = np.random.choice(
                trajectory_file_path, int(len(trajectory_file_path) * proportion)
            )
        else:
            trajectory_file_path_subset = trajectory_file_path
        dataset = tf.data.TFRecordDataset(trajectory_file_path_subset, compression_type="")
        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            for track in proto.tracks:
                if track.object_type != 1:
                    continue
                last_heading = None
                for state in track.states:
                    if last_heading is not None:
                        delta_angle = (state.heading - last_heading) / 180 * np.pi
                        delta_angles.append([trajectory_folders[i], delta_angle])
                    last_heading = state.heading

    df = pd.DataFrame(delta_angles, columns=["dataset_folder", "deltaAngle"])
    plot = sns.FacetGrid(
        df, col="dataset_folder", col_wrap=2, sharex=True, sharey=False, palette="husl"
    )
    plot.map(sns.kdeplot, "deltaAngle", fill=True, alpha=0.5)
    plot.set(xlim=(-np.pi / 2, np.pi / 2))
    plt.show()

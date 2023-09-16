import os
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns


mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "FreeSerif",
        "font.size": 10,
        "font.stretch": "semi-expanded",
    }
)


def _get_plot_range(df: pd.DataFrame):
    x_min = df["x"].min()
    x_max = df["x"].max()
    y_min = df["y"].min()
    y_max = df["y"].max()

    x_min = x_min // 1 - 5
    x_max = x_max // 1 + 5
    y_min = y_min // 1 - 5
    y_max = y_max // 1 + 5

    return x_min, x_max, y_min, y_max


def plot_map_and_trajectories(map_name, data_path, img_path, transform, type_order, configs):
    map_img = mpimg.imread(os.path.join(img_path, configs[map_name]["name"] + ".png"))
    map_img = np.flipud(map_img)

    locations = list()

    for file_id in configs[map_name]["trajectory_files"]:
        vehicle_trajectory_path = os.path.join(
            data_path, map_name, "vehicle_tracks_%03d.csv" % file_id
        )
        df_trajectory = pd.read_csv(vehicle_trajectory_path, chunksize=100000)

        for chunk in df_trajectory:
            for _, line in chunk.iterrows():
                locations.append([line["x"], line["y"], line["agent_type"]])

    df = pd.DataFrame(locations, columns=["x", "y", "class"])
    x_min, x_max, y_min, y_max = _get_plot_range(df)

    fig, ax = plt.subplots()
    fig.set_figwidth(9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    im = ax.imshow(map_img, aspect="equal")
    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    sns.scatterplot(
        df,
        x="x",
        y="y",
        hue="class",
        s=0.05,
        palette="husl",
        hue_order=type_order,
        legend=True,
        ax=ax,
    )
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.show()


def plot_car_speed_distribution(map_name, map_range, data_path, configs: dict):
    x_min, x_max, y_min, y_max = map_range
    matrix_x = int((x_max - x_min) * 5)
    matrix_y = int((y_max - y_min) * 5)
    speed_map = np.zeros((matrix_y, matrix_x, 2))

    for file_id in configs[map_name]["trajectory_files"]:
        vehicle_trajectory_path = os.path.join(
            data_path, map_name, "vehicle_tracks_%03d.csv" % file_id
        )
        df_trajectory = pd.read_csv(vehicle_trajectory_path, chunksize=100000)

        for chunk in df_trajectory:
            for _, line in chunk.iterrows():
                x = int((line["x"] - x_min) * 5)
                y = int((line["y"] - y_min) * 5)
                if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                    continue
                speed_map[y, x, 0] += np.sqrt(line["vx"] ** 2 + line["vy"] ** 2)
                speed_map[y, x, 1] += 1

    speed_map[:, :, 0] /= speed_map[:, :, 1]
    speed_map = np.flip(speed_map, axis=0)

    im = plt.imshow(speed_map[:, :, 0], cmap="cool", vmin=0)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.gcf().set_figwidth(8)
    cax = plt.gcf().add_axes(
        [
            plt.gca().get_position().x1 + 0.01,
            plt.gca().get_position().y0,
            0.02,
            plt.gca().get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_log_angle_distribution(map_name, data_path, configs):
    bin_num = 72
    theta = np.linspace(0.0, 2 * np.pi, bin_num, endpoint=False)
    radii = np.zeros(bin_num)

    for file_id in configs[map_name]["trajectory_files"]:
        vehicle_trajectory_path = os.path.join(
            data_path, map_name, "vehicle_tracks_%03d.csv" % file_id
        )
        df_trajectory = pd.read_csv(vehicle_trajectory_path, chunksize=100000)

        for chunk in df_trajectory:
            for _, line in chunk.iterrows():
                angle = line["psi_rad"]
                if angle < 0:
                    angle += 2 * np.pi

                angle = angle * 180 / np.pi
                radii[int(np.floor(angle / 5))] += 1

        radii = [max(i, 1) for i in radii]

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, np.log10(radii), width=2 * np.pi / bin_num, bottom=4)
    for r, bar in zip(radii, bars):
        bar.set_facecolor("turquoise")
        bar.set_alpha(0.5)
    plt.gcf().set_size_inches(6, 6)
    plt.show()


if __name__ == "__main__":
    with open("../../map/map.config", "r") as f:
        configs = json.load(f)

    data_path = "../../trajectory/INTERACTION/recorded_trackfiles/"
    img_path = "../../img/INTERACTION/"
    trajectory_types = ["car", "bicycle", "pedestrian"]
    transform1 = mtransforms.Affine2D().scale(0.18).translate(1005, 945)
    plot_map_and_trajectories(
        "DR_CHN_Merging_ZS", data_path, img_path, transform1, trajectory_types, configs
    )

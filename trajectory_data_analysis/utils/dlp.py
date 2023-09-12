import os
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def _check_trajectory_types(data_path: str, configs: dict):
    type_set = set()
    for file_id in configs["DLP"]["trajectory_files"]:
        file_path = os.path.join(data_path, "DJI_%04d_agents.json" % file_id)
        with open(file_path, "r") as f:
            agent_data = json.load(f)

        for _, value in agent_data.items():
            type_set.add(value["type"])

    print(type_set)


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


def plot_map_and_trajectories(data_path: str, img_path: str, transform: mpl.transforms.Affine2D, type_order: list, configs: dict):
    map_img = mpimg.imread(img_path)
    map_img = np.flipud(map_img)

    location_list = list()

    for file_id in configs["DLP"]["trajectory_files"]:
        agent_path = os.path.join(data_path, "DJI_%04d_agents.json" % file_id)
        with open(agent_path, "r") as f_agent:
            agent_data = json.load(f_agent)
        instance_path = os.path.join(data_path, "DJI_%04d_instances.json" % file_id)
        with open(instance_path, "r") as f_instance:
            instance_data = json.load(f_instance)

        for  value in instance_data.values():
            type_ = agent_data[value["agent_token"]]["type"]
            location_list.append([value["coords"][0], value["coords"][1], type_])

    df = pd.DataFrame(location_list, columns=["x", "y", "class"])
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
        s=0.01,
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


def plot_class_proportion(data_path: str, type_order: list, configs: dict):
    df_proportion = pd.DataFrame(0, columns=type_order, index=configs["DLP"]["trajectory_files"])
    for file_id in configs["DLP"]["trajectory_files"]:
        agent_path = os.path.join(data_path, "DJI_%04d_agents.json" % file_id)
        with open(agent_path, "r") as f_agent:
            agent_data = json.load(f_agent)
        for _, value in agent_data.items():
            df_proportion.loc[file_id, value["type"]] += 1

        obstacle_path = os.path.join(data_path, "DJI_%04d_obstacles.json" % file_id)
        with open(obstacle_path, "r") as f_obstacle:
            obstacle_data = json.load(f_obstacle)
        for _, value in obstacle_data.items():
            df_proportion.loc[file_id, value["type"]] += 1

    df_proportion = df_proportion.div(df_proportion.sum(axis=1), axis=0)

    df_proportion.plot.barh(stacked=True, title="DLP", colormap="Set2", rot=1)
    plt.gcf().set_size_inches(8, 0.35 * len(df_proportion))
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


def plot_speed_distribution(map_range: list, data_path: str, type: str, configs: dict):
    x_min, x_max, y_min, y_max = map_range
    matrix_x = int((x_max - x_min) * 5)
    matrix_y = int((y_max - y_min) * 5)
    speed_map = np.zeros((matrix_y, matrix_x, 2))

    for file_id in configs["DLP"]["trajectory_files"]:
        agent_path = os.path.join(data_path, "DJI_%04d_agents.json" % file_id)
        with open(agent_path, "r") as f_agent:
            agent_data = json.load(f_agent)

        instance_path = os.path.join(data_path, "DJI_%04d_instances.json" % file_id)
        with open(instance_path, "r") as f_instance:
            instance_data = json.load(f_instance)

        obstacle_path = os.path.join(data_path, "DJI_%04d_obstacles.json" % file_id)
        with open(obstacle_path, "r") as f_obstacle:
            obstacle_data = json.load(f_obstacle)

        frame_path = os.path.join(data_path, "DJI_%04d_frames.json" % file_id)
        with open(frame_path, "r") as f_frame:
            frame_data = json.load(f_frame)
        time_steps = len(frame_data)

        for value in instance_data.values():
            if agent_data[value["agent_token"]]["type"] != type:
                continue

            x = int((value["coords"][0] - x_min) * 5)
            y = int((value["coords"][1] - y_min) * 5)
            if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                continue
            speed_map[y, x, 0] += value["speed"]
            speed_map[y, x, 1] += 1

        for value in obstacle_data.values():
            if value["type"] != type:
                continue
            x = int((value["coords"][0] - x_min) * 5)
            y = int((value["coords"][1] - y_min) * 5)
            if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                continue
            speed_map[y, x, 1] += time_steps
    
    speed_map[:, :, 0] /= speed_map[:, :, 1]
    speed_map = np.flip(speed_map, axis=0)

    im = plt.imshow(speed_map[:, :, 0], cmap="cool", vmin=0, vmax=11)
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


def plot_parking_density(map_range: list, data_path: str, configs: dict):
    x_min, x_max, y_min, y_max = map_range
    matrix_x = int(x_max - x_min)
    matrix_y = int(y_max - y_min)
    density_map = np.zeros((matrix_y, matrix_x, 2))

    for file_id in configs["DLP"]["trajectory_files"]:
        obstacle_path = os.path.join(data_path, "DJI_%04d_obstacles.json" % file_id)
        with open(obstacle_path, "r") as f_obstacle:
            obstacle_data = json.load(f_obstacle)

        for value in obstacle_data.values():
            x = int(value["coords"][0] - x_min)
            y = int(value["coords"][1] - y_min)
            if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                continue
            density_map[y, x, 0] += 1
            density_map[y, x, 1] = 1

    density_map[:, :, 0] /= density_map[:, :, 1]
    density_map = np.flip(density_map, axis=0)

    im = plt.imshow(density_map[:, :, 0], cmap="cool", vmin=0)
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


def plot_log_angle_distribution(data_path: str, type: str, configs: dict):
    bin_num = 72
    theta = np.linspace(0.0, 2 * np.pi, bin_num, endpoint=False)
    radii = np.zeros(bin_num)

    for file_id in configs["DLP"]["trajectory_files"]:
        agent_path = os.path.join(data_path, "DJI_%04d_agents.json" % file_id)
        with open(agent_path, "r") as f_agent:
            agent_data = json.load(f_agent)

        instance_path = os.path.join(data_path, "DJI_%04d_instances.json" % file_id)
        with open(instance_path, "r") as f_instance:
            instance_data = json.load(f_instance)

        for value in instance_data.values():
            if agent_data[value["agent_token"]]["type"] != type:
                continue
            
            angle = value["heading"] * 180 / np.pi
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

    data_path = "../../trajectory/DLP"
    _check_trajectory_types("DLP", data_path, configs)

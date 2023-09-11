import os

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


column_name = {
    "highD": {"x": "x", "y": "y", "id": "id"},
    "inD": {"x": "xCenter", "y": "yCenter", "vx": "xVelocity", "vy": "yVelocity", "id": "trackId"},
    "rounD": {
        "x": "xCenter",
        "y": "yCenter",
        "vx": "xVelocity",
        "vy": "yVelocity",
        "id": "trackId",
    },
    "exiD": {"x": "xCenter", "y": "yCenter", "vx": "xVelocity", "vy": "yVelocity", "id": "trackId"},
    "uniD": {"x": "xCenter", "y": "yCenter", "vx": "xVelocity", "vy": "yVelocity", "id": "trackId"},
}


def check_file_ids(data_path: str, id_range: tuple):
    result = dict()
    for i in range(*id_range):
        file_path = os.path.join(data_path, "%02d_recordingMeta.csv" % i)
        df = pd.read_csv(file_path)
        location = df.loc[0, "locationId"]
        if location not in result:
            result[location] = list()
        result[location].append(i)
    print(result)


def _merge_dictionary(dict1: dict, dict2: dict):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        else:
            if isinstance(value, dict):
                _merge_dictionary(dict1[key], value)
            else:
                dict1[key] += value
    return dict1


def _get_plot_range(df: pd.DataFrame):
    x_min = df["x"].min()
    x_max = df["x"].max()
    y_min = df["y"].min()
    y_max = df["y"].max()

    x_min = x_min // 1
    x_max = x_max // 1 + 1
    y_min = y_min // 1 - 3
    y_max = y_max // 1 + 4

    return x_min, x_max, y_min, y_max


def plot_map_and_trajectories(
    map_name: str,
    data_path: str,
    img_path: str,
    transform: mpl.transforms.Affine2D,
    type_order: list,
    configs: dict,
):
    dataset = configs[map_name]["dataset"]
    map_img = mpimg.imread(os.path.join(img_path, map_name + ".png"))
    if dataset != "highD":
        map_img = np.flipud(map_img)

    file_id = configs[map_name]["trajectory_files"][0]
    trajectory_path = os.path.join(data_path, "%02d_tracks.csv" % file_id)
    df_trajectory = pd.read_csv(trajectory_path, chunksize=100000)
    trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
    df_metadata = pd.read_csv(trajectory_metadata_path)

    trajectory_types = dict()
    for _, line in df_metadata.iterrows():
        trajectory_types[int(line[column_name[dataset]["id"]])] = line["class"]

    locations = list()
    for chunk in df_trajectory:
        for _, line in chunk.iterrows():
            locations.append(
                [
                    line[column_name[dataset]["x"]],
                    line[column_name[dataset]["y"]],
                    trajectory_types[int(line[column_name[dataset]["id"]])],
                ]
            )

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


def plot_class_proportion(map_name: str, data_path: str, type_order: list, configs: dict):
    df_proportion = pd.DataFrame(columns=type_order, index=configs[map_name]["trajectory_files"])

    for file_id in configs[map_name]["trajectory_files"]:
        trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
        df_metadata = pd.read_csv(trajectory_metadata_path)
        for type_ in type_order:
            df_proportion.loc[file_id, type_] = len(
                df_metadata[df_metadata["class"] == type_]
            ) / len(df_metadata)

    df_proportion.plot.barh(stacked=True, title=map_name, colormap="Set2", rot=1)
    plt.gcf().set_size_inches(8, 0.35 * len(df_proportion))
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


def plot_mean_speed_distribution(dataset: str, data_path: str, type_order: list, configs: dict):
    speeds = list()
    for key, value in configs.items():
        if not dataset in key:
            continue

        for file_id in value["trajectory_files"]:
            trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
            df_meta = pd.read_csv(trajectory_metadata_path)

            if "meanXVelocity" in df_meta.columns:
                for _, line in df_meta.iterrows():
                    speeds.append([key, line["class"], line["meanXVelocity"]])
            else:
                trajectory_path = os.path.join(data_path, "%02d_tracks.csv" % file_id)
                df_trajectory = pd.read_csv(trajectory_path, chunksize=100000)
                speed_list = list()
                id_list = list()

                for chunk in df_trajectory:
                    vx = chunk[column_name[dataset]["vx"]]
                    vy = chunk[column_name[dataset]["vy"]]
                    speed_list += list(np.sqrt(vx**2 + vy**2))
                    id_list += list(chunk[column_name[dataset]["id"]])

                for id_ in set(id_list):
                    mean_speed = np.mean(
                        [speed_list[i] for i in range(len(id_list)) if id_list[i] == id_]
                    )
                    class_ = df_meta[df_meta[column_name[dataset]["id"]] == id_].iloc[0]["class"]
                    speeds.append([key, class_, mean_speed])

    df = pd.DataFrame(speeds, columns=["map", "class", "meanSpeed"])
    plot = sns.FacetGrid(
        df,
        col="map",
        col_wrap=2,
        sharex=False,
        sharey=True,
        height=2,
        aspect=1.5,
        hue="class",
        palette="husl",
        hue_order=type_order,
    )
    plot.map(sns.histplot, "meanSpeed", stat="percent", element="step", kde=True)
    plot.add_legend()
    plt.show()


def plot_speed_distribution(map_name: str, map_range: list, data_path, type: str, configs: dict):
    dataset = configs[map_name]["dataset"]
    x_min, x_max, y_min, y_max = map_range
    matrix_x = int((x_max - x_min) * 10)
    matrix_y = int((y_max - y_min) * 10)
    speed_map = np.zeros([matrix_y, matrix_x, 2])

    for file_id in configs[map_name]["trajectory_files"]:
        trajectory_path = os.path.join(data_path, "%02d_tracks.csv" % file_id)
        df_trajectory = pd.read_csv(trajectory_path, chunksize=100000)
        trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
        df_meta = pd.read_csv(trajectory_metadata_path)

        type_dict = dict()
        for _, line in df_meta.iterrows():
            type_dict[int(line[column_name[dataset]["id"]])] = line["class"]

        for chunk in df_trajectory:
            for _, line in chunk.iterrows():
                if type_dict[int(line[column_name[dataset]["id"]])] != type:
                    continue

                x = int((line[column_name[dataset]["x"]] - x_min) * 10)
                y = int((line[column_name[dataset]["y"]] - y_min) * 10)
                if x < 0 or x >= matrix_x or y < 0 or y >= matrix_y:
                    continue
                speed_map[y, x, 0] += np.sqrt(
                    line[column_name[dataset]["vx"]] ** 2 + line[column_name[dataset]["vy"]] ** 2
                )
                speed_map[y, x, 1] += 1

    speed_map[:, :, 0] /= speed_map[:, :, 1]

    if x_max < 0:
        speed_map = np.flip(speed_map, axis=1)
    if y_max < 0:
        speed_map = np.flip(speed_map, axis=0)

    plt.imshow(speed_map[:, :, 0], cmap="cool", vmin=0)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.gcf().set_figwidth(8)
    plt.colorbar()
    plt.show()


def plot_log_angle_distribution(map_name: str, data_path: str, type: str, configs: dict):
    dataset = configs[map_name]["dataset"]
    bin_num = 72
    theta = np.linspace(0.0, 2 * np.pi, bin_num, endpoint=False)
    radii = np.zeros(bin_num)

    for file_id in configs[map_name]["trajectory_files"]:
        trajectory_path = os.path.join(data_path, "%02d_tracks.csv" % file_id)
        df_trajectory = pd.read_csv(trajectory_path, chunksize=100000)
        trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
        df_meta = pd.read_csv(trajectory_metadata_path)

        type_dict = dict()
        for _, line in df_meta.iterrows():
            type_dict[int(line[column_name[dataset]["id"]])] = line["class"]

        for chunk in df_trajectory:
            for _, line in chunk.iterrows():
                if type_dict[int(line[column_name[dataset]["id"]])] != type:
                    continue

                radii[int(np.floor(line["heading"] / (360 / bin_num)) % bin_num)] += 1

    radii = [max(i, 1) for i in radii]

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, np.log10(radii), width=2 * np.pi / bin_num, bottom=4)
    for r, bar in zip(radii, bars):
        bar.set_facecolor("turquoise")
        bar.set_alpha(0.5)
    plt.gcf().set_size_inches(6, 6)
    plt.gca().set_title(map_name)
    plt.show()


def plot_delta_angle_distribution(dataset: str, data_path: str, type: str, configs: dict):
    delta_angles = list()
    last_angle = None
    last_id = None
    for key, value in configs.items():
        if not dataset in key:
            continue

        for file_id in value["trajectory_files"]:
            trajectory_path = os.path.join(data_path, "%02d_tracks.csv" % file_id)
            df_trajectory = pd.read_csv(trajectory_path, chunksize=100000)
            trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
            df_meta = pd.read_csv(trajectory_metadata_path)

            type_dict = dict()
            for _, line in df_meta.iterrows():
                type_dict[int(line[column_name[dataset]["id"]])] = line["class"]

            for chunk in df_trajectory:
                for _, line in chunk.iterrows():
                    if type_dict[int(line[column_name[dataset]["id"]])] != type:
                        continue

                    if last_id is None or last_id != int(line[column_name[dataset]["id"]]):
                        last_id = int(line[column_name[dataset]["id"]])
                        last_angle = line["heading"] * (np.pi / 180)
                        continue
                    else:
                        current_angle = line["heading"] * (np.pi / 180)
                        delta_angles.append(
                            [key, current_angle - last_angle, current_angle, last_angle]
                        )
                        last_angle = current_angle

    df = pd.DataFrame(delta_angles, columns=["map", "deltaAngle", "current", "last"])
    plot = sns.FacetGrid(
        df,
        col="map",
        col_wrap=2,
        sharex=True,
        sharey=False,
        height=2,
        aspect=1.5,
        palette="husl",
    )
    plot.map(sns.kdeplot, "deltaAngle", fill=True, alpha=0.5)
    plot.set(xlim=(-np.pi / 2, np.pi / 2))
    plt.show()


def _count_lane_change(trajectory_file_path: str):
    df = pd.read_csv(trajectory_file_path)
    count = dict()
    for _, line in df.iterrows():
        if line["class"] not in count:
            count[line["class"]] = dict()

        if line["numLaneChanges"] not in count[line["class"]]:
            count[line["class"]][line["numLaneChanges"]] = 0

        count[line["class"]][line["numLaneChanges"]] += 1

    return count


def plot_lane_change_distribution(dataset: str, data_path: str, type_order: list, configs: dict):
    maps = dict()
    files = dict()
    for key, value in configs.items():
        if dataset in key:
            maps[key] = dict()
            for file_id in value["trajectory_files"]:
                files[file_id] = key

    for file_id, map_key in files.items():
        trajectory_metadata_path = os.path.join(data_path, "%02d_tracksMeta.csv" % file_id)
        lane_change_distribution = _count_lane_change(trajectory_metadata_path)
        if map_key not in maps:
            maps[map_key] = lane_change_distribution
        else:
            _merge_dictionary(maps[map_key], lane_change_distribution)

    df = pd.DataFrame(columns=["map", "class", "numLaneChanges", "count"])
    for map_key, lane_change_dist in maps.items():
        for class_, lane_change_count in lane_change_dist.items():
            for num_lane_changes, count in lane_change_count.items():
                df.loc[len(df.index)] = [
                    map_key,
                    class_,
                    int(num_lane_changes),
                    int(count),
                ]

    plot = sns.FacetGrid(
        df,
        col="map",
        col_wrap=3,
        sharex=True,
        sharey=False,
        hue="class",
        palette="husl",
        hue_order=type_order,
    )
    plot.map(sns.lineplot, "numLaneChanges", "count", markers=True)
    plot.add_legend()
    plt.show()


if __name__ == "__main__":
    check_file_ids("../../trajectory/uniD/data", (0, 13))

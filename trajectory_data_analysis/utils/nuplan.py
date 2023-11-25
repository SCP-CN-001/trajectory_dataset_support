import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3

mpl.rcParams.update(
    {"figure.dpi": 300, "font.family": "serif", "font.size": 10, "font.stretch": "semi-expanded"}
)

categories = [
    "vehicle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]
dynamic_category = ["vehicle", "bicycle", "pedestrian", "generic_object"]


def plot_trajectories(data_path, trajectory_folder, trajectory_files):
    x = []
    y = []
    category_col = []

    for file_name in trajectory_files:
        file_path = data_path + trajectory_folder + "/" + file_name
        with sqlite3.connect(file_path) as motion_db:
            df_category = pd.read_sql_query("SELECT * FROM category;", motion_db, index_col="token")
            df_track = pd.read_sql_query("SELECT * FROM track;", motion_db, index_col="token")
            df_lidar_box = pd.read_sql_query("SELECT * FROM lidar_box;", motion_db)

            dict_category = dict(zip(df_category.index, df_category["name"]))
            dict_track = dict(zip(df_track.index, df_track["category_token"]))

            for row in df_lidar_box.iterrows():
                x.append(row[1]["x"])
                y.append(row[1]["y"])
                category_col.append(dict_category[dict_track[row[1]["track_token"]]])
        motion_db.close()

    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    df = pd.DataFrame({"x": x, "y": y, "category": category_col})
    sns.scatterplot(
        df,
        x="x",
        y="y",
        hue="category",
        s=0.05,
        palette="husl",
        hue_order=categories,
        legend=False,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_title(trajectory_folder)
    plt.show()


def plot_class_proportion(data_path, trajectory_folders, trajectory_files):
    df_proportion = pd.DataFrame(columns=dynamic_category, index=trajectory_folders)

    for i, trajectory_folder in enumerate(trajectory_folders):
        for file_name in trajectory_files[i]:
            file_path = data_path + trajectory_folder + "/" + file_name
            with sqlite3.connect(file_path) as motion_db:
                df_category = pd.read_sql_query(
                    "SELECT * FROM category;", motion_db, index_col="token"
                )
                df_track = pd.read_sql_query("SELECT * FROM track;", motion_db, index_col="token")

                dict_category = dict(zip(df_category["name"], df_category.index))

                for category in dynamic_category:
                    cnt = len(df_track[df_track["category_token"] == dict_category[category]])
                    df_proportion.loc[trajectory_folder][category] = cnt

            motion_db.close()

    for row in df_proportion.iterrows():
        print(
            "In %s, there are %d vehicles, %d bicycles, %d pedestrians, %d generic objects."
            % (
                row[0],
                row[1]["vehicle"],
                row[1]["bicycle"],
                row[1]["pedestrian"],
                row[1]["generic_object"],
            )
        )

    df_proportion = df_proportion.div(df_proportion.sum(axis=1), axis=0)
    df_proportion.plot.barh(stacked=True, colormap="Set2", rot=1)
    plt.gcf().set_figwidth(6)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")


def plot_mean_speed_distribution(data_path, trajectory_folders, trajectory_files):
    speeds = []

    for i, trajectory_folder in enumerate(trajectory_folders):
        for file_name in trajectory_files[i]:
            file_path = data_path + trajectory_folder + "/" + file_name
            with sqlite3.connect(file_path) as motion_db:
                df_category = pd.read_sql_query(
                    "SELECT * FROM category;", motion_db, index_col="token"
                )
                df_track = pd.read_sql_query("SELECT * FROM track;", motion_db, index_col="token")
                df_lidar_box = pd.read_sql_query("SELECT * FROM lidar_box;", motion_db)

                dict_category = dict(zip(df_category.index, df_category["name"]))
                dict_track = dict(zip(df_track.index, df_track["category_token"]))

                df_lidar_box["speed"] = np.sqrt(
                    df_lidar_box["vx"] ** 2 + df_lidar_box["vy"] ** 2 + df_lidar_box["vz"] ** 2
                )
                mean_speed = df_lidar_box.groupby("track_token")["speed"].mean()
                track_token = mean_speed.index
                for i, speed in enumerate(mean_speed):
                    speeds.append(
                        [trajectory_folder, dict_category[dict_track[track_token[i]]], speed]
                    )

            motion_db.close()

    df = pd.DataFrame(speeds, columns=["location", "class", "meanSpeed"])
    plot = sns.FacetGrid(
        df,
        col="location",
        col_wrap=2,
        sharex=False,
        sharey=False,
        hue="class",
        palette="husl",
        hue_order=dynamic_category,
    )
    plot.map(sns.histplot, "meanSpeed", stat="percent", element="step", kde=True)
    plot.add_legend()
    plt.show()


def plot_speed_distribution(map_boundary, data_path, trajectory_folder, trajectory_files):
    x_min, x_max, y_min, y_max = map_boundary
    matrix_x = int((x_max - x_min))
    matrix_y = int((y_max - y_min))
    speed_map = np.zeros((matrix_y, matrix_x, 2))

    for file_name in trajectory_files:
        file_path = data_path + trajectory_folder + "/" + file_name
        with sqlite3.connect(file_path) as motion_db:
            df_category = pd.read_sql_query("SELECT * FROM category;", motion_db, index_col="token")
            df_track = pd.read_sql_query("SELECT * FROM track;", motion_db, index_col="token")
            df_lidar_box = pd.read_sql_query("SELECT * FROM lidar_box;", motion_db)

            dict_category = dict(zip(df_category.index, df_category["name"]))
            dict_track = dict(zip(df_track.index, df_track["category_token"]))

            for row in df_lidar_box.iterrows():
                category = dict_category[dict_track[row[1]["track_token"]]]
                if category == "vehicle":
                    x = int((row[1]["x"] - x_min))
                    y = int((row[1]["y"] - y_min))
                    if x < 0 and x >= matrix_x and y < 0 and y >= matrix_y:
                        continue

                    speed_map[y, x, 0] += np.sqrt(
                        row[1]["vx"] ** 2 + row[1]["vy"] ** 2 + row[1]["vz"] ** 2
                    )
                    speed_map[y, x, 1] += 1
        motion_db.close()

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
    plt.colorbar(im, cax=cax)
    plt.show()

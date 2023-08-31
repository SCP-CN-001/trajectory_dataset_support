import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
import time
import seaborn as sns


def merge_dictionary(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        else:
            if isinstance(value, dict):
                merge_dictionary(dict1[key], value)
            else:
                dict1[key] += value
    return dict1


def plot_map_and_trajectories(trajectory_data_path, config):
    map_path = config["osm_path"]


def count_lane_change(trajectory_file_path):
    df = pd.read_csv(trajectory_file_path)
    count = dict()
    for _, line in df.iterrows():
        if line["class"] not in count:
            count[line["class"]] = dict()

        if line["numLaneChanges"] not in count[line["class"]]:
            count[line["class"]][line["numLaneChanges"]] = 0

        count[line["class"]][line["numLaneChanges"]] += 1

    return count


def plot_lane_change_distribution(dataset, data_path, configs):
    maps = dict()
    files = dict()
    pool = ThreadPool(1)

    for key, value in configs.items():
        if dataset in key:
            maps[key] = dict()
            for file_id in value["trajectory_files"]:
                files[file_id] = key

    def process(item):
        file_id = item
        trajectory_metadata_path = os.path.join(
            data_path, "%02d_tracksMeta.csv" % file_id
        )
        lane_change_distribution = count_lane_change(trajectory_metadata_path)
        return file_id, lane_change_distribution

    lane_changes = pool.map(process, files.keys())
    for file_id, lane_change_dist in lane_changes:
        map_key = files[file_id]
        if map_key not in maps:
            maps[map_key] = lane_change_dist
        else:
            merge_dictionary(maps[map_key], lane_change_dist)

    df = pd.DataFrame(columns=["map", "class", "numLaneChanges", "count"])
    for map_key, lane_change_dist in maps.items():
        for class_, lane_change_count in lane_change_dist.items():
            for num_lane_changes, count in lane_change_count.items():
                df.loc[len(df.index)] = [map_key, class_, int(num_lane_changes), int(count)]

    plot = sns.FacetGrid(df, col="map", col_wrap=3, sharey=False, sharex=True)
    plot.map(sns.lineplot, "numLaneChanges", "count", "class")
    plt.show()


if __name__ == "__main__":
    with open("../../map/map.config", "r") as f:
        configs = json.load(f)
    t1 = time.time()
    plot_lane_change_distribution("highD", "../../trajectory/highD/data", configs)
    t2 = time.time()
    print(t2 - t1)

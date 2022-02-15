import math
import os
import json
import pandas as pd
import numpy as np


class highD_map:
    def __init__(self, key, value, input_path):
        self.map_name = key
        self.file_ids = value["ids"]
        self.lanes = value["roads"]
        self.special = (key == "highD_6")
        self.range = value["range"]
        self.input_path = input_path
        if self.special:
            self.node = value["node"]
        self.suffix = ["_recordingMeta.csv", "_tracks.csv", "_tracksMeta.csv"]
        self.traci_init_order = ["id", "length", "width", "vtype", "route", "initialFrame", "finalFrame", "departLane", "departPos", "departSpeed", "arrivalLane", "arrivalPos", "arrivalSpeed"]
        self.traci_order = ["frame", "id", "x", "y", "speed", "angle", "edge", "lane"]
    
    def __get_route(self, df_track_meta):
        '''
        Get the route of a vehicle. This value is needed to initialize a vehicle by traci. 
        Handle "highD_6" as a special case
        '''
        routes = np.round(df_track_meta["drivingDirection"].tolist()).astype("str")
        return routes
    
    def __get_speed(self, df):
        '''
        Get the absolute speed value of the vehicle. This value is needed to update the vehicle state by traci.
        '''
        vx = np.array(df["xVelocity"])
        vy = np.array(df["yVelocity"])
        return np.round(np.sqrt(vx**2 + vy**2),2)
    
    def __get_angle(self, df, df_track_meta):
        '''
        Get the heading direction of the vehicle. This value is needed to update the vehicle state by traci.
        The angle value is assumed to be in navigational degrees between 0 and 360 with 0 at the top, 
        going clockwise
        '''
        angles = []
        for _, line in df.iterrows():
            # initialize
            direction = df_track_meta[df_track_meta["id"]==line["id"]].iloc[0]["drivingDirection"]
            angle = 270
            if direction == 2:
                angle = 90
            vx = line["xVelocity"]
            vy = line["yVelocity"]
            
            if vx == 0:
                if vy > 0:
                    angle = 0
                else:
                    angle = 180
            elif (vx < 0) & (vy > 0):
                angle = 450-math.atan2(vy,vx)/math.pi*180
            else:
                angle = 90-math.atan2(vy,vx)/math.pi*180
            angles.append(round(angle,2))
        return angles
        
    
    def __get_calibrated(self, df, df_meta):
        '''
        Centralization and calibration. 
        These values are needed to initialize and update the vehicle state by traci.
        In highD dataset, the origin of the coordinate system is upper-left. The x, y coordinates 
        record the upper-left point of the vehicles' bounding box.
        In SUMO, the global coordinate system originate from lower-left. The x, y coordinates 
        record the center of the vehicle. 
        What's more, the data files for the same maps have subtle difference at the road boundary 
        values, which was supposed to be the baseline reference for the whole data file. This 
        function will calibrate the road boundaries to keep consistency.
        '''
        # get the location of the central of a vehicle in the lower-left originated coordinate system
        x_center = np.array(df["x"]) + np.array(df["width"])/2
        y_center = 50 - (np.array(df["y"]) + np.array(df["height"])/2)
        
        # calibrate the location of road
        upper =df_meta.iloc[0]["upperLaneMarkings"].split(";")
        upper = 50 - float(upper[0])
        lower = df_meta.iloc[0]["lowerLaneMarkings"].split(";")
        lower = 50 - float(lower[-1])
        k = sum(self.range)/2 - (upper+lower)/2
        y_center += k
        return np.round(x_center,2), np.round(y_center, 2)
    
    def __get_location(self, df_traci, df):
        '''
        Get the edge and lane ids of vehicles. These values are needed to update the vehicle state 
        by traci.
        '''
        edges = []
        lanes = []
        for id, line in df_traci.iterrows():
            for lane_info in self.lanes.values():
                edge = None
                lane = None
                if (line["y"] >= lane_info["boundary"][0]) & (line["y"] <= lane_info["boundary"][1]):
                    edge = lane_info["edge"]
                    lane = lane_info["lane"]
                    break
            if edge is None:
                lane_id = str(int(df.iloc[id]["laneId"]))
                edge = self.lanes[lane_id]["edge"]
                lane = self.lanes[lane_id]["lane"]
            if self.special:
                if (line["x"] < self.node) & (edge[0]=="1") & (lane > 0):
                    edge = "1_r"
                    lane = lane-1
            edges.append(edge)
            lanes.append(lane)
        return edges, lanes
    
    def __get_endnode_info(self, df_init, df):
        '''
        Get "initialFrame", "finalFrame", "departLane", "departPos", "departSpeed", "arrivalLane", 
        "arrivalPos", "arrivalSpeed". These values are needed to initialize a vehicle.
        '''
        departLane = np.zeros(len(df_init))
        departSpeed = np.zeros(len(df_init))
        departPos = np.zeros(len(df_init))
        arrivalLane = np.zeros(len(df_init))
        arrivalSpeed = np.zeros(len(df_init))
        arrivalPos = np.zeros(len(df_init))
        initialFrame = np.zeros(len(df_init))
        finalFrame = np.zeros(len(df_init))
        for id, line in df_init.iterrows():
            df_sub = df[df["id"] == line["id"]]
            initialFrame[id] = df_sub.iloc[0]["frame"]
            finalFrame[id] = df_sub.iloc[-1]["frame"]
            departLane[id] = df_sub.iloc[0]["lane"]
            departSpeed[id] = df_sub.iloc[0]["speed"]
            arrivalLane[id] = df_sub.iloc[-1]["lane"]
            arrivalSpeed[id] = df_sub.iloc[-1]["speed"]
            if int(line["direction"]) == 1:
                departPos[id] = 410 - df_sub.iloc[0]["x"]
                arrivalPos[id] = 410 - df_sub.iloc[-1]["x"]
            else:
                departPos[id] = df_sub.iloc[0]["x"]
                arrivalPos[id] = df_sub.iloc[-1]["x"]
        df_init["initialFrame"] = initialFrame
        df_init["finalFrame"] = finalFrame
        df_init["departLane"] = departLane
        df_init["departSpeed"] = departSpeed
        df_init["departPos"] = np.round(departPos, 2)
        df_init["arrivalLane"] = arrivalLane
        df_init["arrivalSpeed"] = arrivalSpeed
        df_init["arrivalPos"] = np.round(arrivalPos, 2)
        return df_init
    
    def process(self, output_path):
        for id in self.file_ids:
            # initial data frame for a single file
            print("Converting data in file %02d" % id)
            prefix = format(id, "02d")
            df_meta = pd.read_csv(os.path.join(self.input_path, prefix+self.suffix[0]))
            df = pd.read_csv(os.path.join(self.input_path, prefix+self.suffix[1]))
            df_track_meta = pd.read_csv(os.path.join(self.input_path, prefix+self.suffix[2]))
            
            df_traci = pd.DataFrame({
                "frame": df["frame"],
                "id": df["id"]
            })
            df_traci_init = pd.DataFrame({
                "id": df_track_meta["id"],
                "length": df_track_meta["width"],
                "width": df_track_meta["height"],
                "vtype": df_track_meta["class"],
                "direction": df_track_meta["drivingDirection"]
            })
            
            # prepare the data files for sumo-traci
            df_traci["speed"] = self.__get_speed(df)
            df_traci["angle"] = self.__get_angle(df, df_track_meta)
            df_traci["x"], df_traci["y"] = self.__get_calibrated(df, df_meta)
            df_traci["edge"], df_traci["lane"] = self.__get_location(df_traci, df)
            df_traci = df_traci[(df_traci["x"]>=0) & (df_traci["x"]<=410)]
            
            df_traci_init["route"] = self.__get_route(df_track_meta)
            df_traci_init = self.__get_endnode_info(df_traci_init, df_traci)
            
            # store the processed data
            df_traci = df_traci[self.traci_order]
            df_traci_init = df_traci_init[self.traci_init_order]
            df_traci.to_csv(os.path.join(output_path, prefix+self.suffix[1]), index=False)
            df_traci_init.to_csv(os.path.join(output_path, prefix+self.suffix[2]), index=False)


if __name__ == '__main__':
    folder_path = "/data/HighD/highD/data"
    config_path = "./highD.config"
    with open(config_path, "r") as f_config:
        config = json.load(f_config)
    
    output_path = "/data/HighD/highD/processed"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    for key, value in config.items():
        map = highD_map(key, value, folder_path)
        map.process(output_path)
    
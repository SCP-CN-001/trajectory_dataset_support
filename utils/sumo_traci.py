import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
import pandas as pd
import traci
import traci.constants as tc

class Vehicles:
    def __init__(self, df, df_init) :
        self.df = df
        self.df_init = df_init
        self.vehicles = set()
    
    def __remove_vehicle(self, step):
        vehicle_to_remove = self.df_init[self.df_init["finalFrame"]==step-1]["id"].tolist()
        vehicle_existing = traci.vehicle.getIDList()
        for id in vehicle_to_remove:
            if id in vehicle_existing:
                traci.vehicle.remove(vehID="%d" % id, reason=tc.REMOVE_ARRIVED)
            self.vehicles.remove(id)
    
    def __add_vehicle(self, step):
        df_sub = self.df_init[self.df_init["initialFrame"]==step]
        for _, line in df_sub.iterrows():
            veh_id = "%d" % line["id"]
            traci.vehicle.add(
                vehID=veh_id,
                routeID=line["route"],
                typeID=line["vtype"],
                departLane=line["departLane"],
                departPos=line["departPos"],
                departSpeed=line["departSpeed"],
                arrivalLane=line["arrivalLane"],
                arrivalPos=line["arrivalPos"]
                # arrivalSpeed=line["arrivalSpeed"]
            )
            traci.vehicle.setLength(vehID=veh_id, length=line["length"])
            traci.vehicle.setWidth(vehID=veh_id, width=line["width"])
            self.vehicles.add(line["id"])
            
    def __update_state(self, step):
        df_sub = self.df[self.df["frame"] == step]
        for id in self.vehicles:
            veh_id = "%d" % id
            veh_info = df_sub[df_sub["id"]==id]
            traci.vehicle.slowDown(
                vehID=veh_id,
                speed=veh_info.iloc[0]["speed"],
                duration=0.04
            )
            traci.vehicle.moveToXY(
                vehID=veh_id,
                edgeID=veh_info.iloc[0]["edge"],
                lane=veh_info.iloc[0]["lane"],
                x=veh_info.iloc[0]["x"],
                y=veh_info.iloc[0]["y"],
                angle=veh_info.iloc[0]["angle"]
            )
    
    def update(self, step):
        self.__remove_vehicle(step)
        self.__add_vehicle(step)
        self.__update_state(step)
    
if __name__ == '__main__':
    sumo_cmd = ["sumo-gui", "-c", "highD.sumocfg"]
    df = pd.read_csv(
        "/data/HighD/highD/processed/12_tracks.csv"
        )
    df_init = pd.read_csv(
        "/data/HighD/highD/processed/12_tracksMeta.csv"
    )
    df["edge"] = df["edge"].astype(str)
    df_init["route"] = df_init["route"].astype(str)
    df_init["departLane"] = df_init["departLane"].astype(int)
    df_init["arrivalLane"] = df_init["arrivalLane"].astype(int)
    
    traci.start(sumo_cmd)
    step = 1
    vehicles = Vehicles(df, df_init)
    while step < 10000:
        vehicles.update(step)
        traci.simulationStep()
        step += 1
    traci.close()
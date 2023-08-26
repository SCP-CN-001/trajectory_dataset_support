# Trajectory Data Support

## About

This repository provides **third-party** map data and script-based support for the open trajectory datasets, including NGSIM, LevelXdata (highD, inD, rounD, uniD, and exiD), Interaction, and Dragon Lake Park (DLP).

Due to the license of the open datasets, this repository cannot distribute the trajectory data in any form. Please contact the authors to apply for the download links.

## Dataset Overview

| Dataset | Scenario Type | # Map | # Trajectory | Duration (h) | Frequency (Hz) |
| --- | --- | --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) | | 2 | 9206 |  1.5 | 10 |
| [highD](https://www.highd-dataset.com/) | Highway | 6 | 110000 | 16.5 | 25 |
| [inD](https://www.ind-dataset.com/) | Intersection | 4 | 11500 | 10 | 25 |
| [rounD](https://www.round-dataset.com/) | Roundabout | 3 | 13746 | 6 | 25 |
| [uniD](https://www.unid-dataset.com/) | Intersection | 1 | |
| [exiD](https://www.exid-dataset.com/) | Highway | 7 | 69172 | 16 | 25 | 
| [INTERACTION](http://interaction-dataset.com/) | Merging, roundabout, intersection | 10 | 10450 | 16.5 | 10 |
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | Parking | 1 | 5188 | 3.5 | 25 |

## Progress

This repository provides SUMO-style road network, lanelet2 style maps, trajectory data insight of the datasets. Here is the current progress of the construction.

| Dataset | SUMO Road Network | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) Map | Data Insight |
| --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) | | | $\surd$ ([official](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj))
| [highD](https://www.highd-dataset.com/) | $\surd$ | $\surd$ | $\surd$ |
| [inD](https://www.ind-dataset.com/) | $\surd$ | $\surd$ | $\surd$ |
| [rounD](https://www.round-dataset.com/) | $\surd$ | $\surd$ | $\surd$ |
| [uniD](https://www.unid-dataset.com/) | |||
| [exiD](https://www.exid-dataset.com/) ||||
| [INTERACTION](http://interaction-dataset.com/) | $\surd$ | ||
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | | ||

## Citation

```latex
@article{dot2018next,
  title={Next generation simulation (NGSIM) vehicle trajectories and supporting data},
  author={Dot, U},
  journal={US Department of Transportation},
  year={2018}
}

@inproceedings{highDdataset,
    title={The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems},
    author={Krajewski, Robert and Bock, Julian and Kloeker, Laurent and Eckstein, Lutz},
    booktitle={2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
    pages={2118-2125},
    year={2018},
    doi={10.1109/ITSC.2018.8569552}
}

@inproceedings{inDdataset,
    title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
    author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
    booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
    pages={1929-1934},
    year={2020},
    doi={10.1109/IV47402.2020.9304839}
}

@inproceedings{rounDdataset,
    title={The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany},
    author={Krajewski, Robert and Moers, Tobias and Bock, Julian and Vater, Lennart and Eckstein, Lutz},
    booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)},
    pages={1-6},
    year={2020},
    doi={10.1109/ITSC45102.2020.9294728}
}

@inproceedings{moers2022exid,
    title={The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany},
    author={Moers, Tobias and Vater, Lennart and Krajewski, Robert and Bock, Julian and Zlocki, Adrian and Eckstein, Lutz},
    booktitle={2022 IEEE Intelligent Vehicles Symposium (IV)},
    pages={958--964},
    year={2022},
    organization={IEEE}
}

@INPROCEEDINGS{9922162,
    author={Shen, Xu and Lacayo, Matthew and Guggilla, Nidhir and Borrelli, Francesco},
    booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)}, 
    title={ParkPredict+: Multimodal Intent and Motion Prediction for Vehicles in Parking Lots with CNN and Transformer}, 
    year={2022},
    volume={},
    number={},
    pages={3999-4004},
    doi={10.1109/ITSC55140.2022.9922162}}
```

# Trajectory Data Support

[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

## About

This repository provides **third-party** map data and script-based support for the open trajectory datasets, including NGSIM, LevelXdata (highD, inD, rounD, uniD, and exiD), Interaction, and Dragon Lake Park (DLP).

Due to the license of the open datasets, this repository cannot distribute the trajectory data in any form. Please contact the dataset owners to apply for the download links.

## Folder Structure

```shell
.
├── img
│   # the map images
├── map
│   # the maps, including road network format and lanelet2 format
├── trajectory_sample
│   # the trajectory data samples.
│   # Please contact the dataset owners for the complete dataset!
├── trajectory_data_analysis
│   # the official and personal analysis of the trajectory data
└── utils
```

## Dataset Overview

Here are some basic information about the open trajectory datasets.

| Dataset | Publish | Scenario | # Map | # Trajectory | Duration (h) | Frequency (Hz) |
| --- | --- | --- | --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) |  | Highway, </br> intersection | 2 | 9206 | 2.5 | 10 |
| [highD](https://www.highd-dataset.com/) | 2018 |  Highway | 6 | 110000 | 16.5 | 25 |
| [inD](https://www.ind-dataset.com/) | 2020 | Intersection | 4 | 11500 | 10 | 25 |
| [rounD](https://www.round-dataset.com/) | 2020 | Roundabout | 3 | 13746 | 6 | 25 |
| [exiD](https://www.exid-dataset.com/) | 2022 | Highway | 7 | 69172 | 16 | 25 |
| [uniD](https://www.unid-dataset.com/) | 2023 | Intersection | 1 | | | 25 |
| [INTERACTION](http://interaction-dataset.com/) | 2019 | Merging, </br> roundabout, </br> intersection | 11 | 40054 | 16.5 | 10 |
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | 2022 | Parking | 1 | 5188 | 3.5 | 25 |

## Progress

This repository provides SUMO-style road network, lanelet2 style maps, trajectory data insight of the datasets. Here is the current progress of the construction.

| Dataset | Map images | SUMO </br> road network | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) map | Trajectory sample | Trajectory data analysis |
| --- | --- | --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) | $\surd$ | | | [official](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) | $\surd$ ([official](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj))
| [highD](https://www.highd-dataset.com/) | $\surd$ | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| [inD](https://www.ind-dataset.com/) | $\surd$ | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| [rounD](https://www.round-dataset.com/) | $\surd$ | $\surd$ | $\surd$ | $\surd$ | $\surd$ |
| [exiD](https://www.exid-dataset.com/) | $\surd$ |  | $\surd$ | $\surd$ | $\surd$ |
| [uniD](https://www.unid-dataset.com/) | $\surd$ |  || $\surd$ | $\surd$ |
| [INTERACTION](http://interaction-dataset.com/) | $\surd$ | || $\surd$ | |
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | $\surd$ | | $\surd$ | $\surd$ | |

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

@article{zhan2019interaction,
    title={Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps},
    author={Zhan, Wei and Sun, Liting and Wang, Di and Shi, Haojie and Clausse, Aubrey and Naumann, Maximilian and Kummerle, Julius and Konigshof, Hendrik and Stiller, Christoph and de La Fortelle, Arnaud and others},
    journal={arXiv preprint arXiv:1910.03088},
    year={2019}
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

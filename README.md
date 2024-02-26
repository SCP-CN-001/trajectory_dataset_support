# Trajectory Data Support

[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

## About

This repository provides **third-party** map data and script-based support for open trajectory datasets, including NGSIM, LevelXdata (highD, inD, rounD, uniD, and exiD), Interaction, NuPlan, Waymo Open Motion Dataset, and Dragon Lake Park (DLP).

Due to licensing constraints associated with the open datasets, this repository is unable to distribute the complete trajectory datasets. Still, we provide a representative data sample, comprising less than 5% of the original data, to showcase the data format and dataset folder structure. For access to the complete datasets, please reach out to the respective dataset owners to request download links.

> Please raise issues in [this page](https://github.com/WoodOxen/tactics2d/issues) because this repository is a sub project of Tactics2D. You are welcome to join our [discord community](https://discord.com/widget?id=1209363816912126003&theme=dark) if you have more questions about this repository.

## Folder Structure

```shell
.
├── img
│   # the map images
├── map
│   # the maps, including road network format and lanelet2 format
├── trajectory_sample
│   # the trajectory data samples.
├── trajectory_data_analysis
│   # the official and personal analysis of the trajectory data
└── utils
```

## Dataset Overview

Here are some basic information about the open trajectory datasets.

| Dataset | Publish | Scenario | # Map | # Trajectory | Duration (h) | Frequency (Hz) |
| --- | --- | --- | --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) | 2006 | Highway, </br> intersection | 2 | 9206 | 2.5 | 10 |
| [highD](https://www.highd-dataset.com/) | 2018 |  Highway | 6 | 110K+ | 16.5 | 25 |
| [inD](https://www.ind-dataset.com/) | 2020 | Intersection | 4 | 11500 | 10 | 25 |
| [rounD](https://www.round-dataset.com/) | 2020 | Roundabout | 3 | 13746 | 6 | 25 |
| [exiD](https://www.exid-dataset.com/) | 2022 | Highway | 7 | 69172 | 16 | 25 |
| [uniD](https://www.unid-dataset.com/) | 2023 | Intersection | 1 | | | 25 |
| [INTERACTION](http://interaction-dataset.com/) | 2019 | Merging, </br> roundabout, </br> intersection | 11 | 40054 | 16.5 | 10 |
| [NuPlan](https://www.nuscenes.org/nuplan) | 2021 | Urban scenarios | 4 | 38M+ | 1312 | 20 |
| [Waymo Open Motion Dataset v1.2](https://waymo.com/open/about/) | 2021 | Urban scenarios | 6 | | | 10 |
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | 2022 | Parking | 1 | 5188 | 3.5 | 25 |
| [Argoverse 2](https://www.argoverse.org/av2.html) | 2023 | Urban scenarios | 6 | | 763 | 10 |

## Working Progress

This repository plans to provide SUMO-style road network, lanelet2 style maps, trajectory data insight of the datasets. Here is the current progress of the construction.

| Dataset | Map images | SUMO </br> road network | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) map | Trajectory sample | Trajectory data analysis |
| --- | --- | --- | --- | --- | --- |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) | :white_check_mark: | | | [official](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) | :white_check_mark: ([official](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj))
| [highD](https://www.highd-dataset.com/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [inD](https://www.ind-dataset.com/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [rounD](https://www.round-dataset.com/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [exiD](https://www.exid-dataset.com/) | :white_check_mark: |  | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [uniD](https://www.unid-dataset.com/) | :white_check_mark: |  || :white_check_mark: | :white_check_mark: |
| [INTERACTION](http://interaction-dataset.com/) | :white_check_mark: | || :white_check_mark: | :white_check_mark: |
| [NuPlan](https://www.nuscenes.org/nuplan) | :white_check_mark: |  | | :white_check_mark: | :white_check_mark: |
| [Waymo Open Motion Dataset](https://waymo.com/open/about/) | :x: | | | :white_check_mark: | :white_check_mark: |
| [DLP](https://sites.google.com/berkeley.edu/dlp-dataset) | :white_check_mark: | | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Argoverse 2](https://www.argoverse.org/av2.html) | | | | :white_check_mark: | :white_check_mark: | 

## Citation

```latex
% This toolkit is a part of tactics2d project.
@article{li2023tactics2d,
  title={Tactics2D: A Multi-agent Reinforcement Learning Environment for Driving Decision-making},
  author={Li, Yueyuan and Zhang, Songan and Jiang, Mingyang and Chen, Xingyuan and Yang, Ming},
  journal={arXiv preprint arXiv:2311.11058},
  year={2023}
}

% NGSIM
@article{dot2018next,
  title={Next generation simulation (NGSIM) vehicle trajectories and supporting data},
  author={Dot, U},
  journal={US Department of Transportation},
  year={2018}
}

% higD
@inproceedings{highDdataset,
    title={The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems},
    author={Krajewski, Robert and Bock, Julian and Kloeker, Laurent and Eckstein, Lutz},
    booktitle={2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
    pages={2118-2125},
    year={2018},
    doi={10.1109/ITSC.2018.8569552}
}

% inD
@inproceedings{inDdataset,
    title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
    author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
    booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
    pages={1929-1934},
    year={2020},
    doi={10.1109/IV47402.2020.9304839}
}

% rounD
@inproceedings{rounDdataset,
    title={The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany},
    author={Krajewski, Robert and Moers, Tobias and Bock, Julian and Vater, Lennart and Eckstein, Lutz},
    booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)},
    pages={1-6},
    year={2020},
    doi={10.1109/ITSC45102.2020.9294728}
}

% exiD
@inproceedings{moers2022exid,
    title={The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany},
    author={Moers, Tobias and Vater, Lennart and Krajewski, Robert and Bock, Julian and Zlocki, Adrian and Eckstein, Lutz},
    booktitle={2022 IEEE Intelligent Vehicles Symposium (IV)},
    pages={958--964},
    year={2022},
    organization={IEEE}
}

% INTERACTION
@article{zhan2019interaction,
    title={Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps},
    author={Zhan, Wei and Sun, Liting and Wang, Di and Shi, Haojie and Clausse, Aubrey and Naumann, Maximilian and Kummerle, Julius and Konigshof, Hendrik and Stiller, Christoph and de La Fortelle, Arnaud and others},
    journal={arXiv preprint arXiv:1910.03088},
    year={2019}
}

% NuPlan
@article{caesar2021nuplan,
  title={nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles},
  author={Caesar, Holger and Kabzan, Juraj and Tan, Kok Seang and Fong, Whye Kit and Wolff, Eric and Lang, Alex and Fletcher, Luke and Beijbom, Oscar and Omari, Sammy},
  journal={arXiv preprint arXiv:2106.11810},
  year={2021}
}

% Waymo Open Motion Dataset
@inproceedings{ettinger2021large,
  title={Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset},
  author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R and Zhou, Yin and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9710--9719},
  year={2021}
}

% DLP
@INPROCEEDINGS{9922162,
    author={Shen, Xu and Lacayo, Matthew and Guggilla, Nidhir and Borrelli, Francesco},
    booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)}, 
    title={ParkPredict+: Multimodal Intent and Motion Prediction for Vehicles in Parking Lots with CNN and Transformer}, 
    year={2022},
    volume={},
    number={},
    pages={3999-4004},
    doi={10.1109/ITSC55140.2022.9922162}
}

% Argoverse 2
@article{wilson2023argoverse,
  title={Argoverse 2: Next generation datasets for self-driving perception and forecasting},
  author={Wilson, Benjamin and Qi, William and Agarwal, Tanmay and Lambert, John and Singh, Jagjeet and Khandelwal, Siddhesh and Pan, Bowen and Kumar, Ratnesh and Hartnett, Andrew and Pontes, Jhony Kaesemodel and others},
  journal={arXiv preprint arXiv:2301.00493},
  year={2023}
}
```

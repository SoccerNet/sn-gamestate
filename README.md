# SoccerNet Game State Challenge

## Installation
First git clone this repository, and the [TrackLab framework](https://github.com/TrackingLaboratory/tracklab) *in adjacent directories* : 
```bash
mkdir soccernet
cd soccernet
git clone https://github.com/SoccerNet/sn-game-state.git
git clone https://github.com/TrackingLaboratory/tracklab.git
```

### Install using conda
1. Install conda : https://docs.conda.io/projects/miniconda/en/latest/
2. Create a new conda environment : 
```bash 
conda create -n tracklab pip python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate tracklab
```
3. Install all the dependencies with : 
```bash
cd sn-game-state
pip install -e .
pip install -e ../tracklab # You don't need this if you don't plan to change files in tracklab
mim install mmcv-full
```

### Install using Poetry
1. Install poetry : https://python-poetry.org/docs/#installing-with-the-official-installer
2. Install the dependencies : 
```bash
poetry install
mim install mmcv-full
```

### External dependencies

- Get the **SoccerNet Tracking** dataset [here](https://github.com/SoccerNet/sn-tracking), rename the root folder as "SoccerNetMOT" and put it under the global dataset directory (specified under the `data_dir` config as explained below). Otherwise, you can modify the `dataset_path` config in [soccernet_mot.yaml](configs/dataset/soccernet_mot.yaml) with your custom SoccerNet dataset directory.
- Download the pretrained model weights [here](https://drive.google.com/drive/folders/1MmDkSHWJ1S-V9YcLMkFOjm3zo65UELjJ?usp=drive_link) and put the "pretrained_models" directory under the main project directory (i.e. "/path/to/tracklab/pretrained_models").

### Setup

You will need to set up some variables before running the code : 

1. In [soccernet.yaml](sn_gamestate/configs/soccernet.yaml) :
   - `data_dir`: the directory where you will store the different datasets (must be an absolute path !)
   - All the parameters under the "Machine configuration" header
2. For the corresponding modules :
   - The `batch_size`
   - You might want to change the model hyperparameters


## Usage

run TrackLab on the command line with `tracklab` or `python -m tracklab.main`. All the additional
SoccerNet modules will be added automatically when installing this repository.

You can find all possible configuration groups possible at the top when running the following command :  
```bash
python -m tracklab.main --help
```

By default, tracklab will use generic defaults, in order to use the appropriate defaults for the
SoccerNet baseline, run it with :
```bash
python -m tracklab.main -cn soccernet
```

You can change the values of this config in [soccernet.yaml](sn_gamestate/configs/soccernet.yaml).

## Adding a new module

If you want to add a new module in the tracklab pipeline, you can either add it in this repository,
by adding code in (a new directory in) [sn_gamestate](sn_gamestate) and configuration files in 
[sn_gamestate/configs/modules](sn_gamestate/configs/modules), which will be added automatically. 

If you would like to create a separate project that makes use of tracklab, you will need to declare
the location of your config file using an [entry point](https://setuptools.pypa.io/en/stable/userguide/entry_point.html#entry-points-for-plugins).
The entry point group should be `tracklab_plugin` and it should point to a class containing a variable called `config_package`,
as shown [here](sn_gamestate/config_finder.py), this variable should point to the location of your configuration folder.
Temporarily, you can also specify a directory using Hydra's `--config-dir`.

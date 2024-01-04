# SoccerNet Game State Challenge

## Installation
First git clone this repository, and the TrackLab framework *in adjacent directories* : 
```bash
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

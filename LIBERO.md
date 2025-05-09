# Instructions for LIBERO experiments
* Clone LIBERO repo from github, put in the same directory of GR00T repo

## Setup environment
Set current working dir to GR00T repo
```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install -e .
cd ..
pip install -e LIBERO --use-pep517
cd Isaac-GR00T
pip install -r experiments/robot/libero/libero_requirements.txt
```
Extra dependencies:
```bash
sudo apt-get install libosmesa6-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev 
sudo apt-get install python3-opengl
```
Some changes: refer to https://github.com/ARISE-Initiative/robosuite/issues/490
```bash
sudo mkdir /usr/lib/dri
cd /usr/lib/dri
sudo ln -s  /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so swrast_dri.so
conda install -c conda-forge gcc
```

## Setup environment variables
```bash
export MUJOCO_GL="egl"
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
```
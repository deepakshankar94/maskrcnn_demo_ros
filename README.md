# maskrcnn-demo


## Install pipenv 


```bash
pip install pipenv
```

- to make it create environment in the project folder export the following variable (can be added to .bashrc for ease)
```bash
export PIPENV_VENV_IN_PROJECT=1 
```
## Clone this repo 

```bash
git clone https://github.com/deepakshankar94/maskrcnn_demo_ros
cd maskrcnn_demo_ros
```
## Create a virtual environment

- Create the pipenv environment

```bash
pipenv install --skip-env
```

## compile the pytorch library from source 

(This step might break on the ryzen because of mkl-dnn library )

- activate the virtual env

```
pipenv shell
```


- Setup the environment variable required for compiling pytorch without CUDA

```bash
export NO_CUDA=1
```


- Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

- Install PyTorch

```bash
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
cd ..
```

## install project dependencies

```bash



#install this only if you don't compile from source
#conda install pytorch-nightly -c pytorch

export INSTALL_DIR=$PWD
# install torchvision
cd $INSTALL_DIR
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

#install pycoco only for training new model. Currently this is not required for the demo
# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

python setup.py build develop


unset INSTALL_DIR

cd ..
```

## run the demo

```bash
#source the ros environment
source /opt/ros/kinetic/setup.sh

catkin_make

#source your package
source devel/setup.bash

#run the demo
roslaunch maskrcnnpkg demo.launch

#if the tensor error is present check the camera source in the demo
```


### TODO

- [X] setup demo with read me
- [x] Integrate the ROS code 
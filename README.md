# Move BeTween modAlities (MBTA) employs flow matching to predict single cell data modalities
MBTA is written in python 3.13.2



![Alt text](imgs/MBTA.png)

MBTA allows user to connect any two co-profiled molecular modalities or multiple single-modal datasets with guidance from cell level annotations.



## How to Run
You can create a Conda environment and install all required dependencies using the commands below:
```bash
# clone project
git clone https://github.com/bxxu/move_between_modalities.git
cd move_between_modalities

# create conda environment
conda create -n mbta python=3.13
conda activate mbta

# install requirements
pip install -r requirements.txt
```
After installing the dependencies, you can refer to the tutorial notebooks for examples of how to run MBTA.


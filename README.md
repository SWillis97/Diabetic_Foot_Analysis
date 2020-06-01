# Diabetic_Foot_Analysis
This guide provides detailed walkthrough of how to run the standalone programs and the python scripts, if you choose to edit the program.

## Standalone Package
It is recommended this be done on a windows computer. There is a script which works with Mac OS but the app package was tempermental, not working on all devices.

To run the standalone package download `standalone.zip`. Unzip it into a location and double click the executable file to process the test images. 

## Python Code
This information is for users who wish to run the python code for further development


### Scripts
`Foot_Scanner.py`: The python script contained in the standalone program.


### Libraries
Please install the following libraries if they have not been installed previously. These libraries are used in all example scripts.

- Numpy: `pip install numpy`
- OpenCV: `pip install opencv-contrib-python`
- Pillow: `pip install pillow`
- Shutil: `pip install pytest-shutil`
- math: `pip install maths`



### Outputs
After running the Python code, the following files should be created:

`WT_Pout.csv` Contains power output of all wind turbine models at all time stamps of the wind speed data of the whole year. Very large dataset.

`price.csv` Contains wind turbine models specifications and the arbitrary prices. 

## Matlab Code
### Libraries
Go to Add-Ons and install Global Optimization Toolbox

### Input Datasets
`WT_Pout.csv` `oneYearPower.csv` and `price.csv` Place these three files into the same directory as the MATLAB files. 

### Functions
`Demo.m`: Generate a wind turbine farm for an area of 2 km by 2 km with electricity load profile from `oneYearPower.csv`.

`turb_selection.m`: Uses Linear Programming to select the turbine models and the number of them for a build area and a load requirement profile. 

`turb_placement.m`: Uses Pattern Search to place the selected turbine models into the given build area to maximise the power output of the wind turbine farm.

### Outputs
Running `Demo.m` will generate the following outputs:

_Model_Name_: names of the selected turbine models.

_Numbers_: the number of each selected turbine models. 

_Installation_Cost_: the cost of purchasing the selected turbine models. 

_Final_Power_Output_: a one-year power output from the selected turbine models.

_Turbine_Locations_: the locations of each selected turbine model in the given build area

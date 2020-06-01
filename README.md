# Diabetic_Foot_Analysis
This guide provides detailed walkthrough of how to run the standalone programs and the python scripts, if you choose to edit the program. The code, it's purpose and it's functions are explained in `Samuel Willis Final Project Report.pdf`. A video demonstrating the program can be found at https://youtu.be/81_xTH1K_os.

## Standalone Package
It is recommended this be done on a windows computer. There is a script which works with Mac OS but the app package was tempermental, not working on all devices.

To run the standalone package download `Windows_Standalone.zip` from https://drive.google.com/file/d/1Va5EHuX0uSfJVQX0_1vBerb9DdL2RL9a/view?usp=sharing; the file was too large to upload to github but can be found on google drive. Unzip it into a location and double click the executable file to process the test images. 

## Python Code
This information is for users who wish to run the python code for further development


### Scripts
- `Foot_Scanner.py`: The python script contained in the standalone program.
- `Foot_Scanner_with_API.py`: This python script has an API which uploads to a google sheets document. A Google API client secret json will be needed for this.
- `Polar_Coordinates.py`: This python script contains the functions which complete the polar analysis mentioned in the report. It can use any of the example images but was most extensively tested with `Example Foot One.jog`
- `Bresenhams.py`: This script is used with `Bresenhams_Test.jpg` to demonstrate the closing effect.

### Libraries
Please install the following libraries if they have not been installed previously. These libraries are used in all example scripts.

- Numpy: `pip install numpy`
- OpenCV: `pip install opencv-contrib-python`
- Pillow: `pip install pillow`
- Shutil: `pip install pytest-shutil`
- math: `pip install maths`

If you wish to run `Bresenhams.py` you will also need to install the Bresenhams library: `pip install bresenham`

If you wish to run `Foot_Scanner_with_API.py` or `Polar_Coordinates.py` you will also need to install the following libraries:

- pygsheets: `pip install pygsheets`
- matplotlib: `pip install matplotlib`
- Sklearn: `pip install sklearn`
- argparse: `pip install argparse`
- gspread: `pip install gspread`
- google auth: `pip install --upgrade google-auth`
- googleapiclient: Follow instructions on https://pypi.org/project/google-api-python-client/

## Outputs
After running the Python code or standalone program a new folder will be made containing images of the process at different stages and two csv files which can be imported into CAD packages to make 3D models. Fusion360 supports this well with the inbuilt csv2spline script.



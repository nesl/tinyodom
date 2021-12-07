# EuRoC MAV Dataset Download and Splitting Guide:

- Download the dataset from here: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets . Please download the dataset from the links pertaining to 'ASL Dataset Format`. Extract the zip files and put each data folder (e.g., ```MH_01_easy, V2_02_medium```, etc.) in a single folder titled ```euroc_mav```.
- Please the ```.txt``` files in the home folder of the dataset,
- Check the ```data_utils.py``` file to see how TinyOdom imports data.

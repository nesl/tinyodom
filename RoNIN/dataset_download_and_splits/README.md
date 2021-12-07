# RoNIN Dataset Download and Splitting Guide:

- Download the dataset from here: https://ronin.cs.sfu.ca/ (P.S. We just found out that the RoNIN paper authors have deleted the dataset. We cannot upload the copy we have without their permission).
- First, put all the subfolders (they look like `axyz_w```) in the folders ```train_dataset_1``` and ```train_dataset_2``` in a single folder called ```train_dataset_full```. Then put ```list_train_full.txt```, ```list_train.txt```, and ```list_val.txt``` in the ```train_dataset_full``` folder. Put ```list_test_seen.txt``` in the ```seen_subjects_test_set``` folder, and ```list_test_unseen.txt``` in the ```unseen_subjects_test_set``` folder
- Check the ```data_utils.py``` file to see how TinyOdom imports data.

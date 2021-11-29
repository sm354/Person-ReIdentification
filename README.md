![Python](https://img.shields.io/badge/python-3.7-blue?style=flat-square&logo=python)
# Person Re-Identification

## Results

|  Model   | CMC@Rank-1 | CMC@Rank-5 | mAP  |                           Download                           |
| :------: | :--------: | :--------: | :--: | :----------------------------------------------------------: |
| Baseline |    0.92    |    0.96    | 0.91 | [model](https://drive.google.com/file/d/1IxTAUOjS3_S4sF1mRJ72Mp5Xo-omQu6a/view?usp=sharing) |
|   Ours   |    0.93    |   0.964    | 0.95 |                            Model                             |

## Installation

```bash
git clone https://github.com/sm354/Pedestrian-Detection.git
cd Pedestrian-Detection
pip install -r requirements.txt
```

`PennFudanPed_train.json`, and `PennFudanPed_val.json` contains COCO annotations for a randomly generated train-val split of the PennFudan dataset. 

## Running Models

**Training** 

```bash
python run-train.py --train_data_dir ./data/train --model_name la-tf++ --model_dir ./model --num_epochs 25
```

**Testing**

```bash
python run-test.py --model_path ./model/la-tf++.pth --test_data ./test
```

The script `run-test.py` takes in the query and gallery images and computes the following metrics:

1. CMC@Rank-1
2. CMC@Rank-5
3. mean Average Precision (mAP)

## Acknowledgements

- Evaluation Metrics are adapted from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/v1.0.6/torchreid/metrics/rank_cylib/rank_cy.pyx).
- Re-Ranking is adapted from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking_ranklist.py).
- Random Grayscale Patch Replacement is adapted from [Data-Augmentation](https://github.com/finger-monkey/Data-Augmentation/blob/main/trans_gray.py).
- Random Erasing is adapted from [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py).
- Triplet Loss is adapted from [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py).


## Authors

- [Shubham Mittal](https://www.linkedin.com/in/shubham-mittal-6a8644165/)
- [Aditi Khandelwal](https://www.linkedin.com/in/aditi-khandelwal-991b1b19b/)

Course assignment in Computer Vision course ([course webpage](https://www.cse.iitd.ac.in/~chetan/teaching/col780-2020.html)) taken by [Prof. Chetan Arora](https://www.cse.iitd.ac.in/~chetan)




This repository provides the starter code and the dataset (train and val) for the project.

### Dataset
* The dataset has 114 unique persons. The train and val set contain 62 and 12 persons, respectively.
* We have held out the test set of 40 persons.
* Each person has been captured using 2 cameras from 8 different angles. That is, each person would have 16 images. All images of a unique person is stored in a single directory (numbered from 001 to 114).
* The dataset has the following directory structure:

* The images of a person in the val set in split into query and gallery images. The query is the set of images which will be used to retrieve the images of the same person from the gallery. 
* Note that query and gallery are mutually exclusive sets.

### TODO
1. You need to write the code for a Person Re-ID model and train it. 
2. Write the training script and save the model after training.
3. Evaluate and analyze the results. 
	* The quantitative evaluation script has been given to calculate CMC@rank-1, CMC@rank-5, and mAP scores. 
	* You need to write code for any visualization.


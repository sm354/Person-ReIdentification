![Python](https://img.shields.io/badge/python-3.7-blue?style=flat-square&logo=python)
# Person Re-Identification

Person re-identification (Re-ID) is the task of identifying a person-of-interest (query person) at other time and/or location captured using same or different camera at same or different orientation. Re-ID is addressed as image-retrieval problem where we have a set of images of different person, called as gallery, and we retrieve the most similar person to the query person from the gallery. 

In this work we implement Locally Aware Transformer ([LA-TF](https://arxiv.org/pdf/2106.03720.pdf)) on [PKU-Reid](https://arxiv.org/pdf/1605.02464v1.pdf) dataset, and make design changes to address its limitations. 


## Results

|     Model      | CMC@rank-1 | CMC@rank-5 | mAP  |                           Download                           |
| :------------: | :--------: | :--------: | :--: | :----------------------------------------------------------: |
|    LA-TF    |    92.9    |    96.4    | 91.5 | [model](https://drive.google.com/file/d/1IxTAUOjS3_S4sF1mRJ72Mp5Xo-omQu6a/view?usp=sharing) |
| LA-TF++ (ours) |    92.9    |    1.0     | 93.2 | [model](https://drive.google.com/file/d/1alTMrdiupa2McGkSOJCgO_QC4DCNhc0f/view?usp=sharing) |

## Installation

```bash
pip install -r requirements.txt
```

## Running Models

### **Training** 

**Locally-Aware Transformer (Baseline)**

```bash
python train_baseline.py --train_data_dir ./data/train --model_name la-tf_baseline --model_dir ./model
```

**LA-TF++ (Our model)** 

```bash
python run-train.py --train_data_dir ./data/train --model_name la-tf++_final --model_dir ./model
```

### **Testing**

```bash
python run-test.py --model_path <path-to-saved-model> --test_data ./data/val
```

The script `run-test.py` takes in the query and gallery images (present in the `test_data`) and computes the following metrics:

1. CMC@rank-1
2. CMC@rank-5
3. mean Average Precision (mAP)

#### Visualization

```bash
python run-test.py --model_path <path-to-saved-model> --test_data ./data/val --visualize --save_preds <path-to-save-images>
```

## Dataset

The dataset has 114 unique persons. The train and val set contain 62 and 12 persons, respectively. Each person has been captured using 2 cameras from 8 different angles. 

## Acknowledgements

- Locally Aware Transformer (LA-TF) is adaped from [Person Re-Identification with a Locally Aware Transformer](https://github.com/SiddhantKapil/LA-Transformer).
- Triplet Loss and Label Smoothing are adapted from [Alignedreid++](https://github.com/michuanhaohao/AlignedReID).

Computer Vision course project ([course webpage](https://www.cse.iitd.ac.in/~chetan/teaching/col780-2020.html)) taken by [Prof. Chetan Arora](https://www.cse.iitd.ac.in/~chetan)

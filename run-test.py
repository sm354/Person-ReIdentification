import os
import time
import argparse
import random
import timm
import numpy as np
import faiss
from PIL import Image
# from tqdm.notebook import tqdm
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model import *
from utils import *
from metrics import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train Person ReID Model')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--model_path', type=str, default="./model/la-tf++.pth")
    parser.add_argument('--test_data', type=str, default="/home/shubham/CVP/test")
    args = parser.parse_args()
    return args

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    for data in tqdm(dataloaders):
        img, label = data
        img, label = img.to(device), label.to(device)

        output = model(img)
        features = torch.cat((features, output.detach().cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, label in img_path:
        cam_id = int(path.split("/")[-1].split("_")[0])
        # filename = os.path.basename(path)
        # camera = filename.split('_')[0]
        labels.append(int(label))
        camera_id.append(cam_id)
    return camera_id, labels

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k

args = parse_args()
fix_seed(args.seed)

### Hyper parameters
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
lr = 3e-4
gamma = 0.7
lmbd = 8
model_path = args.model_path
# data_dir = "/home/shubham/CVP/data/val"
data_dir = args.test_data

transform_query_list = [
    transforms.Resize((224,224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_gallery_list = [
    transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
data_transforms = {
    'query': transforms.Compose( transform_query_list ),
    'gallery': transforms.Compose(transform_gallery_list),
}

image_datasets = {}
image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

class_names_query = image_datasets['query'].classes
class_names_gallery = image_datasets['gallery'].classes
print("number of classes in query and gallery", len(class_names_query), len(class_names_gallery))

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)
# Create La-Transformer
model = LATransformer(vit_base, lmbd=lmbd, num_classes=62, test=True).to(device)

# Load LA-Transformer
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()

# Extract Query Features
query_feature = extract_feature(model, query_loader)
# Extract Gallery Features
gallery_feature = extract_feature(model, gallery_loader)

# Retrieve labels
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

concatenated_query_vectors = []
for query in tqdm(query_feature):
    fnorm = torch.norm(query, p=2, dim=1, keepdim=True)*np.sqrt(14)
    query_norm = query.div(fnorm.expand_as(query))
    concatenated_query_vectors.append(query_norm.view((-1))) # 14*768 -> 10752
concatenated_gallery_vectors = []
for gallery in tqdm(gallery_feature):
    fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True) *np.sqrt(14)
    gallery_norm = gallery.div(fnorm.expand_as(gallery))
    concatenated_gallery_vectors.append(gallery_norm.view((-1))) # 14*768 -> 10752


## Calculate Similarity using FAISS
index = faiss.IndexIDMap(faiss.IndexFlatIP(10752))
index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label))

## Evaluate 
rank1_score = 0
rank5_score = 0
ap = 0
count = 0
for query, label in zip(concatenated_query_vectors, query_label):
    count += 1
    label = label
    output = search(query, k=10)
    rank1_score += rank1(label, output) 
    rank5_score += rank5(label, output) 
    ap += calc_map(label, output)

print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score))
print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                             rank5_score/len(query_feature), 
                                             ap/len(query_feature)))    
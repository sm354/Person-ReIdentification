import os
from shutil import copyfile
import random
import csv
import torch
from collections import OrderedDict

def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
        
def save_network(network, name):
    save_filename = "_best.pth"
    save_path = os.path.join('/home/shubham/CVP/model/', name + save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    
    if torch.cuda.is_available():
        network.cuda()
        
# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         #filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        label = path.split("/")[-2]
        filename = os.path.basename(path)
        camera = filename.split('_')[0]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

def train_val_split(TrainData_path="./data/train", new_path="./TrainData_split"):
    train_path = os.path.join(new_path, "train")
    val_path = os.path.join(new_path, "val")
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        os.mkdir(train_path)
        os.mkdir(val_path)

    person_ids = os.listdir(TrainData_path)
    for person_id in person_ids:
        person_id_path = os.path.join(TrainData_path, person_id)
        try:
            int(person_id)
        except:
            continue

        os.mkdir(os.path.join(train_path, person_id))
        os.mkdir(os.path.join(val_path, person_id))

        img_names = os.listdir(person_id_path)
        img_names.sort()
        val_imgs = random.sample(img_names, 2)

        for file_name in img_names:
            if file_name in val_imgs:
                target_path = os.path.join(os.path.join(val_path, person_id), file_name)
            else:
                target_path = os.path.join(os.path.join(train_path, person_id), file_name)
            
            copyfile(os.path.join(person_id_path, file_name), target_path)

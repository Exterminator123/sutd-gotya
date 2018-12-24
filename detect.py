## PyTorch implementation of YOLOv3
## from https://github.com/eriklindernoren/PyTorch-YOLOv3

## After detecting images, the cropped images will be uploaded to Firebase Storage.

## Usage:
## $ python3 detect.py --image_path <folder containing sample images>


from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import PIL.Image
import PIL.ImageDraw

import firebase


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='data/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='data/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.2, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.6, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--output_folder', type=str, default='output', help='output directory')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 89, opt.conf_thres, opt.nms_thres)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

print ('\nSaving images:')

os.makedirs(opt.output_folder)

log_file = open(os.path.join(opt.output_folder, 'log.csv'), 'w')
log_file.write("Filename,Label,x1,y1,x2,y2,Conf\n")

# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    os.makedirs(os.path.join(opt.output_folder, "%d" % img_i))

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    graph_img = ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x


    # image to crop later
    img_to_crop = PIL.Image.open(path)

    boxes = []

    print(detections)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for detection_i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            textbox = plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

            # crop and save the image

            boxes.append({ "classification": classes[int(cls_pred)], 
                        "x1": x1.item(), "y1": y1.item(), 
                        "x2": x1.item() + box_w.item(), "y2": y1.item() + box_h.item() })

            log_file.write(",".join([path.split("/")[-1], classes[int(cls_pred)], str(x1.item()), str(y1.item()), str(x1.item() + box_w.item()), str(y1.item() + box_h.item()), str(cls_conf.item())]) + "\n")

            crop_img = img_to_crop.crop((x1.item(), y1.item(), x1.item() + box_w.item(), y1.item() + box_h.item()))
            #text_img = PIL.ImageDraw.Draw(crop_img)
            #text_img.text((0, 0), classes[int(cls_pred)])
            crop_img.save(os.path.join(opt.output_folder, '%d/%d.png' % (img_i, detection_i)), 'PNG')
            crop_img.close()
            

    graph_img.set_clip_path(None)
    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(os.path.join(opt.output_folder, '%d/original.png' % (img_i)), bbox_inches='tight', pad_inches=0.0)
    plt.clf()
    plt.close("all")
    #firebase.upload_batch(os.path.join(CURRENT_DIR, 'output/{}'.format(img_i)), boxes)

log_file.close()

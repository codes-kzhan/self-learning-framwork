# Copyright (c) 2017 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib.image as mpimg
import xml.dom
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from generateXML import GenerateXml
from PIL import Image
import shutil
import  xml.dom.minidom
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES_LABEL_NUM = {
           'aeroplane':0, 'bicycle':0, 'bird':0, 'boat':0,
           'bottle':0, 'bus':0, 'car':0, 'cat':0, 'chair':0,
           'cow':0, 'diningtable':0, 'dog':0, 'horse':0,
           'motorbike':0, 'person':0, 'pottedplant':0,
           'sheep':0, 'sofa':0, 'train':0, 'tvmonitor':0}

CLASSES_SCORE_NUM = {
           'aeroplane':0, 'bicycle':0, 'bird':0, 'boat':0,
           'bottle':0, 'bus':0, 'car':0, 'cat':0, 'chair':0,
           'cow':0, 'diningtable':0, 'dog':0, 'horse':0,
           'motorbike':0, 'person':0, 'pottedplant':0,
           'sheep':0, 'sofa':0, 'train':0, 'tvmonitor':0}

NETS = {'ResNet-101': ('ResNet-101',
                  'voc_0712_baseline_80.2.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}

def demo(folder_image,net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit17/VOC17/JPEGImages/', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    labelarray=""
    scorearray=""
    CONF_THRESH_MAX = 0.4
    NMS_THRESH = 0.3
    scores_list=[]
    for cls_ind, cls in enumerate(CLASSES[1:]): 
        cls_ind += 1 # because we skipped background
        # print cls_ind
        cls_boxes = boxes[:, 4:8]
        # print cls_boxes.shape
        cls_scores = scores[:, cls_ind]
        # print 'cls_score:',cls_scores.shape
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print dets.shape
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH_MAX)[0]
        if len(inds) == 0 :
            continue
        for i in inds:
            score = dets[i, -1]
            labelarray = labelarray + " " + cls
            scorearray = scorearray + " " + str( score )

        scores_list.append(max(dets[:, -1]))    

    if len(scorearray)>0:
        max_num=max(scores_list)
        scores_list.remove(max_num)
        if len(scores_list) > 1:
            second_max_num = max(scores_list)
        else:
            second_max_num = 0
        if second_max_num <= 0.2 or max_num-second_max_num > 0.4:
            label_split=labelarray.split(" ")
            score_split=scorearray.split(" ")
            for i in xrange(len(label_split)-1):
                CLASSES_LABEL_NUM[label_split[i+1]]=CLASSES_LABEL_NUM[label_split[i+1]]+1
                CLASSES_SCORE_NUM[label_split[i+1]]=CLASSES_SCORE_NUM[label_split[i+1]]+float(score_split[i+1])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=3, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    fold_JPGE="/home/wangkeze/yanxp/dataset/VOCdevkit00/JPEGImages/"
    folder_annotation="/home/wangkeze/yanxp/dataset/VOCdevkit00/Annotations/"
    fold_UserLabelImage="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelImages/"
    fold_UserLabelAnnotations="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelAnnotations/"
    shutil.rmtree(fold_JPGE)
    os.mkdir(fold_JPGE)
    shutil.rmtree(folder_annotation)
    os.mkdir(folder_annotation)
    shutil.rmtree(fold_UserLabelImage)
    os.mkdir(fold_UserLabelImage)
    shutil.rmtree(fold_UserLabelAnnotations)
    os.mkdir(fold_UserLabelAnnotations)
   
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    global pred_accuracy_num
    global userlabel_num
    global label_user
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    folder_image='data/VOCdevkit17/VOC17/JPEGImages/'
    trainval="/home/wangkeze/yanxp/dataset/VOCdevkit17/VOC17/ImageSets/Main/trainval.txt"
    i = 0
    for line in open(trainval) :
        line=line.strip('\n')   
        if os.path.isfile(folder_image+line+'.jpg'):
            i=i+1
            if i < 30000:
               continue
            elif i > 40000:
                break
            demo(folder_image,net,line+'.jpg')
            print i

    for d in CLASSES_LABEL_NUM:  
        print "%s:%s" %(d, CLASSES_SCORE_NUM[d]/CLASSES_LABEL_NUM[d])


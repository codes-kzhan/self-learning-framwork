# Copyright (c) 2015 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
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
from myIO import loadData, saveData

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES_SCORE_NUM = {
           'aeroplane':0.731, 'bicycle':0.700, 'bird':0.725, 'boat':0.662,
           'bottle':0.684, 'bus':0.785, 'car':0.790, 'cat':0.798, 'chair':0.696,
           'cow':0.751, 'diningtable':0.714, 'dog':0.742, 'horse':0.759,
           'motorbike':0.832, 'person':0.816, 'pottedplant':0.669,
           'sheep':0.754, 'sofa':0.736, 'train':0.716, 'tvmonitor':0.800}
NETS = {'ResNet-101': ('ResNet-101',
                  'voc_0712_baseline_80.2.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}
                  
fold_JPGE="/home/wangkeze/yanxp/dataset/VOCdevkit00/JPEGImages/"
folder_annotation="/home/wangkeze/yanxp/dataset/VOCdevkit00/Annotations_pseudo/"
fold_UserLabel="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelImages/"
fold_UserLabelAnnotations="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelAnnotations/"
folder_Annotation_original="/home/wangkeze/yanxp/dataset/VOCdevkit2012/VOC2012/Annotations/"   
NMS_THRESH = 0.3             
                  
global pred_accuracy_num
pred_accuracy_num=0
global userlabel_num
userlabel_num=0
global num
num=0
def demo(folder_image,net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit2012/VOC2012/JPEGImages/', image_name)
    
    im = cv2.imread(im_file)
    height=im.shape[0]
    width=im.shape[1]
    resize_width=height
    resize_height=width
    
    #image_file=Image.open(folder_image+image_name)
    timer = Timer()
    
    timer.tic()

    try: 
       res = loadData( 'cache/' + image_name + '.pkl' ) 
       scores = res[0]
       boxes = res[1]
    except:
       scores, boxes = im_detect(net, im)
       saveData( [scores, boxes], 'cache/' + image_name + '.pkl' )
    
    timer.toc()

    labelarray=""
    coord=""
    scorearray=""
    scores_list=[]

    for cls_ind, cls in enumerate(CLASSES[1:]): 
        cls_ind += 1 # because we skipped background
        # print cls_ind
        cls_boxes = boxes[:, 4:8]
        # print cls_boxes.shape
        cls_scores = scores[:, cls_ind]
        CONF_THRESH_MAX=0.6
        # print 'cls_score:',cls_scores.shape
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print dets.shape
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH_MAX)[0]
    
        if len(inds) == 0 :
            continue
        
        scores_list.append(max(dets[:, -1]))
        
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            labelarray=labelarray+" "+cls
            class_folder="/home/wangkeze/yanxp/dataset/VOCdevkit00/"+cls+"/"
            scorearray=scorearray+" "+str(score)
            xmin=str(bbox[0]/resize_height*width)
            ymin=str(bbox[1]/resize_width*height)
            xmax=str(bbox[2]/resize_height*width)
            ymax=str(bbox[3]/resize_width*height)
            coord=coord+" "+xmin+" "+ymin+" "+xmax+" "+ymax

    if len(scorearray)>0:
        max_num=max(scores_list)
        scores_list.remove(max_num)
        if len(scores_list) > 1:
            second_max_num = max(scores_list)
        else:
            second_max_num = 0      
        
        if second_max_num>0.2 and (max_num-second_max_num)<0.4:
            #image_file.save(fold_UserLabel+image_name)
            img_name_1=image_name.split(".")
            if os.path.isfile(folder_Annotation_original+img_name_1[0]+'.xml'):
                shutil.copy(folder_Annotation_original+img_name_1[0]+'.xml',fold_UserLabelAnnotations)
            global userlabel_num
            userlabel_num += 1
            print "userlabel_num:",userlabel_num

        else:
	    scorenum=scorearray.split(" ")
	    global num
            num=num+len(scorenum)-1
	    print "num:",num
            #image_file.save(fold_JPGE+image_name)
            GenerateXml(class_folder,image_name,folder_annotation,width,height,scorearray,labelarray,coord)
            global pred_accuracy_num
            pred_accuracy_num += 1
            print "pred_accuracy_num:",pred_accuracy_num
        
    else:
        #image_file.save(fold_UserLabel+image_name)
        img_name_8=image_name.split(".")
        if os.path.isfile(folder_Annotation_original+img_name_8[0]+'.xml'):
            shutil.copy(folder_Annotation_original+img_name_8[0]+'.xml',fold_UserLabelAnnotations)
        global userlabel_num
        userlabel_num += 1
        print "userlabel_num:",userlabel_num

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
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
    folder_annotation="/home/wangkeze/yanxp/dataset/VOCdevkit00/Annotations_pseudo/"
    fold_UserLabelImage="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelImages/"
    fold_UserLabelAnnotations="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelAnnotations/"

    #shutil.rmtree(fold_JPGE)
    #os.mkdir(fold_JPGE)
    #shutil.rmtree(folder_annotation)
    #os.mkdir(folder_annotation)
    #shutil.rmtree(fold_UserLabelImage)
    #os.mkdir(fold_UserLabelImage)
    #shutil.rmtree(fold_UserLabelAnnotations)
    #os.mkdir(fold_UserLabelAnnotations)

    #for cls_ind, cls in enumerate(CLASSES[1:]): 
    #    class_folder="/home/wangkeze/yanxp/dataset/VOCdevkit00/"+cls+"/"
    #    if not os.path.exists(class_folder):
    #        os.makedirs(class_folder)
    #    shutil.rmtree(class_folder)
    #    os.makedirs(class_folder)
   
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
    global num

    folder_image='data/VOCdevkit2012/VOC2012/JPEGImages/'
    trainval="/home/wangkeze/yanxp/dataset/VOCdevkit2012/VOC2012/ImageSets/Main/trainval.txt"
    total_image_num = len( open(trainval).readlines() )    
    
    i=0
    beginId = total_image_num * 0 / 4
    endId = total_image_num * 4 / 4
    print beginId, endId
    for line in open(trainval) :
        line=line.strip('\n')  
        if os.path.isfile(folder_image+line+'.jpg'):   
            i=i+1
            if i < beginId :
                continue
            elif i > endId :
                break          
            
            demo(folder_image,net,line+'.jpg')
            print 'Current progress %d/%d' % ( i, endId )

    print "predict accuracy:",pred_accuracy_num
    print "userlabel_num:",userlabel_num
    print "num:",num

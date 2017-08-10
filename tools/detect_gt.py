# Copyright (c) 2015 Yuwen Xiong
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

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES_SCORE_NUM = {
           'aeroplane':0.928411, 'bicycle':0.901429, 'bird':0.913278, 'boat':0.860593,
           'bottle':0.841920, 'bus':0.914320, 'car':0.906638, 'cat':0.921350, 'chair':0.815335,
           'cow':0.837749, 'diningtable':0.799092, 'dog':0.897752, 'horse':0.884513,
           'motorbike':0.882651, 'person':0.917197, 'pottedplant':0.824575,
           'sheep':0.875486, 'sofa':0.801787, 'train':0.906840, 'tvmonitor':0.899380}
NETS = {'ResNet-101': ('ResNet-101',
                  'voc_0712_80.3.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}
global pred_accuracy_num
pred_accuracy_num=0
global userlabel_num
userlabel_num=0
global wrong_class_num
wrong_class_num=0
global wrong_iou_num
wrong_iou_num=0
def demo(folder_image,net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    fold_JPGE="/home/wangkeze/yanxp/dataset/VOCdevkit00/JPEGImages/"
    folder_annotation="/home/wangkeze/yanxp/dataset/VOCdevkit00/Annotations/"
    fold_UserLabel="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelImages/"
    fold_UserLabelAnnotations="/home/wangkeze/yanxp/dataset/VOCdevkit00/UserLabelAnnotations/"
    folder_Annotation_original="/home/wangkeze/yanxp/dataset/VOCdevkit17/VOC17/Annotations/"
    image_original = mpimg.imread(folder_image+image_name)
    height=image_original.shape[0]
    width=image_original.shape[1]
    # print (height,width)
    im_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit17/VOC17/JPEGImages/', image_name)
    im = cv2.imread(im_file)
    resize_width=height
    resize_height=width
    im=cv2.resize(im,(resize_height,resize_width),interpolation=cv2.INTER_CUBIC)
    image_file=Image.open(folder_image+image_name)
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    xmlfile=folder_Annotation_original+image_name.split(".")[0]+".xml"
    dom = xml.dom.minidom.parse(xmlfile)
    names_original=dom.getElementsByTagName('name')
    class_name=""
    for i in xrange(len(names_original)):
        class_name=class_name+" "+names_original[i].firstChild.data

    xmin_orig=dom.getElementsByTagName('xmin')
    ymin_orig=dom.getElementsByTagName('ymin')
    xmax_orig=dom.getElementsByTagName('xmax')
    ymax_orig=dom.getElementsByTagName('ymax')
    timer.toc()

    labelarray=""
    coord=""
    scorearray=""
#    CONF_THRESH_MAX = 0.86
    NMS_THRESH = 0.3
    scores_list=[]
    label_iou=""
    coord_iou=""
    score_iou=""
    label_gt=""
    coord_gt=""
    # scores_user=[]
    for cls_ind, cls in enumerate(CLASSES[1:]): 
        cls_ind += 1 # because we skipped background
        # print cls_ind
        cls_boxes = boxes[:, 4:8]
        # print cls_boxes.shape
        cls_scores = scores[:, cls_ind]
        CONF_THRESH_MAX=float(CLASSES_SCORE_NUM[cls])-0.1
        # print 'cls_score:',cls_scores.shape
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print dets.shape
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH_MAX)[0]
        scores_list.append(max(dets[:, -1]))
        # scores_user.append(max(dets[:, -1]))
        if len(inds) == 0 :
            continue
        im = im[:, :, (2, 1, 0)]   

        scores_list.remove(dets[0,-1])
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
        second_max_num=max(scores_list)
        if second_max_num>0.2:
            if (max_num-second_max_num)<0.4:
                image_file.save(fold_UserLabel+image_name)
                img_name_1=image_name.split(".")
                if os.path.isfile(folder_Annotation_original+img_name_1[0]+'.xml'):
                    shutil.copy(folder_Annotation_original+img_name_1[0]+'.xml',fold_UserLabelAnnotations)
                global userlabel_num
                userlabel_num=userlabel_num+1
                print "userlabel_num:",userlabel_num
            else:
                if pred_accuracy_num>(1000000):
                    image_file.save(fold_UserLabel+image_name)
                    img_name_2=image_name.split(".")
                    if os.path.isfile(folder_Annotation_original+img_name_2[0]+'.xml'):
                        shutil.copy(folder_Annotation_original+img_name_2[0]+'.xml',fold_UserLabelAnnotations)
                    global userlabel_num
                    userlabel_num=userlabel_num+1
                    print "userlabel_num:",userlabel_num
                else:
                    names_pred=labelarray.split(" ")
                    coord_pred=coord.split(" ")
                    score_pred=scorearray.split(" ")
                    for i in xrange(len(names_pred)-1):
                        if names_pred[i+1] not in class_name:
                            image_file.save(fold_UserLabel+image_name)
                            img_name_3=image_name.split(".")
                            if os.path.isfile(folder_Annotation_original+img_name_3[0]+'.xml'):
                               shutil.copy(folder_Annotation_original+img_name_3[0]+'.xml',fold_UserLabelAnnotations)
                            global userlabel_num
                            userlabel_num=userlabel_num+1
                            print "userlabel_num:",userlabel_num
                            global wrong_class_num
                            wrong_class_num=wrong_class_num+1
                            print "wrong_class_num:",wrong_class_num
                            break
                        else:
                            for j in xrange(len(names_original)):
                                left_x=float(xmin_orig[j].firstChild.data)
                                left_y=float(ymin_orig[j].firstChild.data)
                                right_x=float(xmax_orig[j].firstChild.data)
                                right_y=float(ymax_orig[j].firstChild.data)
                                if names_pred[i+1]==names_original[j].firstChild.data:
                                    iou=IOU(float(coord_pred[i*4+1]),float(coord_pred[i*4+2]),float(coord_pred[i*4+3]),float(coord_pred[i*4+4]),left_x,left_y,right_x,right_y)
                                    if (iou>0.70):
                                        label_iou=label_iou+" "+names_pred[i+1]
                                        score_iou=score_iou+" "+score_pred[i+1]
                                        coord_iou=coord_iou+" "+coord_pred[i*4+1]+" "+coord_pred[i*4+2]+" "+coord_pred[i*4+3]+" "+coord_pred[i*4+4]
                                    else:
                                        label_iou=label_iou+" "+names_original[j].firstChild.data
                                        score_iou=score_iou+" "+str(1.0)
                                        coord_iou=coord_iou+" "+str(left_x)+" "+str(left_y)+" "+str(right_x)+" "+str(right_y)
                                else:
                                    if names_original[j].firstChild.data in CLASSES:
                                        label_iou=label_iou+" "+names_original[j].firstChild.data
                                        score_iou=score_iou+" "+str(1.0)
                                        coord_iou=coord_iou+" "+str(left_x)+" "+str(left_y)+" "+str(right_x)+" "+str(right_y)
                if len(score_iou)>0:
                    image_file.save(fold_JPGE+image_name)
                    GenerateXml(class_folder,image_name,folder_annotation,width,height,score_iou,label_iou,coord_iou)
                    global pred_accuracy_num
                    pred_accuracy_num=pred_accuracy_num+1
                    print "pred_accuracy_num:",pred_accuracy_num
                else:
                    image_file.save(fold_UserLabel+image_name)
                    img_name_4=image_name.split(".")
                    if os.path.isfile(folder_Annotation_original+img_name_4[0]+'.xml'):
                        shutil.copy(folder_Annotation_original+img_name_4[0]+'.xml',fold_UserLabelAnnotations)
                    global userlabel_num
                    userlabel_num=userlabel_num+1
                    print "userlabel_num:",userlabel_num
                    global wrong_iou_num
                    wrong_iou_num=wrong_iou_num+1
                    print "wrong_iou_num:",wrong_iou_num
        

        else:
            if pred_accuracy_num>(1000000):
                image_file.save(fold_UserLabel+image_name)
                img_name_5=image_name.split(".")
                if os.path.isfile(folder_Annotation_original+img_name_5[0]+'.xml'):
                    shutil.copy(folder_Annotation_original+img_name_5[0]+'.xml',fold_UserLabelAnnotations)
                global userlabel_num
                userlabel_num=userlabel_num+1
                print "userlabel_num:",userlabel_num
            else:
                names_pred=labelarray.split(" ")
                coord_pred=coord.split(" ")
                score_pred=scorearray.split(" ")
                for i in xrange(len(names_pred)-1):
                    if names_pred[i+1] not in class_name:
                        image_file.save(fold_UserLabel+image_name)
                        img_name_6=image_name.split(".")
                        if os.path.isfile(folder_Annotation_original+img_name_6[0]+'.xml'):
                            shutil.copy(folder_Annotation_original+img_name_6[0]+'.xml',fold_UserLabelAnnotations)
                            global userlabel_num
                            userlabel_num=userlabel_num+1
                            print "userlabel_num:",userlabel_num
                            global wrong_class_num
                            wrong_class_num=wrong_class_num+1
                            print "wrong_class_num:",wrong_class_num
                            break
                    else:
                        for j in xrange(len(names_original)):
                            left_x=float(xmin_orig[j].firstChild.data)
                            left_y=float(ymin_orig[j].firstChild.data)
                            right_x=float(xmax_orig[j].firstChild.data)
                            right_y=float(ymax_orig[j].firstChild.data)
                            if names_pred[i+1]==names_original[j].firstChild.data:
                                iou=IOU(float(coord_pred[i*4+1]),float(coord_pred[i*4+2]),float(coord_pred[i*4+3]),float(coord_pred[i*4+4]),left_x,left_y,right_x,right_y)
                                if (iou>0.70):
                                    label_iou=label_iou+" "+names_pred[i+1]
                                    score_iou=score_iou+" "+score_pred[i+1]
                                    coord_iou=coord_iou+" "+coord_pred[i*4+1]+" "+coord_pred[i*4+2]+" "+coord_pred[i*4+3]+" "+coord_pred[i*4+4]
                                else:
                                    label_iou=label_iou+" "+names_original[j].firstChild.data
                                    score_iou=score_iou+" "+str(1.0)
                                    coord_iou=coord_iou+" "+str(left_x)+" "+str(left_y)+" "+str(right_x)+" "+str(right_y)
                            else:
                                if names_original[j].firstChild.data in CLASSES:

                                    label_iou=label_iou+" "+names_original[j].firstChild.data
                                    score_iou=score_iou+" "+str(1.0)
                                    coord_iou=coord_iou+" "+str(left_x)+" "+str(left_y)+" "+str(right_x)+" "+str(right_y)

                if len(score_iou)>0:
                    image_file.save(fold_JPGE+image_name)
                    GenerateXml(class_folder,image_name,folder_annotation,width,height,score_iou,label_iou,coord_iou)
                    global pred_accuracy_num
                    pred_accuracy_num=pred_accuracy_num+1
                    print "pred_accuracy_num:",pred_accuracy_num
                else:
                    image_file.save(fold_UserLabel+image_name)
                    img_name_7=image_name.split(".")
                    if os.path.isfile(folder_Annotation_original+img_name_7[0]+'.xml'):
                        shutil.copy(folder_Annotation_original+img_name_7[0]+'.xml',fold_UserLabelAnnotations)
                    global userlabel_num
                    userlabel_num=userlabel_num+1
                    print "userlabel_num:",userlabel_num
                    global wrong_iou_num
                    wrong_iou_num=wrong_iou_num+1
                    print "wrong_iou_num:",wrong_iou_num
    else:
        image_file.save(fold_UserLabel+image_name)
        img_name_8=image_name.split(".")
        if os.path.isfile(folder_Annotation_original+img_name_8[0]+'.xml'):
            shutil.copy(folder_Annotation_original+img_name_8[0]+'.xml',fold_UserLabelAnnotations)
        global userlabel_num
        userlabel_num=userlabel_num+1
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
# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect == True:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1] 
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

# IOU Part 2
def IOU(xmin_a,ymin_a,xmax_a,ymax_a, xmin_b, ymin_b,xmax_b, ymax_b):
    area_inter = if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b)
    # print ("area_inter:",area_inter)
    if area_inter:
        area_1 = (xmax_a-xmin_a) * (ymax_a-ymin_a)
        area_2 = (xmax_b-xmin_b)*  (ymax_b-ymin_b)
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    else:
        return 0
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
    for cls_ind, cls in enumerate(CLASSES[1:]): 
        class_folder="/home/wangkeze/yanxp/dataset/VOCdevkit00/"+cls+"/"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        shutil.rmtree(class_folder)
        os.makedirs(class_folder)

   
   
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
    global userlabel_num

    folder_image='data/VOCdevkit17/VOC17/JPEGImages/'
    trainval="/home/wangkeze/yanxp/dataset/VOCdevkit17/VOC17/ImageSets/Main/trainval.txt"
    i=0
    for line in open(trainval) :
        line=line.strip('\n')  
        if os.path.isfile(folder_image+line+'.jpg'):
            demo(folder_image,net,line+'.jpg')
            i=i+1
        # if i>(100000):
        #     break


    print "total image:",i
    print "predict accuracy:",pred_accuracy_num
    print "userlabel_num:",userlabel_num
    print "wrong_class_num:",wrong_class_num
    print "wrong_iou_num:",wrong_iou_num

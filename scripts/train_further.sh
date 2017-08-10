python ./tools/train_net_multi_gpu.py --gpu 0,1,2,3 --solver models/pascal_voc/ResNet-101/rfcn_end2end/solver_ohem.prototxt --weights output/rfcn_end2end_ohem/voc_2007_trainval+voc_10_trainval/resnet101_rfcn_voc07_coco_18000_iter_170000.caffemodel --imdb  voc_2007_trainval+voc_10_trainval --iters 110000 --cfg experiments/cfgs/rfcn_end2end_ohem.yml


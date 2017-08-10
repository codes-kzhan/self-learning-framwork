python ./tools/train_net_multi_gpu.py --gpu 0,1,2,3 --solver models/pascal_voc/ResNet-101/rfcn_end2end/solver_ohem.prototxt --weights data/rfcn_models/voc_2012_gt.caffemodel --imdb  voc_2007_trainval+voc_2007_test+voc_2012_trainval+voc_08_trainval --iters 140000 --cfg experiments/cfgs/rfcn_end2end_ohem.yml


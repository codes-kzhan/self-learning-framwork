python ./tools/train_net_multi_gpu.py --gpu 0,1,2,3 --solver models/pascal_voc/ResNet-101/rfcn_end2end/solver_ohem.prototxt --weights output/resume.caffemodel  --imdb  voc_2007_trainval+voc_2012_trainval+voc_coco_trainval --iters 720000 --cfg experiments/cfgs/rfcn_end2end_ohem.yml


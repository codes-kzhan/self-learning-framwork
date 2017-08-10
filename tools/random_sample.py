import os
import math
import random
listname=[]
folder="/home/wangkeze/yanxp/dataset/VOCdevkit08/VOC08/"
annotations="Annotations/"
num = 0
for filename in os.listdir(folder+annotations):
    filename=filename.split("." )
    num = num +1
    listname.append(filename[0])
print "trainval:",num
print "train:",math.ceil(num*0.5)
print "val:",num-math.ceil(num*0.5)
train=random.sample(listname,int(math.ceil(num*0.5)))
file_train=open(folder+'ImageSets/Main/train.txt', 'w')
for i in train:
    file_train.write(i)
    file_train.write("\n")
file_train.close()
val = list(set(listname) - set(train))
file_val=open(folder+'ImageSets/Main/val.txt', 'w')
for i in val:
    file_val.write(i)
    file_val.write("\n")
file_val.close()
file_trainval=open(folder+'ImageSets/Main/trainval.txt','w')
for i in listname:
    file_trainval.write(i)
    file_trainval.write("\n")
file_trainval.close()



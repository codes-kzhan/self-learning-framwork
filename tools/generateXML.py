import xml.dom.minidom
import shutil
import os
  
def GenerateXml(class_folder,imagename,folder_iamge,width_data,height_data,score,labelname,coord):    
  impl = xml.dom.minidom.getDOMImplementation()  
  dom = impl.createDocument(None, 'annotation', None)  
  root = dom.documentElement    
  folder =  dom.createElement('folder')  
  File =  dom.createElement('filename')
  Size =  dom.createElement('size') 
 # Object = dom.createElement('object')  
  root.appendChild(folder)
  root.appendChild(File)
  root.appendChild(Size)
  #root.appendChild(Object)  
    
  foldername=dom.createTextNode('VOC2012')
  filename=dom.createTextNode(imagename)
  
  folder.appendChild(foldername)
  File.appendChild(filename)
 
  width=dom.createElement('width')
  height=dom.createElement('height')
  depth=dom.createElement('depth')
   
  width_str=str(width_data)
  height_str=str(height_data)
#  print width_str,height_str
  widthnum=dom.createTextNode(width_str)
  heightnum=dom.createTextNode(height_str)
  depthnum=dom.createTextNode('3')
 
  width.appendChild(widthnum)
  height.appendChild(heightnum)
  depth.appendChild(depthnum)
  Size.appendChild(width)
  Size.appendChild(height)
  Size.appendChild(depth)
  labelname=labelname.split(" ")
  coord=coord.split(" ")
  score=score.split(" ")
#  print len(labelname) ,len(coord)
#  print labelname
#  print coord
  for i in xrange(len(labelname)-1):
       
       Object = dom.createElement('object')
       root.appendChild(Object) 
      
       nameE=dom.createElement('name')  
       nameT=dom.createTextNode(labelname[i+1])  
       nameE.appendChild(nameT) 
       
       Score= dom.createElement('score')
       score_num=dom.createTextNode(score[i+1])
       Score.appendChild(score_num)

       box=dom.createElement('bndbox') 
       xmin=dom.createElement('xmin') 
       ymin=dom.createElement('ymin')
       xmax=dom.createElement('xmax')
       ymax=dom.createElement('ymax')
 
       xminnum=dom.createTextNode(coord[i*4+1])
       yminnum=dom.createTextNode(coord[i*4+2])   
       xmaxnum=dom.createTextNode(coord[i*4+3])   
       ymaxnum=dom.createTextNode(coord[i*4+4])
  
       xmin.appendChild(xminnum)
       ymin.appendChild(yminnum)
       xmax.appendChild(xmaxnum)
       ymax.appendChild(ymaxnum)
       box.appendChild(xmin)
       box.appendChild(ymin)
       box.appendChild(xmax)
       box.appendChild(ymax)
  
       difficult=dom.createElement('difficult')
       diff_num=dom.createTextNode('0')
       difficult.appendChild(diff_num)
    
       Object.appendChild(nameE)  
       Object.appendChild(Score)
       Object.appendChild(box)
       Object.appendChild(difficult)
  folder_annotation=folder_iamge
  imagename=imagename.split(".")
  f= open(folder_annotation+imagename[0]+'.xml', 'w')  
  dom.writexml(f, addindent='  ', newl='\n')  
  f.close()       
  shutil.copy(folder_annotation+imagename[0]+'.xml',class_folder)
#GenerateXml()


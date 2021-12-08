
import os
import cv2
import glob
import pathlib

#因公共数据集icdar2017rctw中的图片尺寸过大，所以对其做缩小调整，连同标签中的坐标数据一起调整。

data_path = r'E:\\datasets\\icdar2017rctw\\detection'

save_path = r'E:\\datasets\\icdar2017rctw\\detection\\detection_s'

for img_path in glob.glob(data_path + '/imgs/*.jpg', recursive=True):
    d = pathlib.Path(img_path)
    if str(d.stem) == 'image_6089' :
        print(img_path)
    #label_path = os.path.join(data_path,(str(d.stem) + '.txt'))
    label_path = data_path+'/imgs/'+str(d.stem) + '.txt'
    if os.path.exists(img_path) and os.path.exists(label_path):
        print(img_path, label_path)
    else:
        print('不存在', img_path, label_path)
        continue

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    #print('Original Dimensions : ',img.shape)
    scale_percent = 20       # percent of original size
    ratio = scale_percent/100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #save_img_path = os.path.join(save_path, (str(d.stem) + '.jpg'))
    save_img_path = save_path + '/'+str(d.stem) + '.jpg'
    #cv2.imwrite(save_img_path,resized)

    #print('Resized Dimensions : ',resized.shape)

    #save_label_path = os.path.join(save_path +('/'+ str(d.stem) + '.txt'))
    save_label_path = save_path +'/'+ str(d.stem) + '.txt'
    f_w = open(save_label_path, 'w', encoding='utf8')
    try:
        with open(label_path, "r", encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一行元素的换行符
                lineData = line.split(',')
                #list中的str元素转int,然后进行计算
                lineData1 = list(map(int,lineData[0:8]))
                lineData1[0]=(int)(lineData1[0]*ratio)
                lineData1[1]=(int)(lineData1[1]*ratio)
                lineData1[2]=(int)(lineData1[2]*ratio)
                lineData1[3]=(int)(lineData1[3]*ratio)
                lineData1[4]=(int)(lineData1[4]*ratio)
                lineData1[5]=(int)(lineData1[5]*ratio)
                lineData1[6]=(int)(lineData1[6]*ratio)
                lineData1[7]=(int)(lineData1[7]*ratio)

                ptLeftTop = (lineData1[0], lineData1[1])
                ptRightBottom = (lineData1[4], lineData1[5])
                point_color = (0, 255, 0) # BGR
                thickness = 1
                lineType = 4
                #cv2.rectangle(resized, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

                #将list中的int数据转换成str
                num_list = [str(x) for x in lineData1]
                #print(",".join(num_list))

                outline = num_list + lineData[8:]
                outlinestr = ",".join(outline) #list转str
                #print(line)
                #print(outlinestr)
                f_w.write('{}\n'.format(outlinestr))
    except ValueError:
        print('read %s error\n',label_path)
        f_w.close()
        if os.path.exists(label_path):  # 如果文件存在
            # 删除文件，可使用以下两种方法。
            os.remove(label_path)
        continue

    f_w.close()
    cv2.imwrite(save_img_path,resized)
    # cv2.namedWindow("test")
    # cv2.imshow('test', resized)
    # cv2.waitKey(300) # 显示 10000 ms 即 10s 后消失
    # cv2.destroyAllWindows()

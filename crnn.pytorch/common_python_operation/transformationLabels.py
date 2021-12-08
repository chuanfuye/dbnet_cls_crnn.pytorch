import glob
import os
import pathlib
import random

import cv2

# 0、将icdar2015/recognition中的训练数据进行处理
# 1、将train.txt中的数据进行转换，如/home/wt_yechuanfu/work_code/pytorchOCR/icdar2015/train/word_1.png	GenaxisTheatre 转换成word_1.png  
# 2、转换的数据和Synthetic Chinese String Dataset数据集中的train格式一样，便于数据训练；
# 3、并且采用了随机处理，分成train和test数据文件,若早划分好，可以注释
# 4、同时可以输出当前数据集中在之前标准label2.txt中没有的字符。
# 5、此功能可以修改一下可以提取文本中的字符到某个字符串中

data_path = r'/home/wt_yechuanfu/work_code/pytorchOCR/icdar2015'
save_path = r'/home/wt_yechuanfu/work_code/pytorchOCR/icdar2015'
english_label = r'/home/wt_yechuanfu/work_code/crnn.pytorch/common_python_operation/label2.txt'

char_set = open(english_label, 'r', encoding='utf-8').readlines()
std_char_set = ''.join(char_set)
n_class = len(char_set)
# charidx = std_char_set.find('.') 寻找索引

for txt_path in glob.glob(data_path + '/rec_gt_train.txt', recursive=True):
    d = pathlib.Path(txt_path)
    if str(d.stem) == 'image_6089':
        print(txt_path)
    if os.path.exists(txt_path):
        print(txt_path)
    else:
        print('不存在', txt_path)
        continue

    save_label_path = save_path + '/conv_train_' + str(d.stem) + '.txt'

    save_label_path_test = save_path + '/conv_test_' + str(d.stem) + '.txt'
    # cnt = 0
    f_w = open(save_label_path, 'w', encoding='utf8')
    # f_w_test = open(save_label_path_test, 'w', encoding='utf8')
    tmpOtherCharStr = ''    # 用于存储遍历数据中的字符在label标签数据中没有的字符。
    try:
        with open(txt_path, "r", encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一行元素的换行符
                lineData = line.split('\t')
                image = lineData[0]
                img = pathlib.Path(image)
                imagename = str(img.stem)
                imagesuffix = str(img.suffix)
                imgnamewithsuffix = "train/" + imagename + imagesuffix
                # list中的str元素转int,然后进行计算
                idxlist = []  # 空列表
                idxlist.append(imgnamewithsuffix)

                lineDataListLen = len(lineData)
                if lineDataListLen > 3:
                    a=1

                textVal = ''.join(lineData[1])
                if len(textVal) > 20 :
                    continue
                flag=0
                for index, every_char in enumerate(textVal):
                    if every_char ==' ' :
                        every_char = 'ψ' # 将空格换成特殊字符
                    currentcharidx = std_char_set.find(every_char)
                    if currentcharidx == -1 :
                        flag=-1
                        a=1
                        if tmpOtherCharStr.find(every_char) == -1 :
                            tmpOtherCharStr = tmpOtherCharStr + every_char
                            print(every_char) # 打印出label文件中没有的字符
                            #print(tmpOtherCharStr) # 打印出label文件中没有的字符

                    # 将list中的int数据转换成str
                    str_list = str(currentcharidx + 1)
                    idxlist.append(str_list)

                if flag == -1 :
                    continue

                # print(idxlist)
                outlinestr = ' '.join(idxlist)  # list转str
                # if cnt <10000 and random.randint(1,100)%7 == 0 :
                #cnt = cnt +1
                # f_w_test.write('{}\n'.format(outlinestr))
                # else :
                f_w.write('{}\n'.format(outlinestr))
    except ValueError:
        print('read %s error\n', txt_path)
        f_w.close()
        if os.path.exists(txt_path):  # 如果文件存在,删除文件，可使用以下两种方法。
            os.remove(txt_path)
        continue

    f_w.close()
    #f_w_test.close()

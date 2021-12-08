import os
import glob
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cfg
import random
import shutil

if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = cfg.BASE + 'val'
    txtpath = cfg.BASE
    testdata_path = cfg.BASE + "test"
    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(testdata_path, label, '*.bmp'))
        # print(imglist)
        random.shuffle(imglist)
        print(len(imglist))
        # 控制分类数据集比例
        trainlist = imglist[:int(0.6*len(imglist))]
        vallist = imglist[(int(0.6*len(imglist))):]
        testlist = imglist[0:]
        with open(txtpath + 'test.txt', 'a')as f:
            for img in testlist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

#        with open(txtpath + 'val.txt', 'a')as f:
#            for img in vallist:
#                # print(img + ' ' + str(index))
#                f.write(img + ' ' + str(index))
#                f.write('\n')
#    if not os.path.exists(valdata_path):
#        os.mkdir((valdata_path))
#
#    with open(txtpath + 'val.txt', 'r')as f:
#        for line in f:
#            line = line.strip("\n")
#            line = line.split(' ')[0]
#            shutil.move(line, valdata_path)
#        f.close()
#    使用window10替代
#    with open(txtpath + 'val.txt', 'r')as f:
#        for line in f:
#            line = line.strip("\n")
#            line = line.split(' ')[0]
#            line = line.replace('train', 'val')
#            line = line.replace("ok\\", "")
#            line = line.replace("ng\\", "")
#            f.write(line)
#        f.close()
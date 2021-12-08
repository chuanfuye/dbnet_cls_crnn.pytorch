import os
import shutil

print('输入格式：E:\myprojectnew\jupyter\整理文件夹\示例')
## path = input('请键入需要整理的文件夹地址：')
path = "U:/tianma/dataset/真实OK图片/NG"

# new_path = input('请键入要复制到的文件夹地址：')
new_path = "U:/tianma/dataset/train/ok"
j = 1
for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        print(files[i])
        if (files[i][-4:] == 'Jpeg') or (files[i][-3:] == 'bmp') or (files[i][-3:] == 'PNG'):
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + str(j) + '_' + files[i]
            shutil.copy(file_path, new_file_path)
            j += 1
        # yn_close = input('是否退出？')

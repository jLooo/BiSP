import os

# 获取当前目录下的所有文件名
path = 'D:/Datasets/avenue/testing/testing_HR/21/'
file_list = os.listdir(path)

# 过滤出以数字开头的文件名
numbered_files = [filename for filename in file_list if filename[0].isdigit()]

# 按照数字顺序对文件名进行排序
sorted_files = sorted(numbered_files, key=lambda x: int(x.split('.')[0]))

# 重新命名文件
for i, filename in enumerate(sorted_files):
    name, extension = os.path.splitext(filename)
    new_filename = f"{i:06d}{extension}"
    os.rename(path+filename, path+new_filename)
    # print(f"已将文件 {filename} 重命名为 {new_filename}")

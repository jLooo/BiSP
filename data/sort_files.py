import os

def rename_jpg_files(folder_path, new_path, start_number):
    # 获取指定文件夹内所有的 JPG 文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    files.sort()  # 可以按文件名排序，也可以按其他标准排序

    # 重命名文件
    for i, file in enumerate(files, start=int(start_number)):
        # 构建新文件名，格式为六位数字，从指定的数字开始
        new_filename = f"{i:06d}.jpg"
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(new_path, new_filename)

        # 重命名操作
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file}' to '{new_filename}'")

def count_jpg_files(folder_path):
    jpg_count = []
    jpg_count1 = 0
    for root, dirs, files in os.walk(folder_path):
        jpg_count.append(sum(1 for file in files if file.lower().endswith(('.jpg', '.jpeg'))))
        jpg_count1 += sum(1 for file in files if file.lower().endswith(('.jpg', '.jpeg')))
    return jpg_count1

# 使用示例
folder_path = 'D:\Datasets\shanghaitech\\training\\mframes'  # 替换为你的文件夹路径
print(f"Number of JPG files in '{folder_path}': {count_jpg_files(folder_path)}")

# file_name = '06_006.avi'
# # 使用示例
# folder_path = '/home/xust/hjl/MNADv2/dataset/shanghaitech/training/mframes/'+ file_name #+ '/1' # 替换为你的文件夹路径
# new_path = '/home/xust/hjl/MNADv2/dataset/shanghaitech/training/mframes/' + file_name #+ '/1'
# start_number = '000000'  # 替换为你想要开始的数字  # 000137 002000 000000
# rename_jpg_files(folder_path, new_path, start_number)
# Number of JPG files in 'D:\Datasets\shanghaitech\training\frames': [0, 764, 481, 768, 769, 673, 505, 577, 1009, 859, 860, 737, 737, 737, 737, 739, 601, 840, 841, 577, 817, 1009, 481, 1297, 673, 763, 735, 943, 944, 967, 968, 471, 1191, 889, 889, 817, 471, 831, 481, 961, 1167, 877, 877, 877, 457, 601, 1057, 1056, 1057, 804, 805, 961, 961, 1105, 1057, 721, 1009, 733, 733, 733, 577, 889, 745, 1369, 649, 615, 1383, 1383, 1023, 1225, 1047, 1051, 1052, 1039, 1040, 567, 553, 697, 1033, 711, 787, 788, 807, 889, 907, 908, 735, 1431, 361, 450, 300, 300, 600, 325, 275, 225, 575, 250, 200, 325, 400, 350, 525, 425, 275, 575, 912, 913, 865, 975, 327, 425, 1700, 2325, 350, 550, 950, 450, 625, 375, 325, 500, 648, 725, 550, 841, 769, 1273, 601, 625, 1057, 745, 481, 865, 1249, 816, 816, 817, 865, 1417, 553, 808, 808, 809, 1321, 1057, 732, 733, 732, 733, 505, 841, 876, 877, 553, 1201, 769, 1177, 961, 1345, 1020, 1021, 481, 780, 780, 780, 781, 768, 768, 768, 768, 738, 738, 738, 739, 720, 721, 737, 737, 737, 737, 737, 737, 739, 1033, 1068, 1069, 1273, 481, 1033, 721, 763, 764, 1023, 471, 725, 725, 725, 1119, 399, 255, 255, 615, 845, 845, 845, 722, 722, 722, 722, 722, 722, 723, 1143, 567, 1023, 1335, 753, 753, 753, 756, 649, 775, 500, 350, 725, 600, 200, 300, 816, 1297, 768, 769, 769, 784, 784, 785, 1105, 720, 721, 784, 784, 785, 577, 936, 936, 937, 876, 877, 1321, 816, 817, 744, 745, 1081, 1369, 1081, 900, 901, 800, 800, 801, 804, 805, 1153, 1105, 725, 725, 725, 725, 725, 1177, 797, 797, 797, 797, 797, 1417, 1407, 831, 519, 1167, 500, 800, 1575, 750, 1000, 650, 447, 875, 1450, 1825, 1850, 725, 800, 1225, 625, 1100, 775, 550, 1350, 675, 250, 1100, 550, 475, 625, 1800, 525, 425, 750, 725, 1075, 1175, 875, 850, 725, 900, 1396, 1650, 625, 4169, 4119, 1050, 601, 649, 649, 937, 385, 457, 1105]

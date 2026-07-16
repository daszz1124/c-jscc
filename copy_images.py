import os
import shutil
import glob

# 源目录和目标目录
source_dir = '/mnt/d/zsd/project/c-jscc/ConSwinjscc_evaluation/20251010_162944_C16,32,64,96,128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-10-10_19-59-27/model_SwinJSCC_w/_SAandRA_C16,32,64,96,128,192_channel_awgn_snr1,4,7,10,13_size_base_MSE/samples'
target_dir = '/mnt/d/zsd/project/c-jscc/Vis_image_result'

# 搜索包含'distort_004_541_1'的图片文件
search_pattern = os.path.join(source_dir, 'test_*', '*', '*distort_004_541_1*')
image_files = glob.glob(search_pattern)

# 复制找到的文件
for file_path in image_files:
    # 获取完整路径中的目录部分
    file_dir = os.path.dirname(file_path)
    
    # 获取上一级目录（即包含NIGHTS的目录）
    nights_dir = os.path.dirname(file_dir)
    
    # 获取test_*文件夹的名称
    test_folder_name = os.path.basename(nights_dir)
    
    # 在目标目录下创建对应的test_*文件夹
    target_test_dir = os.path.join(target_dir, test_folder_name)
    os.makedirs(target_test_dir, exist_ok=True)
    
    # 获取文件名
    file_name = os.path.basename(file_path)
    
    # 构建目标路径
    target_path = os.path.join(target_test_dir, file_name)
    
    # 复制文件
    shutil.copy2(file_path, target_path)
    print(f'Copied: {file_name} to {target_test_dir}')

print(f'\nTotal files copied: {len(image_files)}')
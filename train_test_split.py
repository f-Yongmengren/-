import os
import config
import shutil

img_root_dir = r'data'
img_name_list = os.listdir(r'data')
for i in range(len(img_name_list)):
    index = int(img_name_list[i].split('.')[0])
    label = index // 100
    data_type = 'train' if index < (label * 100 + 50) else 'test'
    if not os.path.exists(f'process_data/{data_type}/{label}'):
        os.mkdir(f'process_data/{data_type}/{label}')
    shutil.copyfile(os.path.join(img_root_dir, img_name_list[i]), f'process_data/{data_type}/{label}/{img_name_list[i]}')


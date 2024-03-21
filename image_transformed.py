import json
import os
import shutil

import cv2
import albumentations as A

'''
albumentations图片变换的方法：
https://blog.csdn.net/u014297502/article/details/128472186?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-128472186-blog-124682837.235^v36^pc_relevant_default_base3&spm=1001.2101.3001.4242.1&utm_relevant_index=3
'''

def categoryID_not_1_or_2_imgaes_id_mapping(intput_dir):
    # json路径
    coco_json_dir = os.path.join(intput_dir, 'annotations')
    coco_json_path = os.path.join(coco_json_dir, 'annotations.json')

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    category_not_in_1_or_2_images_id = []
    # category_in_1_or_2_images_id_ = [annotation['image_id'] for annotation in annotation_data['annotations'] if not annotation['category_id'] in [1, 2]]
    for annotation in coco_data['annotations']:
        # 获取category_id不是1,和2的image_id
        if not annotation['category_id'] in [1, 2]:
            image_id = annotation['image_id']
            if not image_id in category_not_in_1_or_2_images_id:
                # drop dulplication
                category_not_in_1_or_2_images_id.append(image_id)

    # file_name 与 image_id 映射
    category_not_in_1_or_2_images_id_mapping_names = {image['id']: image['file_name'] for image in coco_data['images']
                                                      if image['id'] in category_not_in_1_or_2_images_id}
    return category_not_in_1_or_2_images_id_mapping_names


def image_augment(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # json路径
    coco_json_dir = os.path.join(input_dir, 'annotations')
    coco_json_path = os.path.join(coco_json_dir, 'annotations.json')

    # 输出json目录
    augmented_coco_json_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(augmented_coco_json_dir, exist_ok=True)

    # 图片目录
    images_dir = os.path.join(input_dir, 'images')

    # 输出图片目录
    output_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)

    # 加载coco_data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 原图片迁移
    for image_name in os.listdir(images_dir):

        # 源文件
        src_file = os.path.join(images_dir, image_name)

        dst_file = os.path.join(output_image_dir, image_name)

        # 判断当前文件是否已存在
        if not os.path.isfile(dst_file):
            # 图片拷贝
            shutil.copy(src_file, output_image_dir)
    print(f'原图片保存到{output_image_dir}目录已完成！！！')
    # 定义增强操作
    augmentations = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ], A.BboxParams(format='coco', label_fields=['category_ids']))

    # 获取类别数量少的图片id和图片名称
    images_id_mapping_names_dict = categoryID_not_1_or_2_imgaes_id_mapping(intput_dir=input_dir)
    for image_info in coco_data['images']:
        if image_info['id'] in list(images_id_mapping_names_dict.keys()):
            image_name = image_info['file_name']
            image_path = os.path.join(images_dir, image_name)
            image_array = cv2.imread(image_path)

            # 获取与当前图像相关联的所有注释
            annotations = [annotation for annotation in coco_data['annotations'] if
                           annotation['image_id'] == image_info['id']]
            bboxes = [annotations['bbox'] for annotations in annotations]
            category_ids = [annotations['category_id'] for annotations in annotations]

            # 在图像和边框上应用增强操作
            try:
                augmented = augmentations(image=image_array, bboxes=bboxes, category_ids=category_ids)
            except Exception as e:
                print('Error:', e)
                continue
            augmented_image = augmented['image']
            augmented_bboxed = augmented['bboxes']

            # 将新的图像信息添加到coco_data['images']
            new_image_info = {
                'file_name': f'augmented_{image_name}',
                'height': augmented_image.shape[0],
                'width': augmented_image.shape[1],
                'id': len(coco_data['images']) + 1
            }
            coco_data['images'].append(new_image_info)

            # 将新的注释信息添加到coco_data['annotations']
            for bbox, category_id in zip(augmented_bboxed, category_ids):
                new_annotation = {
                    'id': len(coco_data['annotations']) + 1,
                    'image_id': new_image_info['id'],
                    'category_id': category_id,
                    'bbox': bbox,
                    'iscrowd': 0
                }
                coco_data['annotations'].append(new_annotation)

            # 保存经过增强操作的图片
            new_image_path = os.path.join(output_image_dir, new_image_info['file_name'])

            # 判断当前文件是否已存在
            if not os.path.isfile(new_image_path):
                # 保存
                cv2.imwrite(new_image_path, augmented_image)
                print(f'增强图片{new_image_path}已保存！！！')
    # 保存新的COCO json文件
    augmented_coco_json_path = os.path.join(augmented_coco_json_dir, 'annotations.json')
    with open(augmented_coco_json_path, 'w') as f:
        # 保存
        json.dump(coco_data, f)
    print(f'新的COCO json文件保存到{augmented_coco_json_path}已完成！！！')


if __name__ == '__main__':
    input_dir = r'E:\SC_search_longfaning\dataset\car_parts\coco_format'
    output_dir = r'E:\SC_search_longfaning\dataset\car_parts\coco_format_augment'
    image_augment(input_dir, output_dir)

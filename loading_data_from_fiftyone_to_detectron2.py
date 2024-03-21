'''
detectron2模型所需要的数据：
file_name
image_id
height
width
annotations
'''
from PIL import Image
import numpy as np
from detectron2.structures import BoxMode

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

category_mapping = {
                    'right_angle_edge_defect': 1,
                    'connection_edge_defect': 2,
                    'burr_defect': 3,
                    'cavity_defect': 4,
                    'huahen': 5,
                    'mosun': 6,
                    'yanse': 7,
                    'jianju': 8,
                    'basi': 9,
                    'chuizhidu': 10}

category_mapping_zh = {
                    '直角边缺陷': 1,
                    '连接处缺陷': 2,
                    '毛刺': 3,
                    '空洞缺陷': 4,
                    '划痕缺陷': 5,
                    '磨损缺陷': 6,
                    '颜色缺陷': 7,
                    '间距缺陷': 8,
                    '拔丝异常': 9,
                    '垂直度问题': 10}
# 图像变换
def apply_augmentations(image, boxes):
    # 创建一个albumentations管道，它将在图像和边界框上应用一系列图片变换
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    transformed = transform(image=image, boxes=boxes)
    return transformed['image'], transformed['bboxes']


def get_data_dicts_from_fiftyone(fiftyone_dataset):
    dataset_dicts = []
    # for sample in fiftyone_dataset:
    #     # 图片信息
    #     file_name = sample.filepath
    #     image_id = sample.id
    #     width,height = sample.metadata.width,sample.metadata.height

    dataset_dicts = []
    for sample in fiftyone_dataset.select_fields(["id", "filepath", "metadata", 'detections']):

        height = sample.metadata["height"]
        width = sample.metadata["width"]
        file_name = sample.filepath
        image_id = sample.id

        # 注释信息
        objs = []
        annotations = sample.detections.detections
        for annotation in annotations:
            tlx, tly, w, h = annotation['bounding_box']
            bbox = [int(tlx * width), int(tly * height), int(w * width), int(h * height)]
            bbox_mode = BoxMode.XYWH_ABS
            category_id = category_mapping_zh[annotation['label']] - 1

            obj = {
                'bbox': bbox,
                'bbox_mode': bbox_mode,
                'category_id': category_id,
                'iscrowd': annotation['iscrowd']
            }
            objs.append(obj)

            # 数据增强
            # 针对训练集
            # if is_train:
            #     # 读取图片
            #     image = np.array(Image.open(file_name))
            #     transformed_image,transformed_boxes = apply_augmentations(image,bbox)
            #     file_name_ = file_name.split('.jpg')[0]
            #     file_name_transformed = f'{file_name_}_augmented.jpg'
            #
            #     # 保存经过变换的图片
            #     # Image.fromarray(transformed_image) --> 将image的array数组转换为PIL对象，然后再使用save()函数将图片保存到磁盘
            #     Image.fromarray(transformed_image).save(file_name_transformed)
            #     objs_transformed = transformed_boxes
            #
            #     objs.append(objs_transformed)
        # 创建最后的记录
        record = {
            'file_name': file_name,
            'image_id': image_id,
            'height': height,
            'width': width,
            'annotations': objs
        }
        dataset_dicts.append(record)
    return dataset_dicts

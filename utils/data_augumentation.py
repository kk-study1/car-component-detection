import torch
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy

def custom_mapper(dataset_dict):
    '''
    作用:
        数据增强
    参数:
        dataset_dict:单张图片及信息
    返回值:
        变换后的单张图片及信息
    '''
    dataset_dict = copy.deepcopy(dataset_dict)  # 不改变原始数据
    image = utils.read_image(dataset_dict["file_name"], format="BGR")  # 获取图片
    transform_list = [
        # T.RandomBrightness(0.8, 1.8),  # 随机改变图像的亮度
        # T.RandomContrast(0.6, 1.3),  # 随机改变图像的亮度
        # T.RandomSaturation(0.8, 1.4),  # 随机改变图像的亮度
        T.RandomRotation(angle=[90, 90],expand=False,sample_style='range'),  # 90°旋转
        # T.RandomLighting(scale=0.7),  # 随机改变图像的亮度
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),  # prob --> probability概率
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False)
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)  # gens --> generations生成器
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # 获得annotations
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    # 将annotations转为instances
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)  # 过滤空实例

    return dataset_dict
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
    '''
    在目标检测和实例分割的任务中，一般会对图像进行各种预处理和数据增强操作，例如缩放、裁剪、旋转等。
    在这些操作之后，有些实例可能完全位于图像的裁剪区域之外，或者大小缩放到无法被检测。这些实例的边界框或者分割掩码就会变为空。
    空的实例在计算损失函数或者评估指标时可能会引发问题，因为它们没有有效的位置信息。
    因此，我们需要在进行训练或者预测之前，先把这些空的实例过滤掉。utils.filter_empty_instances(instances) 这个函数就是用来完成这个任务的。
    '''
    return dataset_dict
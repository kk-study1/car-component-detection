import os

import cv2
import fiftyone as fo
import fiftyone.utils.random as four
import torch
from fiftyone import ViewExpression as E

from datetime import datetime

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor

import loading_data_from_fiftyone_to_detectron2 as loading_data


def main():
    # Define the path to your datasets
    images_dir = r'E:\SC_search_longfaning\dataset\car_parts\coco_format\images'
    annotations_file = r'E:\SC_search_longfaning\dataset\car_parts\coco_format\annotations\annotations.json'

    # 获取当前系统的时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    output_dir = 'output_'+timestamp_str
    # 加载自定义CoCo格式的数据集
    dataset_name = 'test5'
    if dataset_name in fo.list_datasets():
        car_parts_dataset = fo.load_dataset(dataset_name)
    else:
        car_parts_dataset = fo.Dataset.from_dir(
            name=dataset_name,
            data_path=images_dir,
            labels_path=annotations_file,
            dataset_type=fo.types.COCODetectionDataset
        )
    four.random_split(car_parts_dataset, {'train': 0.8, 'val': 0.2})
    # # 按照supercategory来划分数据集
    # supercategoreis = car_parts_dataset.distinct('supercategory')
    # views = {supercategory: car_parts_dataset.match(E('supercategory') == supercategory) for supercategory in
    #          supercategoreis}
    #
    # train_views = []
    # val_views = []
    #
    # # 对每个子集进行随机划分
    # for supercategory, view in views.items():
    #     train_view, val_view = four.random_split(view, {'train': 0.8, 'val': 0.2})
    #     train_views.append(train_view)
    #     val_views.append(val_view)
    #
    # # 在fiftyone创建一个新的数据集
    # new_dataset_name = 'new_dataset'
    # new_dataset = fo.Dataset(name=new_dataset_name)
    #
    # # 合并所有子集的训练集和验证集
    # for view in train_views:
    #     new_dataset.add_sample(view)
    # for view in val_views:
    #     new_dataset.add_sample(view)

    # 将数据加载到detectron2模型中
    for d in ["train", "val"]:
        view = car_parts_dataset.match_tags(d)  # DatasetView
        # 注册数据集
        # 存储和管理数据集
        DatasetCatalog.register("fiftyone_" + d, lambda view=view: loading_data.get_data_dicts_from_fiftyone(view))
        # 存储和管理与数据集相关的元数据（类别信息）
        MetadataCatalog.get("fiftyone_" + d).set(thing_classes=list(loading_data.category_mapping.keys()))

    # # 获取detectron2格式的元数据信息
    # detectron2_frmt_dataset_dicts = loading_data.get_data_dicts_from_fiftyone(car_parts_dataset.match_tags('train'))
    # ids = [dd["image_id"] for dd in detectron2_frmt_dataset_dicts]
    #
    # # 数据视图
    # view = car_parts_dataset.select(ids)

    # # 启动fiftyone
    # session = fo.launch_app(view)

    # 配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # FPN：Region Proposal Network(区域提议网络)
    '''
    RPN是一种全卷积网络（FCN），它可以在任意大小的图像上滑动，产生一系列的矩形框，这些矩形框就是RoI。
    每个RoI不仅有位置信息（即矩形框的坐标），还有一个“objectness”得分，表示这个区域是否包含一个目标对象
    '''
    cfg.DATASETS.TRAIN = ("fiftyone_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2  # 数据加载线程数量
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # 每批次图片的数量
    cfg.SOLVER.BASE_LR = 0.00025  # 学习率
    cfg.SOLVER.MAX_ITER = 900  # 最大迭代次数
    cfg.SOLVER.CHECKPOINT_PERIOD = 300  # 模型保存的周期
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 类别数 (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 516  # ROI：region of interest（感兴趣的区域），这个ROI包括了这个ROI的位置以及是否包含目标对象的得分。对于每一张输入的图像，会切割出大量的ROI。为了提高训练模型的速度，并不是所有的ROI都用来训练模型，因此需要限制ROI的数量
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25  # 设置一张图片中的ROI是正样本的数量的比例

    # 创建训练器
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)  # resume是否从模型的检查点来恢复训练。False：表示不恢复训练，那么模型会加载所有可用的权重，但会忽略那些形状不匹配的权重，那么这就实现了只想加载预训练的主干网络部分
    trainer.train()

    # 创建预测器
    cfg.MODEL.WEIGHTS = './output/model_final.pth'  # 训练好的模型
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 置信度的阈值
    predictor = DefaultPredictor(cfg)

    # 将detectron格式的数据集转为fiftyone格式
    def detectron_to_fo(outputs, img_w, img_h):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        detections = []
        instances = outputs["instances"].to("cpu")
        for pred_box, score, label_numeric in zip(instances.pred_boxes, instances.scores, instances.pred_classes):
            # 获取左对角线两点坐标
            x1, y1, x2, y2 = pred_box

            # bbox归一化
            bbox = [float(x1) / img_w, float(y1) / img_h, float(x2 - x1) / img_w, float(y2 - y1) / img_h]

            # 获取检测的数据（label，confidence，bounding_box）
            detection = fo.Detection(label=list(loading_data.category_mapping.keys())[label_numeric],
                                     confidence=float(score), bounding_box=bbox)
            detections.append(detection)

        return fo.Detections(detections=detections)

    # 获取验证集数据集
    val_view = car_parts_dataset.match_tags("val")

    dataset_dicts = loading_data.get_data_dicts_from_fiftyone(val_view)
    predictions = {}
    for d in dataset_dicts:
        img_w = d["width"]
        img_h = d["height"]
        img = cv2.imread(d["file_name"])

        # 节省内存，不进行梯度计算
        with torch.no_grad():
            # 预测
            outputs = predictor(img)

        # 将预测结果转为fiftyone格式
        detections = detectron_to_fo(outputs, img_w, img_h)

        predictions[d["image_id"]] = detections

    car_parts_dataset.set_values(field_name="predictions", values=predictions, key_field="id")  # 这里的id，为样本集的id

    # 启动fiftyone
    session = fo.launch_app(car_parts_dataset, address="0.0.0.0", port=5151)
    session.wait()


if __name__ == '__main__':
    main()

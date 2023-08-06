import os

import cv2
import fiftyone as fo
import fiftyone.utils.random as four
import torch

from datetime import datetime

import logging

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch, default_setup, default_argument_parser
from detectron2.data import build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from collections import OrderedDict

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils import comm

# 自定义包
import loading_data_from_fiftyone_to_detectron2 as loading_data
from data_augumentation import custom_mapper


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # 通过cfg来获取数据加载器
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluator_type = "coco"
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def load_custom_coco_dataset(dataset_name, images_dir, annotations_file):
    '''
    作用：
        加载自定义CoCo格式的数据集
    参数：
        dataset_name：数据集的名称
        images_dir：图片目录
        annotations_file：注释文件路径
    返回值：
        fiftyone格式的数据集
    '''

    if dataset_name in fo.list_datasets():
        car_parts_dataset = fo.load_dataset(dataset_name)
    else:
        car_parts_dataset = fo.Dataset.from_dir(
            name=dataset_name,
            data_path=images_dir,
            labels_path=annotations_file,
            dataset_type=fo.types.COCODetectionDataset
        )
    return car_parts_dataset


def register_dataset(car_parts_dataset):
    """
    作用：
        将数据加载到detectron2模块中
    参数：
         car_parts_dataset：fiftyone格式的数据集
    返回值：
        无
    """
    for d in ["train", "test"]:
        view = car_parts_dataset.match_tags(d)  # DatasetView
        # 注册数据集
        # 存储和管理数据集
        DatasetCatalog.register("fiftyone_" + d, lambda view=view: loading_data.get_data_dicts_from_fiftyone(view))
        # 存储和管理与数据集相关的元数据（类别信息）
        MetadataCatalog.get("fiftyone_" + d).set(thing_classes=list(loading_data.category_mapping.keys()))


def setup_hyper_parameter(output_dir, train_imgs_count):
    # 详细参数见：https://codeleading.com/article/14733923369/
    # https://detectron2.readthedocs.io/en/latest/modules/config.html
    """
    作用：
        设置超参数
    参数：
        output_dir：模型及日志输出目录
        train_imgs_count：训练集的图片数量
    返回值：
        无
    """

    cfg = get_cfg()  # 获得配置信息
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # 合并主干网络  # FPN：Region Proposal Network(区域提议网络)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # 预训练模型权重
    '''
    RPN是一种全卷积网络（FCN），它可以在任意大小的图像上滑动，产生一系列的矩形框，这些矩形框就是RoI。
    每个RoI不仅有位置信息（即矩形框的坐标），还有一个“objectness”得分，表示这个区域是否包含一个目标对象
    '''

    # 多尺寸训练
    # cfg.INPUT.CROP.ENABLED = True  # 开启中心点随机剪裁数据增强
    # cfg.INPUT.MAX_SIZE_TRAIN = 4050  # 训练图片输入的最大尺寸
    # cfg.INPUT.MIN_SIZE_TRAIN = (826, 900)  # 训练图片输入的最小尺寸，可以指定为多尺度训练
    # cfg.INPUT.MAX_SIZE_TEST = 4050  # 测试数据输入的最大尺寸
    # cfg.INPUT.MIN_SIZE_TEST = (826, 900)
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("fiftyone_train",)
    cfg.DATASETS.TEST = ("fiftyone_test",)
    cfg.DATALOADER.NUM_WORKERS = 2  # 数据加载线程数量

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 类别数（因为有background） (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # 每批次图片的数量
    cfg.SOLVER.IMS_PER_BATCH = 48

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    ITERS_IN_ONE_EPOCH = int(
        train_imgs_count / cfg.SOLVER.IMS_PER_BATCH)  # iters_in_one_epoch = dataset_imgs/batch_size

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 100) - 1  # 100 epochs

    # 初始化学习率
    cfg.SOLVER.BASE_LR = 0.0025

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 516  # ROI：region of interest（感兴趣的区域），这个ROI包括了这个ROI的位置以及是否包含目标对象的得分。对于每一张输入的图像，会切割出大量的ROI。为了提高训练模型的速度，并不是所有的ROI都用来训练模型，因此需要限制ROI的数量
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25  # 设置一张图片中的ROI是正样本的数量的比例

    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9

    # 权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1

    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)

    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"

    # 模型保存的周期
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH * 20 - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH

    # 设置模型及其日志信息保存目录
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 冻结配置信息，将配置信息设置为只读
    cfg.freeze()

    return cfg


def detectron_to_fiftyone(outputs, img_w, img_h):
    '''
    作用：
        将detectron2格式的预测结果转为fiftyone格式
    参数：
        outputs：预测结果
        img_w：图片宽度
        img_h：图片高度
    返回值：
        fiftyone格式的注释信息
    '''
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


def main():
    images_dir = r'E:\SC_search_longfaning\car-component-detect\dataset\car_parts\coco_format\images'
    annotations_file = r'E:\SC_search_longfaning\car-component-detect\dataset\car_parts\coco_format\annotations\annotations.json'

    # 获取当前系统的时间
    now = datetime.now()
    # 将当前时间格式化为字符串
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    output_dir = 'output_' + timestamp_str

    # 加载自定义CoCo格式的数据集
    dataset_name = 'test10'
    car_parts_dataset = load_custom_coco_dataset(dataset_name, images_dir, annotations_file)

    # 划分数据集
    four.random_split(car_parts_dataset, {'train': 0.7, 'test': 0.2, 'val': 0.1})

    # 将数据加载到detectron2模型中
    register_dataset(car_parts_dataset)

    if 0:
        # 获取detectron2格式的元数据信息
        detectron2_frmt_dataset_dicts = loading_data.get_data_dicts_from_fiftyone(car_parts_dataset.match_tags('train'))
        ids = [dd["image_id"] for dd in detectron2_frmt_dataset_dicts]

        # 数据视图
        view = car_parts_dataset.select(ids)

        # 启动fiftyone
        session = fo.launch_app(view)
        session.wait()

    if 1:
        # 设置超参数
        dataset_imgs_count = len(car_parts_dataset.match_tags("train"))  # 训练集的样本总数
        cfg = setup_hyper_parameter(output_dir, dataset_imgs_count)

        # 创建训练器
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(
            resume=False)  # resume是否从模型的检查点来恢复训练。False：表示不恢复训练，那么模型会加载所有可用的权重，但会忽略那些形状不匹配的权重，那么这就实现了只想加载预训练的主干网络部分
        trainer.train()

        # 创建预测器
        cfg.MODEL.WEIGHTS = f'./{output_dir}/model_final.pth'  # 训练好的模型
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 置信度的阈值
        predictor = DefaultPredictor(cfg)

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
            detections = detectron_to_fiftyone(outputs, img_w, img_h)

            predictions[d["image_id"]] = detections

        car_parts_dataset.set_values(field_name="predictions", values=predictions, key_field="id")  # 这里的id，为样本集的id

        # 启动fiftyone
        session = fo.launch_app(car_parts_dataset, address="0.0.0.0", port=5151)
        session.wait()


if __name__ == '__main__':
    main()

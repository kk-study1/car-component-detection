import argparse
import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import fiftyone as fo
import fiftyone.utils.random as four
from fiftyone import ViewExpression as E
import loading_data_from_fiftyone_to_detectron2 as loading_data


# 模型加载
def load_model(config_file, model_weights):
    cfg = get_cfg()  # CfgNode
    # config参数详解：https://detectron2.readthedocs.io/en/latest/modules/config.html
    cfg.merge_from_file(config_file)  # 合并（加载）模型的配置文件（模型结构、优化器设置...）

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # ROI的标签数量
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置ROI的得分的阈值
    cfg.MODEL.DEVICE = 'cpu'

    predictor = DefaultPredictor(cfg)
    return predictor


# 模型推理
def inference(predictor, image_path):
    image_path = image_path.replace('\\', '\\\\')
    img = cv2.imread(image_path)
    print(image_path)
    img_h, img_w, _ = img.shape
    outputs = predictor(img)
    return outputs  # 在推理模式下，内置模型输出一个list[dict]，每个字典可能包含以下字段。https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=DefaultPredictor#model-output-format


# 将推理结果转为fiftyone格式
def detectron_to_fo(outputs, img_w, img_h):
    # 一张图片有多个缺陷
    instances = outputs['instances'].to('cpu')
    detections = []
    for pre_box, socre, label in zip(instances.pred_boxes, instances.scores,
                                     instances.pred_classes):  # pred_classes --> 列别索引：[0, num_categories)
        x1, y1, x2, y2 = pre_box
        bbox = [float(x1) / img_w, float(y1) / img_h, float(x2 - x1) / img_w, float(y2 - y1) / img_h]
        detection = fo.Detection(label=list(loading_data.category_mapping_zh.keys())[label], confidence=float(socre),
                                 bounding_box=bbox)  # label

        detections.append(detection)
    return fo.Detections(detections=detections)


# 验证操作
def evaluate(args, predictor):
    # 加载自定义CoCo格式的数据集
    dataset_name = 'car_components'
    if dataset_name in fo.list_datasets():
        car_parts_dataset = fo.load_dataset(dataset_name)
    else:
        car_parts_dataset = fo.Dataset.from_dir(
            name=dataset_name,
            data_path=args.images_dir,
            labels_path=args.annotations_file,
            dataset_type=fo.types.COCODetectionDataset
        )

    # # 按照supercategory来划分数据集
    # supercategoreis = car_parts_dataset.distinct('supercategoreis')
    # views = {supercategory: car_parts_dataset.match(E('supercategoreis') == supercategory) for supercategory in
    #          supercategoreis}
    #
    # train_views = []
    # val_views = []
    #
    # # 对每个子集进行随机划分
    # for supercategory, view in views.items():
    #     train_view, val_view = four.random_split(sample_collection=view,
    #                                              split_fracs={'train': 0.7, 'test': 0.2, 'val': 0.1})
    #     train_views.append(train_view)
    #     val_views.append(val_view)
    #
    # # 在fiftyone创建一个新的数据集
    # new_dataset_name = 'new_dataset'
    # new_dataset = fo.Dataset(new_dataset_name)
    #
    # # 合并所有子集的训练集和验证集
    # for view in train_views:
    #     new_dataset.add_sample(view)
    # for view in val_views:
    #     new_dataset.add_sample(view)

    # 划分数据集
    four.random_split(car_parts_dataset, {'train': 0.75, 'test': 0.15, 'val': 0.1}, seed=100)

    # 拿到val数据集
    val_view = car_parts_dataset.match_tags("val")
    dataset_dicts = loading_data.get_data_dicts_from_fiftyone(val_view)

    predictions = {}
    # 遍历
    for data in dataset_dicts:
        img_w = data["width"]
        img_h = data["height"]
        img_path = data['file_name']

        # 推理
        outputs = inference(predictor, img_path)

        # 将推理结果转为fiftyone格式
        detections = detectron_to_fo(outputs, img_w, img_h)
        predictions[data['image_id']] = detections

    # 在fiftyone的json文件内容追加predictions字段
    car_parts_dataset.set_values(field_name="predictions", values=predictions,
                                 key_field="id")  # key_field="id" 指定了字典的键对应的是fiftyone的数据集样本的 id 字段
    return car_parts_dataset


# 汽车零件检测
# 一张一张图片进行检测
def detect(args, predictor):
    # 模型推理
    outputs = inference(predictor, args.image)

    img = cv2.imread(args.image)
    img_h, img_w = img.shape[:2]

    # 将推理结果转为fiftyone格式
    detections = detectron_to_fo(outputs, img_w, img_h)

    # 创建Sample对象
    sample = fo.Sample(filepath=args.image)

    sample["detections"] = detections
    dataset_name = "car_parts_detection6"
    if dataset_name in fo.list_datasets():
        # 如果存在dataset_name，直接加载就好
        car_parts_dataset = fo.load_dataset(dataset_name)
    else:
        # 创建数据集
        car_parts_dataset = fo.Dataset(name=dataset_name)
    # 将包含图像和检测结果的sample对象添加到数据集中
    car_parts_dataset.add_sample(sample)  # type(sample) --> class:`fiftyone.core.sample.Sample
    return car_parts_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detectron2 Inference and FiftyOne Visualization")
    parser.add_argument('--config', type=str, default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='Path to the config file')
    parser.add_argument('--weights', type=str, default=r'./weight/model_0012997.pth',
                        help='Path to the model weights')
    parser.add_argument('--dataset_name', type=str, default='car_components', help='datasets name to the fiftyone')
    parser.add_argument('--operation', type=str, default='evaluate', help='operation choose evaluate or detect')
    parser.add_argument('--image', type=str, default='', help='Path to the image')
    parser.add_argument('--images_dir', type=str,
                        default=r'E:\SC_search_longfaning\car-component-detect\dataset\car_parts\coco_format_server\images',
                        help='Directory to the images')
    parser.add_argument('--annotations_file', type=str,
                        default=r'E:\SC_search_longfaning\car-component-detect\dataset\car_parts\coco_format_server\annotations\annotations.json',
                        help='Path to the annotations_file')

    args = parser.parse_args()
    config_file = model_zoo.get_config_file(args.config)
    # model_weights = model_zoo.get_checkpoint_url(args.weights)
    model_weights = args.weights
    # 实例化对象
    predictor = load_model(config_file, model_weights)

    if args.operation == 'evaluate':
        # 评估模型效果
        car_parts_dataset = evaluate(args, predictor)
    elif args.operation == 'detect':
        for image_name in os.listdir(args.images_dir):
            image_path = os.path.join(args.images_dir, image_name)
            args.image = image_path
            # 一张一张进行检测
            car_parts_dataset = detect(args, predictor)

    # 评估模型的质量
    car_parts_dataset.evaluate_detections(pred_field='predictions', gt_field='detections',
                                          eval_key="eval")  # pred_field 模型预测结果的字段，gt_field真实标签字段，eval_key评估结果保存的字段

    # 向sample的tags字段添加内容
    for sample in car_parts_dataset.iter_samples(autosave=True, batch_size=100):
        for detection in sample.detections.detections:
            detection['tags'].append(detection['supercategory'])

    # 导出数据
    export_dir = "datasets/car_parts_dataset"
    car_parts_dataset.export(export_dir=export_dir, dataset_type=fo.types.FiftyOneDataset)

    # 启动程序
    session = fo.launch_app(car_parts_dataset)
    # session.view = car_parts_dataset.to_evaluation_patches("eval")
    session.wait()

import argparse
import gradio as gr
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

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
def inference(predictor, input_image):
    outputs = predictor(input_image)
    return outputs  # 在推理模式下，内置模型输出一个list[dict]，每个字典可能包含以下字段。https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=DefaultPredictor#model-output-format


def show_pred_image(input_image):
    outputs = inference(predictor, input_image)
    instances = outputs['instances'].to('cpu')
    if instances.has("pred_boxes") and instances.has("scores") and instances.has("pred_classes"):
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()

        # 将OpenCV图像转换为PIL图像
        input_image_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(input_image_pil)

        font = ImageFont.truetype("font/simsun.ttc", size=30)

        for pre_box, score, label in zip(pred_boxes, scores, pred_classes):
            x1, y1, x2, y2 = map(int, pre_box)
            label_text = list(loading_data.category_mapping_zh.keys())[label] + ' ' + str(score)
            # 添加矩形框
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=5)
            # 添加中文标签
            draw.text((x1, y1 - 30), label_text, fill=(0, 255, 0), font=font)

            # 将PIL图像转回OpenCV图像
        return cv2.cvtColor(np.array(input_image_pil), cv2.COLOR_RGB2BGR)
    else:
        return input_image


def show_pred_video(input_video):
    cap = cv2.VideoCapture(input_video)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = inference(predictor, frame)
            results = outputs[0].cpu().numpy()
            for i, det in enumerate(results.boxes.xyxy):
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)


parser = argparse.ArgumentParser(description="Detectron2 Inference and FiftyOne Visualization")
parser.add_argument('--config', type=str, default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                    help='Path to the config file')
parser.add_argument('--weights', type=str, default=r'weight/model_0012997.pth',
                    help='Path to the model weights')

args = parser.parse_args()
config_file = model_zoo.get_config_file(args.config)
model_weights = args.weights

# 实例化对象
predictor = load_model(config_file, model_weights)

# app
with gr.Blocks() as demo:
    # 使用HTML组件创建居中的二级标题
    gr.HTML("<h2 style='text-align: center;'>汽车零部件缺陷检测平台</h2>")
    with gr.Tabs():
        with gr.Tab("图像处理"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="输入图像")

                with gr.Column():
                    output_image = gr.Image(label="输出图像")

            # 底部按钮，用于触发图像处理
            process_button = gr.Button("开始处理")

            # 当按钮被点击时，调用process_image函数
            process_button.click(fn=show_pred_image, inputs=input_image, outputs=output_image)

demo.launch()

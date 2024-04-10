import json
import os
import shutil
import base64

'''
{
	'images': [{
		'file_name': '2020-01-11_21_36_02_642.jpg',
		'height': 1288,
		'width': 882,
		'id': 1
	}],
	'annotations': [{
		'id': 1,
		'image_id': 1,
		'category_id': 1,
		'bbox': [1201, 621, 32, 70],
		'area': 2240,
		'iscrowd': 0
	}, {
		'id': 2,
		'image_id': 1,
		'category_id': 2,
		'bbox': [1099, 455, 23, 85],
		'area': 1955,
		'iscrowd': 0
	}, {
		'id': 3,
		'image_id': 1,
		'category_id': 2,
		'bbox': [1084, 789, -13, -27],
		'area': 351,
		'iscrowd': 0
	}],
	'categories': [{
		'id': 1,
		'name': 'right_angle_edge_defect'
		'supercategory':'bearing_interface'
	}, {
		'id': 2,
		'name': 'connection_edge_defect'
		'supercategory':'bearing_interface'
	}]
}
'''


def convert_to_coco_format(input_dir, output_dir):
    # 获取图片和注释的url
    images_dir = os.path.join(input_dir, '../images')
    annotations_dir = os.path.join(input_dir, 'Annotations')

    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id = 1
    annotations_id = 1

    labels = {}
    label_count = 0

    # test = {}  # 存储图片名称

    # 轴承
    bearing_interface = ['connection_edge_defect', 'right_angle_edge_defect', 'cavity_defect', 'burr_defect']
    # 火花塞
    spark_plug = ['chuizhidu', 'basi', 'jianju']
    # 摇把
    car_handle = ['yanse', 'huahen', 'mosun']

    # 遍历annotations_dir的文件
    for filename in os.listdir(annotations_dir):
        print('filename：', filename)
        if not filename.endswith('.json'):
            continue

        # 加载相应的注释文件
        with open(os.path.join(annotations_dir, filename), 'r') as f:
            annotation_data = json.load(f)

        # 获取图片的尺寸
        width, height = annotation_data['imageWidth'], annotation_data['imageHeight']  # 882 1288

        # 添加图片信息
        file_name = annotation_data['imagePath'].split('\\')[-1]  # ..\\×××\\2020-03-07_05_37_28_083.jpg -->2020_03_07_05_37_28_083.jpg
        file_name = str(image_id) + '_' + file_name.replace('-','_')  # 2020-03-07_05_37_28_083.jpg --> 2020_03_07_05_37_28_083.jpg

        # try:
        #     test[file_name].append(filename)
        # except:
        #     test[file_name] = [filename]

        # 从base64编码字符串中提取图片
        image_data = base64.b64decode(annotation_data["imageData"])
        image_catalog = os.path.join(output_dir, '../images')

        # 建立图片目录
        os.makedirs(image_catalog, exist_ok=True)

        image_path = image_catalog + '/' + file_name  # 图片路径

        # 判断image_path是否已存在图片
        if not os.path.isfile(image_path):
            # 保存图片
            with open(image_path, 'wb') as f:
                f.write(image_data)

        # 添加图片信息
        coco_annotations['images'].append({
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        })

        # 添加注释信息
        for shape in annotation_data['shapes']:
            if shape['shape_type'] == 'rectangle':

                # label映射
                if not labels.get(shape['label']):
                    label_count = label_count + 1
                    labels[shape['label']] = label_count

                # 坐标
                try:
                    xmin, ymin = shape['points'][0]
                    xmax, ymax = shape['points'][1]

                    # 边框 [x, y, width, height]
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                except:
                    continue
                obj = {
                    "id": annotations_id,
                    "image_id": image_id,
                    "category_id": labels[shape['label']],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                coco_annotations['annotations'].append(obj)
                annotations_id = annotations_id + 1

        # print(coco_annotations)
        # break
        image_id = image_id + 1

    # 添加类别内容
    for name, id in labels.items():
        if name in bearing_interface:
            super_category = 'bearing_interface'
        elif name in spark_plug:
            super_category = 'spark_plug'
        elif name in car_handle:
            super_category = 'car_handle'
        coco_annotations['categories'].append({"id": id, "name": name, 'supercategory': super_category})

    # 为了适配detectron2框架，这里需要给categories字段添加"category_id:0 name:background supercategory:''"
    # coco_annotations['categories'].append({"id": 0, "name": "background", 'supercategory': ''})
    print(coco_annotations)

    # 保存json文件
    json_catalog = os.path.join(output_dir, 'annotations')
    os.makedirs(json_catalog, exist_ok=True)  # exist_ok=True：如果output_dir存在则不执行

    # 判断是否annotations.json存在
    if not os.path.isfile(os.path.join(json_catalog, "annotations.json")):
        with open(os.path.join(json_catalog, "annotations.json"), "w") as f:
            json.dump(coco_annotations, f)
            print('json文件保存成功')

    # 图片拷贝
    # os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    # for filename in images_file_name:
    #
    #     # 源文件
    #     src_file = os.path.join(images_dir, filename)
    #
    #     # 修改filename
    #     filename_rename = filename.replace('-', '_')  # 2020-01-11_21_36_02_642.jpg --> 2020_01_11_21_36_02_642.jpg
    #     src_file_rename = images_dir + '/' + filename_rename
    #
    #     # 图片重命名
    #     os.rename(src_file, src_file_rename)
    #
    #     # 目标目录
    #     dst_catalog = output_dir + '/' + 'images'
    #
    #     # 检测目标文件是否存在
    #     dst_file = os.path.join(dst_catalog, filename_rename)
    #     if os.path.isfile(src_file) and not os.path.isfile(dst_file):
    #         # 拷贝
    #         shutil.copy(src_file_rename, dst_catalog)
    # print('图片拷贝完成')

    # test_total = 0
    # for key,value in test.items():
    #     if len(value) >1:
    #         print(f'{key}：{test[key]}')
    #         test_total = test_total +len(test[key])
    #     # else:print(f'{key}：{test[key]}')
    # print('test_total：',test_total)


if __name__ == '__main__':
    input_dir = r'C:\Users\ASUS\Desktop\test'
    output_dir = r'C:\Users\ASUS\Desktop\test\coco_format'
    convert_to_coco_format(input_dir, output_dir)

import fiftyone as fo
import fiftyone.utils.random as four

image_dir = r'E:\SC_search_longfaning\car-component-detect\dataset\car_surface\img'
annotations_path = r'E:\SC_search_longfaning\car-component-detect\dataset\car_surface\anno\val_.json'

dataset_name = 'car_surface'
if dataset_name in fo.list_datasets():
    car_parts_dataset = fo.load_dataset(dataset_name)
else:
     car_parts_dataset=fo.Dataset.from_dir(
        name=dataset_name,
        data_path=image_dir,
        labels_path=annotations_path,
        dataset_type=fo.types.COCODetectionDataset
    )


# 向sample的tags字段添加内容
for sample in car_parts_dataset.iter_samples(autosave=True, batch_size=100):
    for detection in sample.detections.detections:
        detection['tags'].append(detection['supercategory'])

session = fo.launch_app(car_parts_dataset)
session.wait()


import fiftyone as fo
import os

datasets_dir = "./datasets"
datasets = {}  # 存储数据
for dataset_name in os.listdir(datasets_dir):
    if dataset_name in fo.list_datasets():
        datasets[dataset_name] = fo.load_dataset(dataset_name)
    dataset_dir = datasets_dir + '/' + dataset_name
    print(dataset_dir)
    datasets[dataset_name] = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name
    )

session = fo.launch_app(datasets["car_parts_dataset"])
session.wait()

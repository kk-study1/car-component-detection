import albumentations as A
import cv2

# 增强变换
transforms = A.Compose([
    A.VerticalFlip(p=1),  # 垂直翻转
    A.HorizontalFlip(p=1),  # 水平翻转
    A.RandomContrast(limit=0.2, p=1),  # 随机对比度
    A.GaussNoise(var_limit=(40, 90), p=1),  # 添加高斯噪声
    A.RandomBrightness(limit=0.2, p=1)  # 亮度增强
])

# 读取图像
image_path = r'../images/1099_5201327_166.jpg'
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
for transform in transforms:
    transform_str = str(transform)
    augmented_image = transform(image=image_rgb)['image']
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'../images/1099_5201327_166_{transform_str.split("(")[0]}.jpg', augmented_image_bgr)
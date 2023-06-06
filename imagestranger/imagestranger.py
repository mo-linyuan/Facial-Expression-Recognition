

from PIL import Image
import numpy as np

# 读取图像
image = Image.open('../assets/image.png')
image = np.expand_dims(image, axis=-1)

# 将图像转换为NumPy数组
image = np.array(image)
# 扩展图像数组的维度,
image = np.expand_dims(image, axis=0)
#打印图像数组的形状
print(image.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 创建ImageDataGenerator实例
datagen = ImageDataGenerator(
 zoom_range=0.2)

# 创建图像生成器
image_generator = datagen.flow(image, batch_size=1,
                               save_to_dir='',
                               save_prefix='aug',
                               save_format='png')

# 获取增强后的图像
for i in range(6):
    augmented_image = next(image_generator)

print('Image generated and saved successfully')

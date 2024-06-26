# 如何训练和测试NeRF模型
## 简介
本项目实现了一个基于NeRF（Neural Radiance Fields）的3D重建和新视图合成模型。通过多视角图像数据进行训练，生成高质量的3D重建和新视图。

## 环境配置
在开始训练和测试之前，请确保你的环境中安装了以下依赖：

    Python 3.x
    TensorFlow 2.x
    Keras
    NumPy
    Matplotlib
    ImageIO

## 数据准备
请将你的多视角图像数据和相机参数文件放在指定的目录下。确保图像文件和相机参数文件（如 transforms.json）在相同的目录中。

## 训练模型
使用以下命令运行训练脚本：

### 设置环境变量
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

### 导入必要的库
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    import json

### 设定随机种子以确保结果可重复
tf.random.set_seed(42)

### 初始化全局变量
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 10
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 200

### 定义和加载数据、模型的函数...

### 加载并预处理数据
data = load_blender_data(basedir='local_images')
images = data["images"]
poses = data["poses"]
focal = data["focal"]

### 划分训练集和验证集
split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]
train_poses = poses[:split_index]
val_poses = poses[split_index:]

### 创建训练和验证数据集
train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
train_ds = (
    tf.data.Dataset.zip((train_img_ds, train_ray_ds))
    .shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=AUTO)
val_ds = (
    tf.data.Dataset.zip((val_img_ds, val_ray_ds))
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

### 构建和编译模型
num_pos = train_images.shape[1] * train_images.shape[2] * NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)
model = NeRF(nerf_model)
model.compile(optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError())

### 训练模型
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[TrainMonitor()],
)

### 保存模型
model.save('nerf_model')
测试模型
要测试训练好的模型，请使用以下脚本：

### 加载模型
model = keras.models.load_model('nerf_model', compile=False)
nerf_model = model.nerf_model

### 渲染测试图像和深度图
test_recons_images, depth_maps = render_rgb_depth(
    model=nerf_model,
    rays_flat=test_rays_flat,
    t_vals=test_t_vals,
    rand=True,
    train=False,
)

### 创建子图并展示原始图像、重建图像和深度图
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

for ax, ori_img, recons_img, depth_map in zip(axes, test_imgs, test_recons_images, depth_maps):
    ax[0].imshow(keras.utils.array_to_img(ori_img))
    ax[0].set_title("Original")

    ax[1].imshow(keras.utils.array_to_img(recons_img))
    ax[1].set_title("Reconstructed")

    ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]), cmap="inferno")
    ax[2].set_title("Depth Map")

plt.show()

## 权重下载
你可以从以下链接下载训练好的模型权重：

数据预处理和训练代码：https://github.com/Non-Physics/Final_mission3
渲染视频：https://pan.baidu.com/s/1pDdfiK7g_K6hCejB8ZZJHw?pwd=e83s
提取码：e83s 
模型权重：https://pan.baidu.com/s/14SuQFoH0lzF3KzYNdTxx_A?pwd=n2ke
提取码：n2ke 
图片链接：https://pan.baidu.com/s/1IBqjFe_GlBPkg8yvOjSBCQ?pwd=f65v
提取码：f65v 
相机参数：https://pan.baidu.com/s/1yx1mXIS1DnMDMGotNF__Dw?pwd=laqx
提取码：laqx 

请确保所有数据和权重文件路径正确设置。通过上述步骤，你可以成功训练和测试NeRF模型，实现3D重建和新视图合成。

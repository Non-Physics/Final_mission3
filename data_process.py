import json
import os

import cv2
import numpy as np


def load_images(image_dir, suffix):
    images = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.endswith(suffix):
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path)
            images.append((file_name, image))
    return images


def load_camera_params(camera_dir):
    camera_params = {}
    for file_name in sorted(os.listdir(camera_dir)):
        if file_name.endswith('_cam.txt'):
            with open(os.path.join(camera_dir, file_name), 'r') as f:
                data = f.readlines()
                extrinsic = [list(map(float, line.strip().split())) for line in data[1:5]]
                intrinsic = [list(map(float, line.strip().split())) for line in data[7:10]]
                camera_params[file_name.split('_')[0]] = {'extrinsic': extrinsic, 'intrinsic': intrinsic}
    return camera_params

def calculate_camera_angle_x(intrinsic):
    f_x = intrinsic[0, 0]
    image_width = 2 * intrinsic[0, 2]  # Assuming c_x is half the image width
    camera_angle_x = 2 * np.arctan(image_width / (2 * f_x))
    return camera_angle_x


def pool_image(image, pool_size):
    compressed_size = image.shape[0] // pool_size, image.shape[1] // pool_size, image.shape[2]
    compressed_image = np.zeros(compressed_size, dtype=np.uint8)
    for i in range(compressed_size[0]):
        for j in range(compressed_size[1]):
            for k in range(compressed_size[2]):
                compressed_image[i, j, k] = np.mean(
                    image[i * pool_size: (i + 1) * pool_size, j * pool_size: (j + 1) * pool_size, k])
    return compressed_image


def prepare_nerf_data(image_dir, camera_dir, output_file, pool_size):
    images = load_images(image_dir, '_masked.jpg')
    camera_params = load_camera_params(camera_dir)

    transform_data = []
    for file_name, image in images:
        idx = file_name.split('_')[0]

        if idx in camera_params:
            extrinsic = camera_params[idx]['extrinsic']
            intrinsic = camera_params[idx]['intrinsic']

            # 对图像进行池化操作
            image = pool_image(image, pool_size)

            # 调整内参参数
            H, W = image.shape[0], image.shape[1]
            focal = intrinsic[0][0] / pool_size  # 仅调整焦距
            camera_angle_x = np.arctan(W / (2 * focal)) * 2
            
            transform_data.append({
                'file_path': f"{idx}_masked.jpg",  # 原始文件名，这里仅用于标识
                "camera_angle_x": camera_angle_x,
                'H': H,
                'W': W,
                'focal': focal,
                'extrinsic': extrinsic
            })

    with open(output_file, 'w') as f:
        json.dump(transform_data, f, indent=4)


if __name__ == '__main__':
    # 修改为适合WSL的路径格式
    image_dir = '/mnt/d/复旦/研究生/研一下/神经网络与深度学习/Final/nerf_project/images'
    camera_dir = '/mnt/d/复旦/研究生/研一下/神经网络与深度学习/Final/nerf_project/camera_params'
    output_file = '/mnt/d/复旦/研究生/研一下/神经网络与深度学习/Final/nerf_project/output/transforms.json'

    pool_size = 5  # 定义池化大小

    prepare_nerf_data(image_dir, camera_dir, output_file, pool_size)

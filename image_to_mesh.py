"""
图像到3D网格转换工具
使用Marching Cubes算法将PNG图像序列重建为3D mesh模型
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure, morphology
from scipy import ndimage
import trimesh
from tqdm import tqdm


class ImageToMeshConverter:
    """图像到网格转换器"""

    def __init__(
        self,
        input_folder,
        output_path,
        threshold=128,
        spacing=(1.0, 1.0, 1.0),
        auto_binarize=True,
        remove_small_objects=True,
        min_size=1000,
    ):
        """
        初始化转换器

        Args:
            input_folder: 输入图像文件夹路径
            output_path: 输出mesh文件路径（支持.obj, .stl, .ply等格式）
            threshold: 二值化阈值，用于分割前景和背景
            spacing: 体素间距 (z, y, x)，用于调整模型比例
            auto_binarize: 是否自动二值化图像
            remove_small_objects: 是否移除小碎片
            min_size: 保留的最小连通域体素数量
        """
        self.input_folder = Path(input_folder)
        self.output_path = Path(output_path)
        self.threshold = threshold
        self.spacing = spacing
        self.auto_binarize = auto_binarize
        self.remove_small_objects = remove_small_objects
        self.min_size = min_size

    def load_images(self):
        """
        加载文件夹中的所有图像（PNG/JPG）

        Returns:
            numpy array: 3D体数据 (depth, height, width)
        """
        # 获取所有PNG和JPG文件并排序
        image_files = sorted(
            list(self.input_folder.glob("*.png"))
            + list(self.input_folder.glob("*.jpg"))
            + list(self.input_folder.glob("*.jpeg"))
        )

        if not image_files:
            raise ValueError(f"在 {self.input_folder} 中未找到图像文件（PNG/JPG）")

        print(f"找到 {len(image_files)} 张图像")

        # 读取第一张图片以获取尺寸
        first_image = np.array(Image.open(image_files[0]))

        # 如果是彩色图像，转换为灰度
        if len(first_image.shape) == 3:
            first_image = np.mean(first_image, axis=2)

        height, width = first_image.shape
        depth = len(image_files)

        # 创建3D数组
        volume = np.zeros((depth, height, width), dtype=np.float32)

        # 读取所有图像
        for i, image_file in tqdm(
            enumerate(image_files), total=len(image_files), desc="加载图像"
        ):
            img = np.array(Image.open(image_file))

            # 转换为灰度图
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)

            volume[i] = img

        print(f"体数据形状: {volume.shape}")
        print(f"体数据范围: [{volume.min():.1f}, {volume.max():.1f}]")
        return volume

    def preprocess_volume(self, volume):
        """
        预处理体数据，防止碎片化

        Args:
            volume: 3D体数据

        Returns:
            numpy array: 处理后的体数据
        """
        processed = volume.copy()

        # 1. 二值化处理
        if self.auto_binarize:
            print(f"执行二值化处理 (阈值: {self.threshold})...")
            processed = (processed > self.threshold).astype(np.uint8) * 255
            print(f"二值化后数据范围: [{processed.min()}, {processed.max()}]")

        # 2. 形态学闭运算 - 填充小孔洞
        print("执行形态学闭运算，填充孔洞...")
        struct = ndimage.generate_binary_structure(3, 1)  # 3D结构元素
        binary = processed > self.threshold
        binary = ndimage.binary_closing(binary, structure=struct, iterations=2)
        processed = binary.astype(np.uint8) * 255

        # 3. 移除小碎片 - 保留最大连通域
        if self.remove_small_objects:
            print("分析连通域，移除碎片...")
            binary = processed > self.threshold

            # 标记连通域
            labeled, num_features = ndimage.label(binary)
            print(f"检测到 {num_features} 个连通域")

            if num_features > 0:
                # 计算每个连通域的大小
                sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

                # 保留大于最小尺寸的连通域
                mask = sizes > self.min_size
                valid_labels = np.where(mask)[0] + 1

                print(
                    f"保留 {len(valid_labels)} 个连通域（阈值: {self.min_size} 体素）"
                )

                # 创建清理后的二值图像
                cleaned = np.isin(labeled, valid_labels)
                processed = cleaned.astype(np.uint8) * 255

        # 4. 形态学开运算 - 平滑边界
        print("执行形态学开运算，平滑边界...")
        binary = processed > self.threshold
        binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
        processed = binary.astype(np.uint8) * 255

        print(f"预处理完成，非零体素数: {np.sum(processed > 0)}")
        return processed

    def reconstruct_mesh(self, volume):
        """
        使用Marching Cubes算法重建网格

        Args:
            volume: 3D体数据

        Returns:
            trimesh.Trimesh: 重建的网格模型
        """
        # 预处理体数据
        volume = self.preprocess_volume(volume)

        print(f"开始使用Marching Cubes算法重建网格 (阈值: {self.threshold})...")

        # 应用Marching Cubes算法
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=self.threshold, spacing=self.spacing
        )

        print(f"生成顶点数: {len(verts)}")
        print(f"生成面片数: {len(faces)}")

        # 创建trimesh对象
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # 后处理：移除孤立的小网格片段
        if self.remove_small_objects:
            print("移除孤立的网格片段...")
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                print(f"检测到 {len(components)} 个网格片段")
                # 保留最大的组件
                largest = max(components, key=lambda m: len(m.vertices))
                print(f"保留最大片段（{len(largest.vertices)} 顶点）")
                mesh = largest

        return mesh

    def export_mesh(self, mesh):
        """
        导出网格到文件

        Args:
            mesh: trimesh.Trimesh对象
        """
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 导出网格
        mesh.export(str(self.output_path))
        print(f"网格已导出到: {self.output_path}")

    def convert(self):
        """
        执行完整的转换流程
        """
        print("=" * 60)
        print("开始图像到网格转换")
        print("=" * 60)

        # 加载图像
        volume = self.load_images()

        # 重建网格
        mesh = self.reconstruct_mesh(volume)

        # 导出网格
        self.export_mesh(mesh)

        print("=" * 60)
        print("转换完成！")
        print("=" * 60)

        return mesh


def main():
    """主函数示例"""
    # 配置参数
    input_folder = "C:\IMCT\眼科\杯盘重建数据\数据标签\ops_mask"  # 输入图像文件夹
    output_path = "output/ops_mask.obj"  # 输出mesh文件路径
    threshold = 128  # 二值化阈值 (0-255)
    spacing = (1.0, 1.0, 1.0)  # 体素间距 (z, y, x)

    # 创建转换器并执行转换
    converter = ImageToMeshConverter(
        input_folder=input_folder,
        output_path=output_path,
        threshold=threshold,
        spacing=spacing,
        auto_binarize=True,  # 自动二值化
        remove_small_objects=True,  # 移除小碎片
        min_size=1000,  # 最小连通域大小
    )

    mesh = converter.convert()

    # 可选：打印网格信息
    print(f"\n网格信息:")
    print(f"  - 顶点数: {len(mesh.vertices)}")
    print(f"  - 面片数: {len(mesh.faces)}")
    print(f"  - 是否闭合: {mesh.is_watertight}")
    print(f"  - 边界框: {mesh.bounds}")


if __name__ == "__main__":
    main()

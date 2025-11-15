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
        smooth_iterations=5,
        use_gaussian_smoothing=True,
        gaussian_sigma=1.0,
        use_laplacian_smoothing=True,
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
            smooth_iterations: Laplacian平滑迭代次数
            use_gaussian_smoothing: 是否在体数据上使用高斯平滑
            gaussian_sigma: 高斯平滑的标准差
            use_laplacian_smoothing: 是否对网格进行Laplacian平滑
        """
        self.input_folder = Path(input_folder)
        self.output_path = Path(output_path)
        self.threshold = threshold
        self.spacing = spacing
        self.auto_binarize = auto_binarize
        self.remove_small_objects = remove_small_objects
        self.min_size = min_size
        self.smooth_iterations = smooth_iterations
        self.use_gaussian_smoothing = use_gaussian_smoothing
        self.gaussian_sigma = gaussian_sigma
        self.use_laplacian_smoothing = use_laplacian_smoothing

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

        with tqdm(
            total=5, desc="预处理体数据", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ) as pbar:
            # 0. 高斯平滑 - 在二值化之前平滑体数据，减少方块感
            if self.use_gaussian_smoothing:
                pbar.set_description(f"高斯平滑 (sigma={self.gaussian_sigma})")
                processed = ndimage.gaussian_filter(
                    processed, sigma=self.gaussian_sigma
                )
                print(
                    f"高斯平滑后数据范围: [{processed.min():.1f}, {processed.max():.1f}]"
                )
            pbar.update(1)

            # 1. 二值化处理
            if self.auto_binarize:
                pbar.set_description(f"二值化处理 (阈值={self.threshold})")
                processed = (processed > self.threshold).astype(np.uint8) * 255
                print(f"二值化后数据范围: [{processed.min()}, {processed.max()}]")
            pbar.update(1)

            # 2. 形态学闭运算 - 填充小孔洞
            pbar.set_description("形态学闭运算-填充孔洞")
            struct = ndimage.generate_binary_structure(3, 1)  # 3D结构元素
            binary = processed > self.threshold
            binary = ndimage.binary_closing(binary, structure=struct, iterations=2)
            processed = binary.astype(np.uint8) * 255
            pbar.update(1)

            # 3. 移除小碎片 - 保留最大连通域
            if self.remove_small_objects:
                pbar.set_description("分析连通域-移除碎片")
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
            pbar.update(1)

            # 4. 形态学开运算 - 平滑边界
            pbar.set_description("形态学开运算-平滑边界")
            binary = processed > self.threshold
            binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
            processed = binary.astype(np.uint8) * 255
            pbar.update(1)

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

        # 平滑网格 - Laplacian平滑
        if self.use_laplacian_smoothing and self.smooth_iterations > 0:
            print(f"执行Laplacian平滑 ({self.smooth_iterations} 次迭代)...")
            mesh = self.laplacian_smooth(mesh, iterations=self.smooth_iterations)

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

    def laplacian_smooth(self, mesh, iterations=5, lambda_factor=0.5):
        """
        对网格执行Laplacian平滑以消除方块感

        Args:
            mesh: trimesh.Trimesh对象
            iterations: 平滑迭代次数
            lambda_factor: 平滑强度因子 (0-1)，越大越平滑但可能丢失细节

        Returns:
            trimesh.Trimesh: 平滑后的网格
        """
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()

        for iteration in tqdm(range(iterations), desc="Laplacian平滑"):
            # 构建顶点邻接关系
            vertex_neighbors = [set() for _ in range(len(vertices))]
            for face in faces:
                for i in range(3):
                    v1, v2, v3 = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
                    vertex_neighbors[v1].add(v2)
                    vertex_neighbors[v1].add(v3)

            # 计算每个顶点的Laplacian
            new_vertices = vertices.copy()
            for i, neighbors in enumerate(vertex_neighbors):
                if len(neighbors) > 0:
                    # 计算邻居顶点的平均位置
                    neighbor_positions = vertices[list(neighbors)]
                    laplacian = neighbor_positions.mean(axis=0) - vertices[i]
                    # 应用平滑
                    new_vertices[i] = vertices[i] + lambda_factor * laplacian

            vertices = new_vertices

        # 创建平滑后的网格
        smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"平滑完成，顶点数: {len(vertices)}")
        return smoothed_mesh

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


files = [
    (r"C:\IMCT\眼科\杯盘重建数据\images", "origin"),
    (r"C:\IMCT\眼科\杯盘重建数据\数据标签\ocs_mask", "ocs_mask"),
    (r"C:\IMCT\眼科\杯盘重建数据\数据标签\ops_mask", "ops_mask"),
]
file = files[0]


def main():
    """主函数示例"""
    # 配置参数
    input_folder = file[0]  # 输入图像文件夹
    output_path = "output/" + file[1] + ".obj"  # 输出mesh文件路径
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
        smooth_iterations=5,  # Laplacian平滑迭代次数
        use_gaussian_smoothing=True,  # 使用高斯平滑
        gaussian_sigma=1.0,  # 高斯平滑强度
        use_laplacian_smoothing=True,  # 使用Laplacian平滑
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

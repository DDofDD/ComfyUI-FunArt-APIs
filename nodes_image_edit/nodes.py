"""
Image Edit Nodes for ComfyUI
包含各种图像编辑相关的节点，使用不同的 AI 模型
"""

from inspect import cleandoc
import io
from http import HTTPStatus

import torch
import numpy as np
from PIL import Image

try:
    import dashscope
    from dashscope import ImageSynthesis
    import requests

    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


class Wan2_5ImageEdit:
    """
    Wan 2.5 图像编辑节点 - 使用 DashScope ImageSynthesis API
    支持通过自然语言描述对图像进行编辑和合成

    API文档: https://bailian.console.aliyun.com/?spm=5176.fcnext.console-base_product-drawer-right.dproducts-and-services-sfm.62952f033vAVNr&tab=api#/api/?type=model&url=2982258

    模型: wan2.5-i2i-preview
    支持功能: 多图输入，通过文字描述进行图像编辑和合成
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "DashScope API密钥"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "图像生成提示词"}),
                "image_url_1": ("STRING", {"multiline": False, "default": "", "tooltip": "第一张图片URL"}),
            },
            "optional": {
                "image_url_2": ("STRING", {"multiline": False, "default": "", "tooltip": "第二张图片URL（可选）"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "负面提示词"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "step": 1, "tooltip": "随机种子，-1表示随机"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "生成图片数量"}),
                "watermark": ("BOOLEAN", {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate_image"
    CATEGORY = "FunArt/ImageEdit"
    OUTPUT_IS_LIST = (True,)

    def download_and_convert_image(self, url):
        """下载图片并转换为ComfyUI的IMAGE tensor"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # 从字节流创建PIL图像
        pil_image = Image.open(io.BytesIO(response.content))

        # 转换为RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # 转换为numpy array并归一化到[0, 1]
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # 转换为tensor [1, H, W, C]
        return torch.from_numpy(img_array)[None,]

    def generate_image(self, api_key, prompt, image_url_1, image_url_2="", negative_prompt="", seed=-1, n=1, watermark=False):
        """
        使用 DashScope Wan 2.5 模型生成图像（图生图）
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope 未安装。请运行: pip install dashscope requests")

        if not api_key:
            raise ValueError("请提供 DashScope API Key")

        if not image_url_1:
            raise ValueError("至少需要提供一张图片URL")

        # 调试信息：打印关键参数
        print("=" * 60)
        print("🔍 Wan2_5ImageEdit 调试信息")
        print("-" * 60)
        print(f"API Key: {api_key}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        print(f"Image URL 1: {image_url_1}")
        print(f"Image URL 2: {image_url_2 if image_url_2 else '(未提供)'}")
        print(f"Seed: {seed}")
        print(f"生成数量: {n}")
        print("=" * 60)

        # 设置 API Key
        dashscope.api_key = api_key
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        print(f"✅ 已设置 dashscope.api_key: {dashscope.api_key[:10]}...")
        print(f"✅ 已设置 dashscope.base_http_api_url: {dashscope.base_http_api_url}")

        # 准备图片URL列表
        image_urls = [image_url_1]
        if image_url_2:
            image_urls.append(image_url_2)

        # 准备API调用参数
        params = {
            "model": "wan2.5-i2i-preview",
            "prompt": prompt,
            "images": image_urls,
            "n": n,
            "watermark": watermark,
        }

        # 添加可选参数
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        if seed >= 0:
            params["seed"] = seed

        # 调用 API
        response = ImageSynthesis.call(**params)

        # 检查响应
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"API调用失败: {response.code} - {response.message}")

        # 下载并转换生成的图片
        output_images = []
        for result in response.output.results:
            tensor = self.download_and_convert_image(result.url)
            output_images.append(tensor)

        return (output_images,)


# 节点类映射 - 用于ComfyUI识别和加载节点
NODE_CLASS_MAPPINGS = {
    "Wan2_5ImageEdit": Wan2_5ImageEdit,
}

# 节点显示名称映射 - 在ComfyUI界面中显示的友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan2_5ImageEdit": "Wan 2.5 图像编辑",
}

"""
Wan 2.5 文生图节点
使用 DashScope ImageSynthesis API 实现文字生成图像功能
"""

from inspect import cleandoc
import io
import time
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


# 支持的图片尺寸
SUPPORTED_SIZES = [
    "1024*1024",
    "768*1024",
    "1024*768",
    "720*1280",
    "1280*720",
]


class Wan2_5T2I:
    """
    Wan 2.5 文生图节点 - 使用 DashScope ImageSynthesis API
    通过文字描述生成图像

    模型: wan2.5-t2i-preview
    支持功能: 文字生成图像，支持提示词扩展
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "DashScope API密钥"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "图像生成提示词"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "负面提示词"}),
                "size": (SUPPORTED_SIZES, {"default": "1024*1024", "tooltip": "输出图片尺寸"}),
                "prompt_extend": ("BOOLEAN", {"default": True, "tooltip": "是否扩展提示词"}),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "随机种子，-1表示随机，范围[0,2147483647]",
                    },
                ),
                "watermark": ("BOOLEAN", {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate_image"
    CATEGORY = "FunArt/Wan"

    def download_and_convert_image(self, url):
        """下载图片并转换为ComfyUI的IMAGE tensor"""
        start_time = time.time()

        # 下载图片
        download_start = time.time()
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        download_time = time.time() - download_start

        # 从字节流创建PIL图像
        pil_image = Image.open(io.BytesIO(response.content))

        # 转换为RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # 转换为numpy array并归一化到[0, 1]
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # 转换为tensor [1, H, W, C]
        tensor = torch.from_numpy(img_array)[None,]

        elapsed_time = time.time() - start_time
        print(
            f"⏱️  download_and_convert_image 耗时: {elapsed_time:.3f}秒 (下载: {download_time:.3f}秒, 转换: {elapsed_time-download_time:.3f}秒, 尺寸: {tensor.shape})"
        )

        return tensor

    def generate_image(
        self,
        api_key,
        prompt,
        negative_prompt="",
        size="1024*1024",
        prompt_extend=True,
        seed=-1,
        watermark=False,
    ):
        """
        使用 DashScope Wan 2.5 模型生成图像（文生图）
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope 未安装。请运行: pip install dashscope requests")

        if not api_key:
            raise ValueError("请提供 DashScope API Key")

        if not prompt:
            raise ValueError("请提供图像生成提示词")

        # 设置 API Key
        dashscope.api_key = api_key
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        # 准备API调用参数
        params = {
            "model": "wan2.5-t2i-preview",
            "prompt": prompt,
            "n": 1,  # 固定生成1张图片
            "size": size,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        # 添加可选参数
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        if seed >= 0:
            # 确保 seed 在 DashScope API 允许的范围内 [0, 2147483647]
            valid_seed = seed % 2147483648  # 2^31
            if valid_seed != seed:
                print(f"⚠️  Seed {seed} 超出 API 范围，已调整为 {valid_seed}")
            params["seed"] = valid_seed

        # 调用 API
        print("🚀 正在调用 DashScope API (模型: wan2.5-t2i-preview)")
        print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
        print(f"📐 Size: {size}")
        print(f"🔄 Prompt Extend: {prompt_extend}")

        response = ImageSynthesis.call(**params)

        print(f"📥 API 响应状态: {response.status_code}")
        print(f"📋 Request ID: {response.request_id if hasattr(response, 'request_id') else 'N/A'}")

        # 检查响应状态
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"API调用失败: {response.code} - {response.message}")

        # 检查结果是否为空
        if not response.output or not response.output.results:
            print("=" * 60)
            print("❌ API 调用异常：返回成功但没有生成图片")
            print("-" * 60)
            print(f"Status Code: {response.status_code}")
            print(f"Request ID: {response.request_id if hasattr(response, 'request_id') else 'N/A'}")
            print(f"Code: {response.code if hasattr(response, 'code') else 'N/A'}")
            print(f"Message: {response.message if hasattr(response, 'message') else 'N/A'}")
            print(f"Output: {response.output if hasattr(response, 'output') else 'N/A'}")
            print("=" * 60)

            error_msg = "API 返回成功但没有生成图片，可能是配额限制、频率限制或其他 API 问题"
            raise RuntimeError(error_msg)

        print(f"✅ 成功生成 {len(response.output.results)} 张图片")

        # 打印扩展后的提示词（如果有）
        result = response.output.results[0]
        if hasattr(result, "actual_prompt") and result.actual_prompt:
            print(
                f"📝 扩展后提示词: {result.actual_prompt[:100]}..."
                if len(result.actual_prompt) > 100
                else f"📝 扩展后提示词: {result.actual_prompt}"
            )

        # 下载并转换生成的图片
        output_tensor = self.download_and_convert_image(result.url)

        # 返回单张图片，shape: [1, H, W, C]
        return (output_tensor,)

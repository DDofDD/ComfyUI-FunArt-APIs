"""
Image Edit Nodes for ComfyUI
包含各种图像编辑相关的节点，使用不同的 AI 模型
"""

from inspect import cleandoc
import io
import base64
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
                "image_1": ("IMAGE", {"tooltip": "第一张输入图像"}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "第二张输入图像（可选）"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "负面提示词"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "step": 1, "tooltip": "随机种子，-1表示随机"}),
                "watermark": ("BOOLEAN", {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate_image"
    CATEGORY = "FunArt/ImageEdit"

    def tensor_to_base64(self, tensor):
        """将ComfyUI的IMAGE tensor转换为base64字符串"""
        start_time = time.time()

        # tensor shape: [B, H, W, C] 或 [H, W, C]
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # 取第一张图片

        # 转换为 numpy array (H, W, C)，值范围 [0, 1]
        img_array = tensor.cpu().numpy()

        # 转换为 0-255 范围的 uint8
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

        # 转换为 PIL Image
        pil_image = Image.fromarray(img_array, mode="RGB")

        # 转换为 bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Base64 编码
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")

        elapsed_time = time.time() - start_time
        print(f"⏱️  tensor_to_base64 耗时: {elapsed_time:.3f}秒 (图片大小: {len(encoded_string)//1024}KB)")

        # 返回 data URI 格式
        return f"data:image/png;base64,{encoded_string}"

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

    def generate_image(self, api_key, prompt, image_1, image_2=None, negative_prompt="", seed=-1, watermark=False):
        """
        使用 DashScope Wan 2.5 模型生成图像（图生图）
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope 未安装。请运行: pip install dashscope requests")

        if not api_key:
            raise ValueError("请提供 DashScope API Key")

        # 设置 API Key
        dashscope.api_key = api_key
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        # 将 IMAGE tensor 转换为 base64
        image_base64_list = [self.tensor_to_base64(image_1)]
        if image_2 is not None:
            image_base64_list.append(self.tensor_to_base64(image_2))

        # 准备API调用参数
        params = {
            "model": "wan2.5-i2i-preview",
            "prompt": prompt,
            "images": image_base64_list,
            "n": 1,  # 固定生成1张图片
            "watermark": watermark,
        }

        # 添加可选参数
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        if seed >= 0:
            params["seed"] = seed

        # 调用 API
        print("🚀 正在调用 DashScope API (模型: wan2.5-i2i-preview)")
        print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
        print(f"🖼️  图片数量: {len(image_base64_list)}")

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

        # 下载并转换生成的图片（只有一张）
        result = response.output.results[0]
        output_tensor = self.download_and_convert_image(result.url)

        # 返回单张图片，shape: [1, H, W, C]
        return (output_tensor,)


# 节点类映射 - 用于ComfyUI识别和加载节点
NODE_CLASS_MAPPINGS = {
    "Wan2_5ImageEdit": Wan2_5ImageEdit,
}

# 节点显示名称映射 - 在ComfyUI界面中显示的友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan2_5ImageEdit": "Wan 2.5 图像编辑",
}

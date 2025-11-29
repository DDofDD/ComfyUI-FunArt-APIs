"""
Wan Model API Nodes for ComfyUI
This module contains nodes for interacting with Wan model APIs
"""

from inspect import cleandoc


class WanModelBase:
    """
    Base class for Wan model nodes - Wan模型节点基类
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        """
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Wan API密钥"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "输入提示词"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "FunArt/Wan"

    def execute(self, api_key, prompt):
        """
        执行Wan模型调用
        """
        # TODO: 实现实际的API调用逻辑
        result = f"Wan模型调用 - Prompt: {prompt}"
        return (result,)


class WanTextGeneration:
    """
    Wan文本生成节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Wan API密钥"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "输入提示词"}),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 4096, "step": 1, "display": "number", "tooltip": "最大生成token数"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "display": "number", "tooltip": "温度参数，控制随机性"},
                ),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "系统提示词"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate"
    CATEGORY = "FunArt/Wan"

    def generate(self, api_key, prompt, max_tokens, temperature, system_prompt=""):
        """
        生成文本
        """
        # TODO: 实现实际的Wan API调用
        result = f"生成的文本 (max_tokens={max_tokens}, temp={temperature})\nPrompt: {prompt}"
        if system_prompt:
            result = f"System: {system_prompt}\n" + result
        return (result,)


class WanImageGeneration:
    """
    Wan图像生成节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Wan API密钥"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "图像生成提示词"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "display": "number", "tooltip": "图像宽度"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "display": "number", "tooltip": "图像高度"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "display": "number", "tooltip": "生成步数"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "负面提示词"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "step": 1, "tooltip": "随机种子，-1表示随机"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate_image"
    CATEGORY = "FunArt/Wan"

    def generate_image(self, api_key, prompt, width, height, steps, negative_prompt="", seed=-1):
        """
        生成图像
        """
        # TODO: 实现实际的Wan图像生成API调用
        # 这里需要返回一个符合ComfyUI IMAGE格式的tensor
        import torch
        import numpy as np

        # 临时返回一个占位图像
        dummy_image = np.zeros((height, width, 3), dtype=np.float32)
        image_tensor = torch.from_numpy(dummy_image)[None,]

        return (image_tensor,)


class WanModelConfig:
    """
    Wan模型配置节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Wan API密钥"}),
                "api_endpoint": ("STRING", {"multiline": False, "default": "https://api.wan.example.com", "tooltip": "API端点URL"}),
                "model": (["wan-v1", "wan-v2", "wan-pro"], {"default": "wan-v1", "tooltip": "选择模型版本"}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 300, "step": 1, "display": "number", "tooltip": "请求超时时间（秒）"}),
            },
        }

    RETURN_TYPES = ("WAN_CONFIG",)
    RETURN_NAMES = ("config",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "create_config"
    CATEGORY = "FunArt/Wan"

    def create_config(self, api_key, api_endpoint, model, timeout):
        """
        创建Wan模型配置
        """
        config = {"api_key": api_key, "api_endpoint": api_endpoint, "model": model, "timeout": timeout}
        return (config,)


# 节点类映射 - 用于ComfyUI识别和加载节点
NODE_CLASS_MAPPINGS = {
    "WanModelBase": WanModelBase,
    "WanTextGeneration": WanTextGeneration,
    "WanImageGeneration": WanImageGeneration,
    "WanModelConfig": WanModelConfig,
}

# 节点显示名称映射 - 在ComfyUI界面中显示的友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanModelBase": "Wan模型基础",
    "WanTextGeneration": "Wan文本生成",
    "WanImageGeneration": "Wan图像生成",
    "WanModelConfig": "Wan模型配置",
}

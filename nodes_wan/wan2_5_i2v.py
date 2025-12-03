"""
Wan 2.5 图生视频节点
使用 DashScope VideoSynthesis API 实现首帧图生视频功能
"""

from inspect import cleandoc
import io
import os
import base64
import time
import uuid
from http import HTTPStatus

import numpy as np
from PIL import Image

try:
    import folder_paths

    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    import dashscope
    from dashscope import VideoSynthesis
    import requests

    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

# 尝试导入音频处理库
try:
    import scipy.io.wavfile as wavfile

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# 支持的分辨率
SUPPORTED_RESOLUTIONS = ["1080P", "720P", "480P"]


class Wan2_5I2V:
    """
    Wan 2.5 图生视频节点 - 使用 DashScope VideoSynthesis API
    基于首帧图片生成视频，支持音频驱动

    模型: wan2.5-i2v-preview
    支持功能: 首帧图生视频，音频驱动，提示词扩展
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"multiline": False, "default": "", "tooltip": "DashScope API密钥"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "视频生成提示词"},
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "首帧图片。"
                            "格式: JPEG/JPG/PNG(不支持透明通道)/BMP/WEBP; "
                            "分辨率: 宽高范围[360,2000]像素; "
                            "大小: 不超过10MB"
                        )
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "音频输入（可选，用于音频驱动）。"
                            "格式: wav/mp3; 时长: 3~30秒; 大小: 不超过15MB。"
                            "若音频超过视频时长则自动截取，不足则超出部分无声"
                        )
                    },
                ),
                "resolution": (
                    SUPPORTED_RESOLUTIONS,
                    {"default": "1080P", "tooltip": "输出视频分辨率: 480P/720P/1080P"},
                ),
                "duration": (
                    [5, 10],
                    {"default": 5, "tooltip": "视频时长（秒），可选5或10"},
                ),
                "prompt_extend": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "是否扩展提示词"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "负面提示词"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "随机种子，-1表示随机",
                    },
                ),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否添加水印"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate_video"
    CATEGORY = "FunArt/Wan"

    def get_output_directory(self):
        """获取视频输出目录"""
        if FOLDER_PATHS_AVAILABLE:
            return folder_paths.get_output_directory()
        else:
            # 回退到当前目录下的 output 文件夹
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir

    def download_video(self, url, filename_prefix="wan_i2v"):
        """下载视频并保存到输出目录

        Args:
            url: 视频URL
            filename_prefix: 文件名前缀

        Returns:
            保存的视频文件完整路径
        """
        start_time = time.time()

        # 下载视频
        print("📥 正在下载视频...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        # 生成唯一文件名
        unique_id = uuid.uuid4().hex[:8]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.mp4"

        # 保存到输出目录
        output_dir = self.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(response.content)

        elapsed_time = time.time() - start_time
        file_size_mb = len(response.content) / (1024 * 1024)
        print(f"⏱️  download_video 耗时: {elapsed_time:.3f}秒 (文件大小: {file_size_mb:.2f}MB)")
        print(f"💾 视频已保存: {output_path}")

        return output_path

    def tensor_to_base64_image(self, tensor):
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
        print(f"⏱️  tensor_to_base64_image 耗时: {elapsed_time:.3f}秒 (图片大小: {len(encoded_string)//1024}KB)")

        # 返回 data URI 格式
        return f"data:image/png;base64,{encoded_string}"

    def audio_to_base64(self, audio):
        """将ComfyUI的AUDIO转换为base64字符串

        ComfyUI AUDIO 格式: {"waveform": torch.Tensor, "sample_rate": int}
        waveform shape: [batch, channels, samples] 或 [channels, samples]
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy 未安装，无法处理音频。请运行: pip install scipy")

        start_time = time.time()

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # 处理 waveform tensor
        if len(waveform.shape) == 3:
            waveform = waveform[0]  # 取第一个 batch

        # waveform shape: [channels, samples]
        # 转换为 numpy array
        audio_array = waveform.cpu().numpy()

        # 如果是立体声，转换为 [samples, channels]
        if audio_array.shape[0] <= 2:  # channels first
            audio_array = audio_array.T  # 转置为 [samples, channels]

        # 归一化到 int16 范围
        audio_array = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)

        # 保存为 WAV 格式
        buffered = io.BytesIO()
        wavfile.write(buffered, sample_rate, audio_array)
        audio_bytes = buffered.getvalue()

        # Base64 编码
        encoded_string = base64.b64encode(audio_bytes).decode("utf-8")

        elapsed_time = time.time() - start_time
        print(f"⏱️  audio_to_base64 耗时: {elapsed_time:.3f}秒 (音频大小: {len(encoded_string)//1024}KB)")

        # 返回 data URI 格式
        return f"data:audio/wav;base64,{encoded_string}"

    def generate_video(
        self,
        api_key,
        prompt,
        image,
        audio=None,
        resolution="1080P",
        duration=5,
        prompt_extend=True,
        negative_prompt="",
        seed=-1,
        watermark=False,
    ):
        """
        使用 DashScope Wan 2.5 模型生成视频（图生视频）
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope 未安装。请运行: pip install dashscope requests")

        if not api_key:
            raise ValueError("请提供 DashScope API Key")

        if not prompt:
            raise ValueError("请提供视频生成提示词")

        # 设置 API Key
        dashscope.api_key = api_key
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        # 将 IMAGE tensor 转换为 base64
        image_base64 = self.tensor_to_base64_image(image)

        # 准备 API 调用参数
        params = {
            "model": "wan2.5-i2v-preview",
            "prompt": prompt,
            "img_url": image_base64,
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        # 添加音频（如果有）
        if audio is not None:
            audio_base64 = self.audio_to_base64(audio)
            params["audio_url"] = audio_base64

        # 添加可选参数
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        if seed >= 0:
            valid_seed = seed % 2147483648
            if valid_seed != seed:
                print(f"⚠️  Seed {seed} 超出 API 范围，已调整为 {valid_seed}")
            params["seed"] = valid_seed

        # ========== 步骤1: 异步调用 ==========
        print("🚀 正在调用 DashScope VideoSynthesis API (模型: wan2.5-i2v-preview)")
        print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
        print(f"📐 Resolution: {resolution}, Duration: {duration}s")
        print(f"🎵 Audio: {'有' if audio is not None else '无'}")

        rsp = VideoSynthesis.async_call(**params)

        print(f"📋 异步调用响应: Task ID = {rsp.output.task_id if rsp.output else 'N/A'}")

        if rsp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"API异步调用失败: {rsp.code} - {rsp.message}")

        task_id = rsp.output.task_id
        print(f"✅ 任务已提交! Task ID: {task_id}")

        # ========== 步骤2: 等待任务完成 ==========
        print("⏳ 等待视频生成完成 (可能需要几分钟)...")

        result = VideoSynthesis.wait(task=rsp, api_key=api_key)

        print(f"📥 最终响应状态: {result.status_code}")

        if result.status_code != HTTPStatus.OK:
            raise RuntimeError(f"视频生成失败: {result.code} - {result.message}")

        if not result.output or not result.output.video_url:
            print("=" * 60)
            print("❌ API 调用异常：返回成功但没有生成视频")
            print("-" * 60)
            print(f"Status Code: {result.status_code}")
            print(f"Task ID: {task_id}")
            print(f"Output: {result.output if hasattr(result, 'output') else 'N/A'}")
            print("=" * 60)
            raise RuntimeError("API 返回成功但没有生成视频")

        video_url = result.output.video_url
        print("✅ 视频生成成功!")
        print(f"🎬 视频URL: {video_url}")

        # 打印扩展后的提示词（如果有）
        if hasattr(result.output, "actual_prompt") and result.output.actual_prompt:
            actual_prompt = result.output.actual_prompt
            print(f"📝 扩展后提示词: {actual_prompt[:100]}..." if len(actual_prompt) > 100 else f"📝 扩展后提示词: {actual_prompt}")

        # 下载并保存视频
        video_path = self.download_video(video_url, filename_prefix="wan_i2v")

        return (video_path,)

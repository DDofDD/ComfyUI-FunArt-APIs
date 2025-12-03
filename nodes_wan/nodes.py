"""
Wan 系列节点 - 节点映射
"""

from .wan2_5_image_edit import Wan2_5ImageEdit
from .wan2_5_i2v import Wan2_5I2V
from .wan2_5_t2i import Wan2_5T2I
from .wan2_5_t2v import Wan2_5T2V

# 节点类映射 - 用于ComfyUI识别和加载节点
NODE_CLASS_MAPPINGS = {
    "Wan2_5ImageEdit": Wan2_5ImageEdit,
    "Wan2_5I2V": Wan2_5I2V,
    "Wan2_5T2I": Wan2_5T2I,
    "Wan2_5T2V": Wan2_5T2V,
}

# 节点显示名称映射 - 在ComfyUI界面中显示的友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan2_5ImageEdit": "Wan 2.5 图像编辑",
    "Wan2_5I2V": "Wan 2.5 图生视频",
    "Wan2_5T2I": "Wan 2.5 文生图",
    "Wan2_5T2V": "Wan 2.5 文生视频",
}

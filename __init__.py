"""Top-level package for funart_apis."""

import os
import sys

# 将 src 目录添加到 Python 路径中，以便 ComfyUI 可以正确导入
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 从各个节点组导入并合并
from wan_nodes import NODE_CLASS_MAPPINGS as WAN_NODE_CLASS_MAPPINGS
from wan_nodes import NODE_DISPLAY_NAME_MAPPINGS as WAN_NODE_DISPLAY_NAME_MAPPINGS
from fc_nodes import NODE_CLASS_MAPPINGS as FC_NODE_CLASS_MAPPINGS
from fc_nodes import NODE_DISPLAY_NAME_MAPPINGS as FC_NODE_DISPLAY_NAME_MAPPINGS

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(WAN_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(WAN_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(FC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FC_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """zijian"""
__email__ = "qiucheng.wzj@alibaba-inc.com"
__version__ = "0.0.1"

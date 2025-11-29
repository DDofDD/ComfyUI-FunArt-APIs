"""
FC Nodes for ComfyUI
"""

from inspect import cleandoc


class FCExampleNode:
    """
    FC示例节点 - 这是一个示例模板，后续可以根据实际需求修改
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "输入文本"}),
                "number": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number", "tooltip": "数值参数"}),
            },
            "optional": {
                "optional_param": ("STRING", {"multiline": False, "default": "", "tooltip": "可选参数"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "process"
    CATEGORY = "FunArt/FC"

    def process(self, text, number, optional_param=""):
        """
        处理方法 - 在这里实现你的逻辑
        """
        # TODO: 实现实际的处理逻辑
        result = f"FC处理结果:\n文本: {text}\n数值: {number}"
        if optional_param:
            result += f"\n可选参数: {optional_param}"

        return (result,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "FCExampleNode": FCExampleNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "FCExampleNode": "FC示例节点",
}

# Wan模型节点组

这是一组用于ComfyUI的Wan模型API调用节点。

## 节点列表

### 1. WanModelConfig (Wan模型配置)
配置Wan模型的基本参数，包括API密钥、端点URL、模型版本等。

**输入参数：**
- `api_key`: Wan API密钥
- `api_endpoint`: API端点URL
- `model`: 模型版本选择 (wan-v1, wan-v2, wan-pro)
- `timeout`: 请求超时时间（秒）

**输出：**
- `config`: Wan配置对象，可传递给其他Wan节点使用

---

### 2. WanTextGeneration (Wan文本生成)
使用Wan模型生成文本内容。

**输入参数：**
- `api_key`: Wan API密钥
- `prompt`: 输入提示词（必填）
- `max_tokens`: 最大生成token数 (1-4096, 默认1024)
- `temperature`: 温度参数，控制随机性 (0.0-2.0, 默认0.7)
- `system_prompt`: 系统提示词（可选）

**输出：**
- `generated_text`: 生成的文本内容

---

### 3. WanImageGeneration (Wan图像生成)
使用Wan模型生成图像。

**输入参数：**
- `api_key`: Wan API密钥
- `prompt`: 图像生成提示词（必填）
- `width`: 图像宽度 (64-2048, 步长64, 默认512)
- `height`: 图像高度 (64-2048, 步长64, 默认512)
- `steps`: 生成步数 (1-100, 默认20)
- `negative_prompt`: 负面提示词（可选）
- `seed`: 随机种子，-1表示随机（可选）

**输出：**
- `image`: 生成的图像（ComfyUI IMAGE格式）

---

### 4. WanModelBase (Wan模型基础)
基础的Wan模型调用节点，可用于简单的API测试。

**输入参数：**
- `api_key`: Wan API密钥
- `prompt`: 输入提示词

**输出：**
- `result`: 返回结果字符串

---

## 使用方法

### 基本工作流示例

1. **配置API**
   - 添加 `WanModelConfig` 节点
   - 填入你的API密钥和端点URL
   - 选择合适的模型版本

2. **文本生成**
   - 添加 `WanTextGeneration` 节点
   - 输入提示词
   - 调整参数（max_tokens, temperature等）
   - 执行生成

3. **图像生成**
   - 添加 `WanImageGeneration` 节点
   - 输入图像描述提示词
   - 设置图像尺寸和生成参数
   - 执行生成

## 开发说明

### 文件结构
```
wan_nodes/
├── __init__.py          # 包初始化，导出节点映射
├── nodes.py             # 节点类定义
└── README.md            # 本文档
```

### 添加新节点

1. 在 `nodes.py` 中创建新的节点类
2. 实现以下必需属性和方法：
   - `INPUT_TYPES()`: 类方法，定义输入参数
   - `RETURN_TYPES`: 返回值类型元组
   - `FUNCTION`: 入口方法名称
   - `CATEGORY`: 节点分类
   - 执行方法（方法名与FUNCTION属性值一致）

3. 将新节点添加到 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`

### 示例：添加新节点

```python
class WanNewFeature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param1": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "FunArt/Wan"
    
    def process(self, param1):
        # 实现你的逻辑
        result = f"Processed: {param1}"
        return (result,)

# 添加到映射
NODE_CLASS_MAPPINGS["WanNewFeature"] = WanNewFeature
NODE_DISPLAY_NAME_MAPPINGS["WanNewFeature"] = "Wan新功能"
```

## TODO

- [ ] 实现实际的Wan API调用逻辑
- [ ] 添加错误处理和重试机制
- [ ] 支持流式输出
- [ ] 添加更多模型参数选项
- [ ] 实现API调用缓存
- [ ] 添加使用统计和监控

## 注意事项

1. 当前版本的API调用逻辑为占位符，需要根据实际的Wan API文档进行实现
2. 图像生成节点目前返回占位图像，需要实现真实的API调用和图像转换
3. 确保API密钥安全，不要硬编码在代码中


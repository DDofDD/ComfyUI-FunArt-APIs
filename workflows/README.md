# ComfyUI 工作流示例

本目录包含 FunArt Wan 系列节点的 ComfyUI 工作流示例。

## 目录结构

```
workflows/
├── Wan2_5_ImageEdit/    # 图像编辑工作流
│   ├── flow.json        # ComfyUI 工作流文件
│   └── input/           # 输入文件（图片等）
├── Wan2_5_I2V/          # 图生视频工作流
│   ├── flow.json
│   └── input/
├── Wan2_5_T2I/          # 文生图工作流
│   ├── flow.json
│   └── input/
└── Wan2_5_T2V/          # 文生视频工作流
    ├── flow.json
    └── input/
```

## 使用方法

1. **导入工作流**
   - 在 ComfyUI 中，点击 "Load" 按钮
   - 选择对应节点目录下的 `flow.json` 文件

2. **准备输入文件**
   - 将所需的输入文件（图片、音频等）放入对应的 `input/` 目录
   - 在 ComfyUI 中配置节点参数时引用这些文件

3. **配置 API Key**
   - 在工作流中配置 DashScope API Key
   - 或设置环境变量 `DASHSCOPE_API_KEY`

## 节点说明

### Wan2_5_ImageEdit - 图像编辑
使用 DashScope Wan 2.5 模型进行图像编辑。

**输入:**
- image: 待编辑的图片
- prompt: 编辑提示词

**输出:**
- IMAGE: 编辑后的图片

### Wan2_5_I2V - 图生视频
基于首帧图片生成视频。

**输入:**
- image: 首帧图片
- prompt: 视频描述
- audio (可选): 音频文件

**输出:**
- VIDEO: 生成的视频

### Wan2_5_T2I - 文生图
基于文字描述生成图片。

**输入:**
- prompt: 图片描述

**输出:**
- IMAGE: 生成的图片

### Wan2_5_T2V - 文生视频
基于文字描述生成视频。

**输入:**
- prompt: 视频描述
- audio (可选): 音频文件

**输出:**
- VIDEO: 生成的视频

## 注意事项

- 所有节点都需要有效的 DashScope API Key
- 图片支持格式：JPEG、JPG、PNG、BMP、WEBP
- 音频支持格式：WAV、MP3
- 视频输出保存在 `output/temp/` 目录

[中文](README-CN.md)|[English](README.md)

# 为所有影,音,图,文创作生成提示的 ComfyUI 节点

**增加所有 Ollama 模型支持, 任何 Ollama 模型都能跑, 包括自定义模型, 可本地生成任何提示, 反推图像, 视频等等**, 尽情畅玩吧. 

**增加音频/音乐反推, 打标, 音频/音乐训练 LoRA 利器. 速度快, 效果好, 甚至超过使用 API**. 

**更新 JoyCaption 到最新, 支持 NF4, 小显存福音.**, 

目前支持的 API: DeepSeek-R1/V3, 支持阿里云 Qwen 几乎全家桶, 支持谷歌 gemini 全家桶. 更多好用的 API 会陆续更新. 

精选的本地模型, 每一个都会独立开来, 方便大家使用, 想用什么就下载什么即可.

## 📣 更新

[2025-06-12]⚒️: 发布版本 v2.0.0. 

[2025-05-10]⚒️: 支持 gemini 最新模型, `gemini-2.5-pro-preview-05-06` 需要付费. 支持最新 Qwen3.

[2025-04-12]⚒️: 支持 JoyCaption.

[2025-04-01]⚒️: 发布版本 v1.0.0. 

- 支持 DeepSeek-R1/V3 模型 API, 需要申请 API 密钥, [DeepSeek 官网](https://platform.deepseek.com/api_keys) 申请. 然后新建环境变量 `DEEPSEEK_API_KEY = <your key>` 到系统环境变量中. 添加方法见 [将API Key配置到环境变量](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.38b26132lodett#e4cd73d544i3r). 也可以不用系统变量, 直接输入节点使用, 但是**注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**.

- 支持 Qwen API, 申请 API 密钥, [阿里云百炼](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3f7d7980x2Vg6r&apiKey=1#/api-key), 然后新建环境变量 `DASHSCOPE_API_KEY = <your key>` 到系统环境变量中. 添加方法, 使用方法同上, **注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**. 后缀为 “_R” 的推理模型有一个思维过程.

- 支持 Gemini API, 申请 API 密钥, [Google AI Studio](https://aistudio.google.com/app/apikey), 然后新建环境变量 `GOOGLE_API_KEY = <your key>` 到系统环境变量中. 添加方法, 使用方法同上, **注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**.

## 用法

音频/音乐反推:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_22-18-44.png)

Ollama 模型 Flux 提示生成:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-47-27.png)

Ollama 模型图像/视频反推:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-31-01.png)
![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-46-28.png)

JoyCaption 图像反推描述:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-04-12_04-09-15.png)

- images_dir: 批量打标图片路径.

- save_img_prompt_to_folder: 图片和提示保存路径. 如果提供, 批量打标图片和提示将保存到这个文件夹中, 否则默认保存到 images_dir, 与图片同名; 如果提供, 即使单张图片也可保存到这个文件夹中.

DeepSeek:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekr1.png)

Qwen:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen1.png)

Gemini:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini1.png)

## 模型下载

**不是所有模型都需要下载, 你需要什么就下载什么**.**

以下模型整个文件夹手动下载到 `LLM` 文件夹下:

- JoyCaption:
  - [llama-joycaption-alpha-two-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-alpha-two-hf-llava-nf4/tree/main).
  - [llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava/tree/main).
  - [llama-joycaption-beta-one-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-beta-one-hf-llava-nf4).
  - [llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/tree/main).

- Ke-Omni-R-3B:
  - https://hf-mirror.com/KE-Team/Ke-Omni-R-3B/tree/main. 
  - 注意: **模型缺少索引文件, 我生成了索引文件 `model.safetensors.index.json`, 就在本仓库中, 请移动到该模型的同一个根目录中**.

Ollama 模型:

首先安装 [ollama](https://ollama.com/download). 然后任何 Ollama 模型都可以使用, 包括自定义模型.

强烈推进的本地消费级模型, 性能强, 速度快, 应用广. 执行如下命令下载模型(**下载速度非常快, 网速好, 10g 分分钟下完, 最后速度慢可以关掉重新下, 会接着高速下载**):

- `ollama pull artifish/llama3.2-uncensored` 未经审查 llama3.2.

https://ollama.com/artifish/llama3.2-uncensored

- `ollama pull poluramus/llama-3.2ft_flux-prompting_v0.5` 超强 Flux 提示生成模型.

https://ollama.com/poluramus/llama-3.2ft_flux-prompting_v0.5

- `ollama pull abedalswaity7/flux-prompt` 另一个超强 Flux 提示生成模型.

https://ollama.com/abedalswaity7/flux-prompt

- `ollama pull qwen2.5vl:7b` 阿里超强多模态, 图像, 视频反推利器, 多个参数版本可选, 7b 就非常棒, 3b 超快.

https://ollama.com/library/qwen2.5vl

- `ollama pull fanyx/openbmb.MiniCPM4-8B-GGUF-Q8_0:latest` 刚上线热乎的 面壁小钢炮, huggingface 趋势榜前六, 号称同参数最好, 最快模型.

https://ollama.com/fanyx/openbmb.MiniCPM4-8B-GGUF-Q8_0

**感谢模型作者的无私奉献**.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Prompt-All-In-One.git
cd ComfyUI_Prompt-All-In-One
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 鸣谢

- [joycaption](https://github.com/fpgaminer/joycaption)
- [Ke-Omni-R](https://github.com/shuaijiang/Ke-Omni-R/)
- [ollama](https://ollama.com/download)
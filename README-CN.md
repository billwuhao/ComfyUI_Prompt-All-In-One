[中文](README-CN.md)|[English](README.md)

# 为所有影,音,图,文创作生成提示的 ComfyUI 节点

目前支持的 API: DeepSeek-R1/V3, 支持阿里云 Qwen 几乎全家桶, 支持谷歌 gemini 全家桶. 更多好用的 API 会陆续更新. 有好用的 API 也欢迎提 PR.

还会增加精选的本地模型, 每一个都会独立开来, 方便大家使用, 想用什么就下载什么即可.

## 📣 更新

[2025-05-10]⚒️: 支持 gemini 最新模型, `gemini-2.5-pro-preview-05-06` 需要付费. 支持最新 Qwen3.

[2025-04-12]⚒️: 支持 JoyCaption.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-04-12_04-09-15.png)

- images_dir: 批量打标图片路径.

- save_img_prompt_to_folder: 图片和提示保存路径. 如果提供, 批量打标图片和提示将保存到这个文件夹中, 否则默认保存到 images_dir, 与图片同名; 如果提供, 即使单张图片也可保存到这个文件夹中.

模型手动下载放到 `LLM` 文件夹下即可:
- [llama-joycaption-alpha-two-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-alpha-two-hf-llava-nf4/tree/main), 估计需要 8g 显存.
- [llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava/tree/main), 估计需要 16g 显存.

[2025-04-01]⚒️: 发布版本 v1.0.0. 

- 支持 DeepSeek-R1/V3 模型 API, 需要申请 API 密钥, [DeepSeek 官网](https://platform.deepseek.com/api_keys) 申请. 然后新建环境变量 `DEEPSEEK_API_KEY = <your key>` 到系统环境变量中. 添加方法见 [将API Key配置到环境变量](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.38b26132lodett#e4cd73d544i3r). 也可以不用系统变量, 直接输入节点使用, 但是**注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekr1.png)

- 支持 Qwen API, 申请 API 密钥, [阿里云百炼](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3f7d7980x2Vg6r&apiKey=1#/api-key), 然后新建环境变量 `DASHSCOPE_API_KEY = <your key>` 到系统环境变量中. 添加方法, 使用方法同上, **注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**. 后缀为 “_R” 的推理模型有一个思维过程.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen1.png)

- 支持 Gemini API, 申请 API 密钥, [Google AI Studio](https://aistudio.google.com/app/apikey), 然后新建环境变量 `GOOGLE_API_KEY = <your key>` 到系统环境变量中. 添加方法, 使用方法同上, **注意保密你的 key, 不要随工作流泄露**. **Windows 添加完环境变量可能需要重启电脑才会生效**.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini1.png)

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

[joycaption](https://github.com/fpgaminer/joycaption)
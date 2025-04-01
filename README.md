[中文](README-CN.md)|[English](README.md)

# ComfyUI Nodes for Generating Prompts for All Media Creations (Image, Video, Audio, Text)

Currently supports DeepSeek-R1/V3, Alibaba Cloud Qwen (almost all models), and Google Gemini (including the `gemini-2.0-flash-exp-image-generation` model for image editing with your mouth). More useful APIs will be added in the future.  Contributions with useful APIs are welcome.

Will also add a curated selection of local models, each kept separate for easy use.  Just download the ones you want.

## 📣 Updates

[2025-04-01] ⚒️: Released version v1.0.0.

- Supports DeepSeek-R1/V3 model API.  Requires an API key. Obtain a key from the [DeepSeek website](https://platform.deepseek.com/api_keys). Then set the environment variable `DEEPSEEK_API_KEY = <your key>`. See [Configuring API Keys Using Environment Variables](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.38b26132lodett#e4cd73d544i3r) for setup instructions.  You can also input the key directly in the node, but **be sure to keep your key confidential and do not expose it in your workflows.**  **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.**

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekr1.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekv3.png)

- Supports Qwen API.  Requires an API key. Obtain from [Alibaba Cloud Bailian](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3f7d7980x2Vg6r&apiKey=1#/api-key).  Then set the environment variable `DASHSCOPE_API_KEY = <your key>`.  Setup and usage instructions are the same as above. **Be sure to keep your key confidential and do not expose it in your workflows.** **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.**

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen1.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen2.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen3.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen4.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen5.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen6.png)

- Supports Gemini API.  Requires an API key. Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey).  Then set the environment variable `GOOGLE_API_KEY = <your key>`. Setup and usage instructions are the same as above. **Be sure to keep your key confidential and do not expose it in your workflows.** **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.**

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini1.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini2.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini3.png)

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini4.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Prompt-All-In-One.git
cd ComfyUI_Prompt-All-In-One
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```
<!--
## Acknowledgments

[csm](https://github.com/SesameAILabs/csm) -->

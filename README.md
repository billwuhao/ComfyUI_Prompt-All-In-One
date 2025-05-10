[中文](README-CN.md)|[English](README.md)

# ComfyUI Nodes for Generating Prompts for All Media Creations (Image, Video, Audio, Text)

Currently supports DeepSeek-R1/V3, Alibaba Cloud Qwen (almost all models), and Google Gemini. More useful APIs will be added in the future.  Contributions with useful APIs are welcome.

Will also add a curated selection of local models, each kept separate for easy use.  Just download the ones you want.

## 📣 Updates

[2025-05-10] ⚒️: Support for the latest Gemini model, `Gemini-2.5-pro-proview-05-06` requires payment. Support the latest `Qwen3`.

[2025-04-12] ⚒️: JoyCaption support added.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-04-12_04-09-15.png)

- `images_dir`: Path to the directory containing images for batch labeling.

- `save_img_prompt_to_folder`: Path to save images and generated prompts. If provided, batch labeled images and prompts will be saved to this folder; otherwise, they will be saved to `images_dir` with the same name as the image. If provided, single images can also be saved to this folder.

Manually download the models and place them in the `LLM` folder:

- [llama-joycaption-alpha-two-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-alpha-two-hf-llava-nf4/tree/main), estimated 8GB VRAM required.
- [llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava/tree/main), estimated 16GB VRAM required.

[2025-04-01] ⚒️: Released version v1.0.0.

- Supports DeepSeek-R1/V3 model API.  Requires an API key. Obtain a key from the [DeepSeek website](https://platform.deepseek.com/api_keys). Then set the environment variable `DEEPSEEK_API_KEY = <your key>`. See [Configuring API Keys Using Environment Variables](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.38b26132lodett#e4cd73d544i3r) for setup instructions.  You can also input the key directly in the node, but **be sure to keep your key confidential and do not expose it in your workflows.**  **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.**

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekr1.png)

- Supports Qwen API.  Requires an API key. Obtain from [Alibaba Cloud Bailian](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3f7d7980x2Vg6r&apiKey=1#/api-key).  Then set the environment variable `DASHSCOPE_API_KEY = <your key>`.  Setup and usage instructions are the same as above. **Be sure to keep your key confidential and do not expose it in your workflows.** **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.** The Reasoning model with the suffix '_R' has a thought process.

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen1.png)

- Supports Gemini API.  Requires an API key. Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey).  Then set the environment variable `GOOGLE_API_KEY = <your key>`. Setup and usage instructions are the same as above. **Be sure to keep your key confidential and do not expose it in your workflows.** **On Windows, restarting your computer after adding the environment variable may be necessary for the change to take effect.**

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini1.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Prompt-All-In-One.git
cd ComfyUI_Prompt-All-In-One
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Acknowledgments

[joycaption](https://github.com/fpgaminer/joycaption)
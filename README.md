[中文](README-CN.md)|[English](README.md)

# ComfyUI Nodes for Generating Prompts for All Video, Audio, Image, and Text Creations

**Added support for all Ollama models. Any Ollama model can run, including custom models. Locally generate any prompt, reverse-prompt images, videos, and more.** Have fun!

**Added audio/music reverse-prompting and tagging, a powerful tool for audio/music LoRA training. It's fast, effective, and even surpasses API performance.**

**Updated JoyCaption to the latest version, with NF4 support. A blessing for low VRAM users.**

Currently supported APIs: DeepSeek-R1/V3, almost the entire Alibaba Cloud Qwen family, and the entire Google Gemini family. More useful APIs will be added successively.

Selected local models are provided separately for your convenience. Just download what you need.

## 📣 Updates

[2025-06-12]⚒️: Released version v2.0.0.

[2025-05-10]⚒️: Support for the latest Gemini model, `gemini-2.5-pro-preview-05-06`, which requires payment. Support for the latest Qwen3.

[2025-04-12]⚒️: Added support for JoyCaption.

[2025-04-01]⚒️: Released version v1.0.0.

- Supports DeepSeek-R1/V3 model API. You need to apply for an API key on the [DeepSeek official website](https://platform.deepseek.com/api_keys). Then, create a new environment variable `DEEPSEEK_API_KEY = <your key>` in your system's environment variables. For instructions, see [Configure API Key through Environment Variables](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.38b26132lodett#e4cd73d544i3r). Alternatively, you can skip using the system variable and enter the key directly into the node, but **be careful to keep your key confidential and do not leak it with your workflow**. **On Windows, you may need to restart your computer for the environment variable to take effect**.

- Supports Qwen API. Apply for an API key at [Alibaba Cloud Bailian](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3f7d7980x2Vg6r&apiKey=1#/api-key). Then, create a new environment variable `DASHSCOPE_API_KEY = <your key>` in your system's environment variables. The method for adding and using it is the same as above. **Be careful to keep your key confidential and do not leak it with your workflow**. **On Windows, you may need to restart your computer for the environment variable to take effect**. Inference models with the "_R" suffix have a thought process.

- Supports Gemini API. Apply for an API key at [Google AI Studio](https://aistudio.google.com/app/apikey). Then, create a new environment variable `GOOGLE_API_KEY = <your key>` in your system's environment variables. The method for adding and using it is the same as above. **Be careful to keep your key confidential and do not leak it with your workflow**. **On Windows, you may need to restart your computer for the environment variable to take effect**.

## Usage

Audio/Music Reverse-Prompting:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_22-18-44.png)

Ollama Model Flux Prompt Generation:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-47-27.png)

Ollama Model Image/Video Reverse-Prompting:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-31-01.png)
![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-06-12_02-46-28.png)

JoyCaption Image Interrogation/Description:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/2025-04-12_04-09-15.png)

- images_dir: Path for batch-tagging images.

- save_img_prompt_to_folder: Save path for images and prompts. If provided, batch-tagged images and their prompts will be saved to this folder. Otherwise, they are saved by default in `images_dir` with the same name as the image. If provided, even single images can be saved to this folder.

DeepSeek:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/deepseekr1.png)

Qwen:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/qwen1.png)

Gemini:

![](https://github.com/billwuhao/ComfyUI_Prompt-All-In-One/blob/main/images/gemini1.png)

## Model Download

**You do not need to download all models. Only download what you need.**

Manually download the entire folder for the following models into the `LLM` directory:

- JoyCaption:
  - [llama-joycaption-alpha-two-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-alpha-two-hf-llava-nf4/tree/main).
  - [llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava/tree/main).
  - [llama-joycaption-beta-one-hf-llava-nf4](https://huggingface.co/John6666/llama-joycaption-beta-one-hf-llava-nf4).
  - [llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/tree/main).

- Ke-Omni-R-3B:
  - https://hf-mirror.com/KE-Team/Ke-Omni-R-3B/tree/main.
  - Note: **The model is missing an index file. I have generated the index file `model.safetensors.index.json`, which is in this repository. Please move it to the root directory of this model**.

Ollama Models:

First, install [ollama](https://ollama.com/download). Then, any Ollama model can be used, including custom models.

Highly recommended local consumer-grade models: powerful, fast, and versatile. Run the following commands to install:

- `ollama pull artifish/llama3.2-uncensored` Uncensored llama3.2.

https://ollama.com/artifish/llama3.2-uncensored

- `ollama pull poluramus/llama-3.2ft_flux-prompting_v0.5` Super powerful Flux prompt generation model.

https://ollama.com/poluramus/llama-3.2ft_flux-prompting_v0.5

- `ollama pull abedalswaity7/flux-prompt` Another super powerful Flux prompt generation model.

https://ollama.com/abedalswaity7/flux-prompt

- `ollama pull qwen2.5vl:7b` Alibaba's super powerful multimodal model, a great tool for image and video reverse-prompting. Multiple parameter versions are available; 7b is excellent, and 3b is ultra-fast.

https://ollama.com/library/qwen2.5vl

- `ollama pull fanyx/openbmb.MiniCPM4-8B-GGUF-Q8_0:latest` A brand new hot release, the "Mini Cannon" from ModelBest (面壁智能), top six on Hugging Face trending, claiming to be the best and fastest model in its parameter class.

https://ollama.com/fanyx/openbmb.MiniCPM4-8B-GGUF-Q8_0

**Thanks to the model authors for their selfless contributions.**

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Prompt-All-In-One.git
cd ComfyUI_Prompt-All-In-One
pip install -r requirements.txt

# For ComfyUI's embedded python
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Acknowledgements

- [joycaption](https://github.com/fpgaminer/joycaption)
- [Ke-Omni-R](https://github.com/shuaijiang/Ke-Omni-R/)
- [ollama](https://ollama.com/download)
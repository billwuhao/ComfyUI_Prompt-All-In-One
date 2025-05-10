import os
from openai import OpenAI
import base64
from PIL import Image
import io
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import torch


def imgtensor_to_base64(tensor):
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    img_array = tensor.cpu().squeeze().to(torch.uint8).numpy()
    img = Image.fromarray(img_array, mode='RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def audio_tensor_to_mp3_base64(audio_tensor, sample_rate=44100):
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=0)  # 取平均转为单通道
    # 转换为 numpy 数组并调整范围
    audio_data = audio_tensor.cpu().numpy()
    if audio_data.max() <= 1.0:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # 创建 WAV 缓冲区
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_data)
    
    # 转换为 MP3
    audio = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")

    return base64.b64encode(mp3_buffer.getvalue()).decode('utf-8')


class APIQwenTextGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "qwen-max-latest",
                        "qwen2.5-7b-instruct-1m",
                        "qwen2.5-14b-instruct-1m",
                        "deepseek-v3",
                        "qwen-plus-latest",
                        "qwen-turbo-latest",
                        "qwen3-235b-a22b",
                        "qwen3-30b-a3b"
                    ],
                    {"default": "qwen-max-latest"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}],
            )
            
        return (completion.choices[0].message.content,)


class APIQwenTextGen_R:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "qwq-plus",
                        "qwq-plus-latest",
                        "deepseek-r1",
                    ],
                    {"default": "deepseek-r1"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "thinking",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        reasoning_content = ""  # 存储完整思考过程
        answer_content = ""     # 存储完整回复
        is_answering = False    # 判断是否进入回复阶段

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True  # 启用流式输出
        )

        # 遍历流式输出的每个 chunk
        for chunk in completion:
            if not chunk.choices:
                continue
            else:
                delta = chunk.choices[0].delta
                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始处理回复内容
                    if delta.content != "" and not is_answering:
                        is_answering = True
                    answer_content += delta.content if delta.content else ""

        # 返回完整结果
        return (answer_content.strip(), reasoning_content.strip(),)


class APIQwenImage2Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "qwen2.5-vl-7b-instruct",
                        "qwen2.5-vl-32b-instruct",
                        "qwen2.5-vl-72b-instruct",
                        "qwen-vl-plus",
                        "qwen-omni-turbo-latest",
                        "qwen2.5-omni-7b",
                    ],
                    {"default": "qwen2.5-vl-32b-instruct"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        image,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{imgtensor_to_base64(image)}"}, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        answer_content = ""     # 存储完整回复

        if "omini" in model:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=["text"],
                stream=True,
            )
            # 遍历流式输出的每个 chunk
            for chunk in completion:
                if not chunk.choices:
                    continue
                else:
                    delta = chunk.choices[0].delta
                    answer_content += delta.content if delta.content else ""

            return (answer_content.strip(),)

        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return (completion.choices[0].message.content,)


class APIQwenImage2Text_R:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        # "qvq-max-latest",
                        "qvq-max",
                    ],
                    {"default": "qvq-max"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "thinking",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        image,
        api_key,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        reasoning_content = ""  # 存储完整思考过程
        answer_content = ""     # 存储完整回复
        is_answering = False    # 判断是否进入回复阶段

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{imgtensor_to_base64(image)}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=True,  # 启用流式输出
        )

        # 遍历流式输出的每个 chunk
        for chunk in completion:
            if not chunk.choices:
                continue
            else:
                delta = chunk.choices[0].delta
                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始处理回复内容
                    if delta.content != "" and not is_answering:
                        is_answering = True
                    answer_content += delta.content if delta.content else ""

        # 返回完整结果
        return (answer_content.strip(), reasoning_content.strip(),)


class APIQwenImgOrVideo2Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "qwen-vl-plus-latest",
                        "qwen-vl-max-latest",
                        "qwen-omni-turbo-latest",
                        "qwen2.5-omni-7b",
                    ],
                    {"default": "qwen-vl-plus-latest"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
        image = None,
        video = None,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{imgtensor_to_base64(image)}"}, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        if video is not None:
            messages[1]["content"][0] = {
                "type": "video",
                "video": [f"data:image/png;base64,{imgtensor_to_base64(image)}" for image in video],
            }

        answer_content = ""     # 存储完整回复

        if "omni" in model:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=["text"],
                stream=True,
            )
            # 遍历流式输出的每个 chunk
            for chunk in completion:
                if not chunk.choices:
                    continue
                else:
                    delta = chunk.choices[0].delta
                    answer_content += delta.content if delta.content else ""

            return (answer_content.strip(),)

        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
                
            return (completion.choices[0].message.content,)


class APIQwenAudio2Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "qwen-omni-turbo-latest",
                        "qwen2.5-omni-7b",
                    ],
                    {"default": "qwen-omni-turbo-latest"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🎤MW/MW-Prompt-All-In-One"

    def generate(
        self,
        audio,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DASHSCOPE_API_KEY") is not None:
            API_KEY = os.getenv("DASHSCOPE_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        audio_data = audio_tensor_to_mp3_base64(
                                audio["waveform"].squeeze(0),
                                audio["sample_rate"]
                            )
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio_data}",
                            "format": "mp3",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        answer_content = ""     # 存储完整回复
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            modalities=["text"],
            stream=True,
        )
        # 遍历流式输出的每个 chunk
        for chunk in completion:
            if not chunk.choices:
                continue
            else:
                delta = chunk.choices[0].delta
                answer_content += delta.content if delta.content else ""

        return (answer_content.strip(),)
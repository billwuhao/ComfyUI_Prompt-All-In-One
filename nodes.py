from .qwen import (
    APIQwenTextGen,
    APIQwenTextGen_R,
    APIQwenImage2Text,
    APIQwenImage2Text_R,
    APIQwenImgOrVideo2Text,
    APIQwenAudio2Text,
)
from .deepseek import DeepSeekV3, DeepSeekR1
from .gemini import APIGeminiTextGen, APIGeminiImgOrAudioOrVideo2Text, APIGeminiImageGen, APIGeminiTextUnderstand
from .joycaption import JoyCaptionRun
from .ollama_prompt_gen import OllamaPromptGen
from .audio_to_prompt import KeOmniRRun, MultiLinePromptKOR


NODE_CLASS_MAPPINGS = {
    "APIGeminiTextUnderstand": APIGeminiTextUnderstand,
    "DeepSeekV3": DeepSeekV3,
    "DeepSeekR1": DeepSeekR1,
    "APIQwenTextGen": APIQwenTextGen,
    "APIQwenTextGen_R": APIQwenTextGen_R,
    "APIQwenImage2Text": APIQwenImage2Text,
    "APIQwenImage2Text_R": APIQwenImage2Text_R,
    "APIQwenImgOrVideo2Text": APIQwenImgOrVideo2Text,
    "APIQwenAudio2Text": APIQwenAudio2Text,
    "APIGeminiTextGen": APIGeminiTextGen,
    "APIGeminiImgOrAudioOrVideo2Text": APIGeminiImgOrAudioOrVideo2Text,
    "APIGeminiImageGen": APIGeminiImageGen,
    "JoyCaptionRun": JoyCaptionRun,
    "OllamaPromptGen": OllamaPromptGen,
    "KeOmniRRun": KeOmniRRun,
    "MultiLinePromptKOR": MultiLinePromptKOR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIGeminiTextUnderstand": "API Gemini Text Understand",
    "DeepSeekV3": "DeepSeek V3",
    "DeepSeekR1": "DeepSeek R1",
    "APIQwenTextGen": "API Qwen Text Gen",
    "APIQwenTextGen_R": "API Qwen Text Gen_R",
    "APIQwenImage2Text": "API Qwen Image2Text",
    "APIQwenImage2Text_R": "API Qwen Image2Text_R",
    "APIQwenImgOrVideo2Text": "API Qwen ImgOrVideo2Text",
    "APIQwenAudio2Text": "API Qwen Audio2Text",
    "APIGeminiTextGen": "API Gemini Text Gen",
    "APIGeminiImgOrAudioOrVideo2Text": "API Gemini ImgOrAudioOrVideo2Text",
    "APIGeminiImageGen": "API Gemini Image Gen",
    "JoyCaptionRun": "JoyCaption Run",
    "OllamaPromptGen": "Ollama Prompt Generate",
    "KeOmniRRun": "Ke-Omni-R Audio2Text",
    "MultiLinePromptKOR": "MultiLine Prompt",
}
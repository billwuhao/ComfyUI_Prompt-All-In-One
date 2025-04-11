from .qwen import (
    APIQwenTextGen,
    APIQwenTextGen_R,
    APIQwenImage2Text,
    APIQwenImage2Text_R,
    APIQwenImgOrVideo2Text,
    APIQwenAudio2Text,
)
from .deepseek import DeepSeekV3, DeepSeekR1
from .gemini import APIGeminiTextGen, APIGeminiImgOrAudioOrVideo2Text, APIGeminiImageGen
from .joycaption import JoyCaptionRun

NODE_CLASS_MAPPINGS = {
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
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
    "JoyCaptionRun": "JoyCaption Run"
}
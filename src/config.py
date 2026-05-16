import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def get_llm(provider=None):
    """
    Get LLM instance, convenient for switching models in paper experiments.
    Supported model providers:
    - zhipu (default): Zhipu AI (GLM series)
    - deepseek: DeepSeek model
    - glm4.7: Zhipu AI GLM-4.7
    - chatanywhere: ChatAnywhere API (fill in CHATANYWHERE_API_KEY)
    - qwen3: Alibaba Qwen3
    
    Args:
        provider: Model provider, if None read from environment variable LLM_PROVIDER
    """
    # Get model provider from argument or environment variable, default is zhipu
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "zhipu")
    
    if provider == "deepseek":
        # DeepSeek configuration
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    elif provider == "glm4.7":
        # GLM-4.7 configuration
        api_key = os.getenv("GLM47_API_KEY", "")  # Fill in your GLM-4.7 API Key
        base_url = os.getenv("GLM47_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        model_name = os.getenv("GLM47_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
    elif provider == "glm4.7-no-thinking":
        api_key = os.getenv("GLM47_API_KEY", "")
        base_url = os.getenv("GLM47_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        model_name = os.getenv("GLM47_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    elif provider == "chatanywhere":
        # ChatAnywhere API configuration (fill in your API Key)
        api_key = os.getenv("CHATANYWHERE_API_KEY", "")
        base_url = os.getenv("CHATANYWHERE_BASE_URL", "https://api.chatanywhere.tech/v1")
        model_name = os.getenv("CHATANYWHERE_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
    elif provider == "chatanywhere2":
        # ChatAnywhere2 API configuration (using the second API Key)
        api_key = os.getenv("CHATANYWHERE_API_KEY_2", "")
        base_url = os.getenv("CHATANYWHERE_BASE_URL", "https://api.chatanywhere.tech/v1")
        model_name = os.getenv("CHATANYWHERE_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
    elif provider == "aliyun":
        api_key = os.getenv("Aliyun-GLM47_API_KEY", "")
        base_url = os.getenv("Aliyun-GLM47_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("Aliyun-GLM47_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
    elif provider == "Aliyun-glm4.7":
        api_key = os.getenv("Aliyun-GLM47_API_KEY", "")
        base_url = os.getenv("Aliyun-GLM47_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("Aliyun-GLM47_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )
    elif provider == "Aliyun-glm4.7-no-thinking":
        api_key = os.getenv("Aliyun-GLM47_API_KEY", "")
        base_url = os.getenv("Aliyun-GLM47_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("Aliyun-GLM47_MODEL_NAME", "glm-4.7")

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    elif provider == "qwen3":
        # Qwen3 configuration
        api_key = os.getenv("QWEN3_API_KEY", "")  # Fill in your Qwen3 API Key
        base_url = os.getenv("QWEN3_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("QWEN3_MODEL_NAME", "qwen-max")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    elif provider == "qwen3-coder-480b":
        # Qwen3-Coder-480B configuration
        api_key = os.getenv("QWEN3_CODER_480B_API_KEY", "")  # Fill in your Qwen3-Coder-480B API Key
        base_url = os.getenv("QWEN3_CODER_480B_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("QWEN3_CODER_480B_MODEL_NAME", "qwen3-coder-480b-a35b-instruct")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    elif provider == "qwen3-coder-chatanywhere":
        # Qwen3-Coder-480B via ChatAnywhere API (using specified Key)
        api_key = os.getenv("CHATANYWHERE_API_KEY", "")
        base_url = "https://api.chatanywhere.tech/v1"
        model_name = "qwen3-coder-480b-a35b-instruct"

        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url
        )
    else:
        # Zhipu AI configuration (default)
        api_key = os.getenv("OPENAI_API_KEY") 
        base_url = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        model_name = os.getenv("OPENAI_MODEL_NAME", "glm-4.7-flash")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.95,
            api_key=api_key,
            base_url=base_url,
            extra_body={
                "thinking": {
                    "type": "enabled"
                }
            }
        )

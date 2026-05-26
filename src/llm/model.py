from langchain.chat_models import init_chat_model

from llm.config import ChatModelSettings

_settings = ChatModelSettings()

model = init_chat_model(
    model=_settings.normalized_model,
    model_provider="openai",
    base_url=_settings.normalized_base_url,
    api_key=_settings.normalized_api_key,
)

import logging
import os
import json
import random
import time
import asyncio
from functools import wraps
from typing import List, Dict, Any, Union, AsyncGenerator, Optional

from pydantic import BaseModel

try:
    from pydantic_ai.models import Model
except Exception:
    class _AnyModel:
        pass
    Model = _AnyModel

try:
    from litellm import litellm, AsyncOpenAI, acompletion
except Exception:
    class _LiteLLMStub:
        num_retries = 0
    litellm = _LiteLLMStub()
    AsyncOpenAI = None
    async def acompletion(*args, **kwargs):
        raise RuntimeError("litellm not available")

try:
    import instructor
except Exception:
    instructor = None

logger = logging.getLogger(__name__)

from src.api.v1.provider.schemas import (
    ProviderInfo,
    GetProviderResponse,
    AvailableModelsResponse,
    AvailableModelOption,
    SetProviderRequest,
    ModelInfo,
)
from src.infrastructure.llm.llm_config import (
    LLMProviderConfig,
    build_llm_provider_config,
    get_config_for_model,
)
from src.infrastructure.llm.exceptions import UnsupportedProviderError

try:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.anthropic import AnthropicProvider
except Exception:
    OpenAIModel = AnthropicModel = OpenAIProvider = AnthropicProvider = None

try:
    import litellm as _litellm_mod
    litellm = _litellm_mod
except Exception:
    pass


class _ConfigProvider:
    def get_is_multimodal_enabled(self) -> bool:
        val = os.environ.get("LLM_SUPPORTS_VISION")
        if val is None:
            return False
        return val.strip().lower() in {"1", "true", "yes", "on"}


config_provider = _ConfigProvider()

litellm.num_retries = getattr(litellm, "num_retries", 5)

OVERLOAD_ERROR_PATTERNS = {
    "anthropic": ["overloaded", "overloaded_error", "capacity", "rate limit exceeded"],
    "openai": [
        "rate_limit_exceeded",
        "capacity",
        "overloaded",
        "server_error",
        "timeout",
    ],
    "general": [
        "timeout",
        "insufficient capacity",
        "server_error",
        "internal_server_error",
    ],
}


class RetrySettings:
    def __init__(
        self,
        max_retries: int = 8,
        min_delay: float = 1.0,
        max_delay: float = 120.0,
        base_delay: float = 2.0,
        jitter_factor: float = 0.2,
        step_increase: float = 1.8,
        retry_on_timeout: bool = True,
        retry_on_overloaded: bool = True,
        retry_on_rate_limit: bool = True,
        retry_on_server_error: bool = True,
    ):
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.base_delay = base_delay
        self.jitter_factor = jitter_factor
        self.step_increase = step_increase
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_overloaded = retry_on_overloaded
        self.retry_on_rate_limit = retry_on_rate_limit
        self.retry_on_server_error = retry_on_server_error


def identify_provider_from_error(error: Exception) -> str:
    error_str = str(error).lower()
    for provider in ["anthropic", "openai", "cohere", "azure"]:
        if provider.lower() in error_str.lower():
            return provider
    return "unknown"


def is_recoverable_error(error: Exception, settings: RetrySettings) -> bool:
    error_str = str(error).lower()
    provider = identify_provider_from_error(error)
    if settings.retry_on_timeout and "timeout" in error_str:
        return True
    if settings.retry_on_overloaded:
        overload_patterns = (
            OVERLOAD_ERROR_PATTERNS.get(provider, [])
            + OVERLOAD_ERROR_PATTERNS["general"]
        )
        if any(pattern in error_str for pattern in overload_patterns):
            return True
    if settings.retry_on_rate_limit and any(
        limit_pattern in error_str
        for limit_pattern in [
            "rate limit",
            "rate_limit",
            "ratelimit",
            "requests per minute",
        ]
    ):
        return True
    if settings.retry_on_server_error and any(
        server_err in error_str
        for server_err in [
            "server_error",
            "internal_server_error",
            "500",
            "502",
            "503",
            "504",
        ]
    ):
        return True
    return False


def calculate_backoff_time(retry_count: int, settings: RetrySettings) -> float:
    delay = min(settings.max_delay, settings.base_delay * (settings.step_increase**retry_count))
    jitter = random.uniform(1 - settings.jitter_factor, 1 + settings.jitter_factor)
    final_delay = max(settings.min_delay, min(settings.max_delay, delay * jitter))
    return final_delay


def custom_litellm_retry_handler(retry_count: int, exception: Exception) -> bool:
    settings = RetrySettings(max_retries=getattr(litellm, "num_retries", 5))
    if not is_recoverable_error(exception, settings):
        return False
    delay = calculate_backoff_time(retry_count, settings)
    provider = identify_provider_from_error(exception)
    logging.warning(
        f"{provider.capitalize()} API error: {str(exception)}. "
        f"Retry {retry_count}/{settings.max_retries}, "
        f"waiting {delay:.2f}s before next attempt..."
    )
    time.sleep(delay)
    return True


def robust_llm_call(settings: Optional[RetrySettings] = None):
    if settings is None:
        settings = RetrySettings()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries <= settings.max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not is_recoverable_error(e, settings):
                        raise
                    provider = identify_provider_from_error(e)
                    if retries >= settings.max_retries:
                        logging.error(
                            f"Max retries ({settings.max_retries}) exceeded for {provider} API call. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    delay = calculate_backoff_time(retries, settings)
                    logging.warning(
                        f"{provider.capitalize()} API error: {str(e)}. "
                        f"Retry {retries+1}/{settings.max_retries}, "
                        f"waiting {delay:.2f}s before next attempt..."
                    )
                    await asyncio.sleep(delay)
                    retries += 1
            raise last_exception
        return wrapper
    return decorator


AVAILABLE_MODELS = [
    AvailableModelOption(
        id="openai/gpt-4.1",
        name="GPT-4.1",
        description="OpenAI's latest model for complex tasks with large context",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-4o",
        name="GPT-4o",
        description="High-intelligence model for complex tasks",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-4.1-mini",
        name="GPT-4.1 Mini",
        description="Smaller model for fast, lightweight tasks",
        provider="openai",
        is_chat_model=False,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openai/o4-mini",
        name="O4 mini",
        description="reasoning model",
        provider="openai",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        description="Best model for complex agents and coding",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        description="Faster, even surpasses Claude Sonnet 4 at certain tasks",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-opus-4-1-20250805",
        name="Claude Opus 4.1",
        description="Exceptional model for specialized complex tasks",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        description="Faster, more efficient Claude model for code generation",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-3-7-sonnet-20250219",
        name="Claude Sonnet 3.7",
        description="Highest level of intelligence and capability with toggleable extended thinking",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-3-5-haiku-20241022",
        name="Claude Haiku 3.5",
        description="Faster, more efficient Claude model",
        provider="anthropic",
        is_chat_model=False,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/deepseek/deepseek-chat-v3-0324",
        name="DeepSeek V3",
        description="DeepSeek's latest chat model",
        provider="deepseek",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/meta-llama/llama-3.3-70b-instruct",
        name="Llama 3.3 70B",
        description="Meta's latest Llama model",
        provider="meta-llama",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        description="Google's Gemini model optimized for speed",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-2.5-pro-preview",
        name="Gemini 2.5 Pro",
        description="Google's Latest pro Gemini model",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
]

PLATFORM_PROVIDERS = list(
    {model.provider for model in AVAILABLE_MODELS}
    | {
        get_config_for_model(model.id).get("auth_provider", model.provider)
        for model in AVAILABLE_MODELS
    }
)


class ProviderService:
    def __init__(self, user_id: str):
        try:
            litellm.modify_params = True
        except Exception:
            pass
        self.user_id = user_id
        user_config: Dict[str, Any] = {}
        self.chat_config = build_llm_provider_config(user_config, config_type="chat")
        self.inference_config = build_llm_provider_config(user_config, config_type="inference")
        self.retry_settings = RetrySettings(max_retries=8, base_delay=2.0, max_delay=120.0)

    @classmethod
    def create(cls, user_id: str):
        return cls(user_id)

    async def list_available_llms(self) -> List[ProviderInfo]:
        providers = {
            model.provider: ProviderInfo(
                id=model.provider,
                name=model.provider,
                description=f"Provider for {model.provider} models",
            )
            for model in AVAILABLE_MODELS
        }
        return list(providers.values())

    async def list_available_models(self) -> AvailableModelsResponse:
        return AvailableModelsResponse(models=AVAILABLE_MODELS)

    async def set_global_ai_provider(self, request: SetProviderRequest):
        if request.chat_model:
            os.environ["CHAT_MODEL"] = request.chat_model
            self.chat_config = build_llm_provider_config({"chat_model": request.chat_model}, "chat")
        if request.inference_model:
            os.environ["INFERENCE_MODEL"] = request.inference_model
            self.inference_config = build_llm_provider_config({"inference_model": request.inference_model}, "inference")
        return {"message": "AI provider configuration updated successfully"}

    def _get_api_key(self, provider: str) -> str | None:
        env_key = os.getenv("LLM_API_KEY", None)
        if env_key:
            return env_key
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key
        return None

    def _build_llm_params(self, config: LLMProviderConfig) -> Dict[str, Any]:
        api_key = self._get_api_key(config.auth_provider)
        if not api_key and config.auth_provider == "ollama":
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", api_key)
        params = config.get_llm_params(api_key)
        if config.base_url:
            base_url = config.base_url
            if config.auth_provider == "ollama":
                base_url = base_url.rstrip("/")
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
            params["base_url"] = base_url
        elif config.auth_provider == "ollama":
            params["base_url"] = os.environ.get("LLM_API_BASE", "http://localhost:11434")
        if config.api_version:
            params["api_version"] = config.api_version
        if config.auth_provider == "openrouter":
            referer = os.environ.get("OPENROUTER_SITE_URL")
            title = os.environ.get("OPENROUTER_APP_TITLE")
            if not referer or not title:
                from src.config.settings import settings
                referer = referer or "http://localhost:8000"
                title = title or getattr(settings, "app_name", "AI Backend")
            extra_headers = {
                "HTTP-Referer": referer,
                "X-Title": title,
                "Content-Type": "application/json",
            }
            if api_key:
                extra_headers["Authorization"] = f"Bearer {api_key}"
            params["extra_headers"] = extra_headers
        return {key: value for key, value in params.items() if value is not None}

    def _build_config_for_model_identifier(self, model_identifier: str) -> LLMProviderConfig:
        config_data = get_config_for_model(model_identifier).copy()
        default_params = dict(config_data.get("default_params", {}))
        return LLMProviderConfig(
            provider=config_data["provider"],
            model=model_identifier,
            default_params=default_params,
            capabilities=config_data.get("capabilities", {}),
            base_url=config_data.get("base_url"),
            api_version=config_data.get("api_version"),
            auth_provider=config_data.get("auth_provider"),
        )

    async def get_global_ai_provider(self) -> GetProviderResponse:
        chat_model_id = os.environ.get("CHAT_MODEL") or "openai/gpt-4o"
        inference_model_id = os.environ.get("INFERENCE_MODEL") or "openai/gpt-4.1-mini"
        chat_provider = chat_model_id.split("/")[0] if chat_model_id else ""
        chat_model_name = chat_model_id
        inference_provider = inference_model_id.split("/")[0] if inference_model_id else ""
        inference_model_name = inference_model_id
        for model in AVAILABLE_MODELS:
            if model.id == chat_model_id:
                chat_model_name = model.name
                chat_provider = model.provider
            if model.id == inference_model_id:
                inference_model_name = model.name
                inference_provider = model.provider
        return GetProviderResponse(
            chat_model=ModelInfo(provider=chat_provider, id=chat_model_id, name=chat_model_name),
            inference_model=ModelInfo(provider=inference_provider, id=inference_model_id, name=inference_model_name),
        )

    def supports_pydantic(self, config_type: str = "chat") -> bool:
        config = self.chat_config if config_type == "chat" else self.inference_config
        return config.capabilities.get("supports_pydantic", False)

    @robust_llm_call()
    async def call_llm_with_specific_model(
        self,
        model_identifier: str,
        messages: list,
        output_schema: Optional[BaseModel] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, AsyncGenerator[str, None], Any]:
        config = self._build_config_for_model_identifier(model_identifier)
        params = self._build_llm_params(config)
        params.update(kwargs)
        routing_provider = config.provider
        try:
            if output_schema:
                request_kwargs = {key: params[key] for key in ("api_key", "base_url", "api_version") if key in params}
                if config.provider == "ollama":
                    ollama_base_root = (
                        params.get("base_url")
                        or config.base_url
                        or os.environ.get("LLM_API_BASE")
                        or "http://localhost:11434"
                    )
                    ollama_base_url = ollama_base_root.rstrip("/") + "/v1"
                    ollama_api_key = params.get("api_key") or os.environ.get("OLLAMA_API_KEY", "ollama")
                    client = instructor.from_openai(
                        AsyncOpenAI(base_url=ollama_base_url, api_key=ollama_api_key),
                        mode=instructor.Mode.JSON,
                    )
                    ollama_request_kwargs = {key: value for key, value in request_kwargs.items() if key not in {"base_url", "api_key", "api_version"}}
                    response = await client.chat.completions.create(
                        model=params["model"].split("/")[-1],
                        messages=messages,
                        response_model=output_schema,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **ollama_request_kwargs,
                    )
                else:
                    client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
                    response = await client.chat.completions.create(
                        model=params["model"],
                        messages=messages,
                        response_model=output_schema,
                        strict=True,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **request_kwargs,
                    )
                return response
            else:
                if stream:
                    async def generator() -> AsyncGenerator[str, None]:
                        response = await acompletion(messages=messages, stream=True, **params)
                        async for chunk in response:
                            yield chunk.choices[0].delta.content or ""
                    return generator()
                else:
                    response = await acompletion(messages=messages, **params)
                    return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling LLM with model {model_identifier}: {e}, provider: {routing_provider}")
            raise e

    @robust_llm_call()
    async def call_llm(self, messages: list, stream: bool = False, config_type: str = "chat") -> Union[str, AsyncGenerator[str, None]]:
        config = self.chat_config if config_type == "chat" else self.inference_config
        params = self._build_llm_params(config)
        routing_provider = config.provider
        try:
            if stream:
                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(messages=messages, stream=True, **params)
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""
                return generator()
            else:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling LLM: {e}, provider: {routing_provider}")
            raise e

    @robust_llm_call()
    async def call_llm_with_structured_output(self, messages: list, output_schema: BaseModel, config_type: str = "chat") -> Any:
        config = self.chat_config if config_type == "chat" else self.inference_config
        params = self._build_llm_params(config)
        routing_provider = config.provider
        request_kwargs = {key: params[key] for key in ("api_key", "base_url", "api_version") if key in params}
        try:
            if instructor is not None:
                if config.provider == "ollama":
                    ollama_base_root = (
                        params.get("base_url")
                        or config.base_url
                        or os.environ.get("LLM_API_BASE")
                        or "http://localhost:11434"
                    )
                    ollama_base_url = ollama_base_root.rstrip("/") + "/v1"
                    ollama_api_key = params.get("api_key") or os.environ.get("OLLAMA_API_KEY", "ollama")
                    client = instructor.from_openai(
                        AsyncOpenAI(base_url=ollama_base_url, api_key=ollama_api_key),
                        mode=instructor.Mode.JSON,
                    )
                    ollama_request_kwargs = {key: value for key, value in request_kwargs.items() if key not in {"base_url", "api_key", "api_version"}}
                    response = await client.chat.completions.create(
                        model=params["model"].split("/")[-1],
                        messages=messages,
                        response_model=output_schema,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **ollama_request_kwargs,
                    )
                else:
                    client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
                    response = await client.chat.completions.create(
                        model=params["model"],
                        messages=messages,
                        response_model=output_schema,
                        strict=True,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **request_kwargs,
                    )
                return response
            else:
                fields = []
                try:
                    if hasattr(output_schema, "model_fields"):
                        fields = list(output_schema.model_fields.keys())
                    elif hasattr(output_schema, "__fields__"):
                        fields = list(output_schema.__fields__.keys())
                except Exception:
                    fields = []
                extra = {
                    "role": "system",
                    "content": "Return a JSON object with keys: " + ", ".join(fields or ["category", "confidence", "reason"]) + ".",
                }
                m = messages + [extra]
                resp = await acompletion(messages=m, **params)
                content = resp.choices[0].message.content
                data = {}
                try:
                    data = json.loads(content)
                except Exception:
                    for ln in content.splitlines():
                        if ":" in ln:
                            k, v = ln.split(":", 1)
                            data[k.strip().lower()] = v.strip()
                    if "category" not in data:
                        data["category"] = "OTHERS"
                try:
                    mapped = {}
                    for k in (fields or ["category", "confidence", "reason"]):
                        mapped[k] = data.get(k) or data.get(k.lower())
                    return output_schema(**mapped)
                except Exception:
                    return output_schema(**{"category": "OTHERS", "confidence": None, "reason": None})
        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            return output_schema(**{"category": "OTHERS", "confidence": None, "reason": None})

    @robust_llm_call()
    async def call_llm_multimodal(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[Dict[str, Dict[str, Union[str, int]]]] = None,
        stream: bool = False,
        config_type: str = "chat",
    ) -> Union[str, AsyncGenerator[str, None]]:
        if not config_provider.get_is_multimodal_enabled():
            logger.info("Multimodal disabled - falling back to text-only processing")
            return await self.call_llm(messages, stream=stream, config_type=config_type)
        if not images:
            return await self.call_llm(messages, stream=stream, config_type=config_type)
        config = self.chat_config if config_type == "chat" else self.inference_config
        params = self._build_llm_params(config)
        routing_provider = config.provider
        if images:
            validated_images = self._validate_images_for_multimodal(images)
            if validated_images:
                messages = self._format_multimodal_messages(messages, validated_images, routing_provider)
            else:
                images = None
        try:
            if stream:
                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(messages=messages, stream=True, **params)
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""
                return generator()
            else:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling multimodal LLM: {e}, provider: {routing_provider}")
            raise e

    def _format_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        images: Dict[str, Dict[str, Union[str, int]]],
        provider: str,
    ) -> List[Dict[str, Any]]:
        if not images:
            return messages
        formatted_messages = []
        for message in messages:
            if message.get("role") == "user" and len(formatted_messages) == len(messages) - 1:
                formatted_message = self._format_multimodal_message(message, images, provider)
                formatted_messages.append(formatted_message)
            else:
                formatted_messages.append(message)
        return formatted_messages

    def _format_multimodal_message(
        self,
        message: Dict[str, Any],
        images: Dict[str, Dict[str, Union[str, int]]],
        provider: str,
    ) -> Dict[str, Any]:
        text_content = message.get("content", "")
        if provider == "openai":
            return self._format_openai_multimodal_message(text_content, images)
        elif provider == "anthropic":
            return self._format_anthropic_multimodal_message(text_content, images)
        elif provider == "gemini":
            return self._format_gemini_multimodal_message(text_content, images)
        else:
            return self._format_openai_multimodal_message(text_content, images)

    def _format_openai_multimodal_message(self, text: str, images: Dict[str, Dict[str, Union[str, int]]]) -> Dict[str, Any]:
        content = [{"type": "text", "text": text}]
        for attachment_id, image_data in images.items():
            mime_type = image_data.get("mime_type", "image/jpeg")
            base64_data = image_data["base64"]
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}", "detail": "high"}})
        return {"role": "user", "content": content}

    def _format_anthropic_multimodal_message(self, text: str, images: Dict[str, Dict[str, Union[str, int]]]) -> Dict[str, Any]:
        content = []
        for attachment_id, image_data in images.items():
            mime_type = image_data.get("mime_type", "image/jpeg")
            base64_data = image_data["base64"]
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": base64_data}})
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def _format_gemini_multimodal_message(self, text: str, images: Dict[str, Dict[str, Union[str, int]]]) -> Dict[str, Any]:
        return self._format_openai_multimodal_message(text, images)

    def _validate_images_for_multimodal(self, images: Dict[str, Dict[str, Union[str, int]]]) -> Dict[str, Dict[str, Union[str, int]]]:
        validated_images = {}
        for img_id, img_data in images.items():
            try:
                if "base64" not in img_data or not img_data["base64"]:
                    continue
                base64_data = str(img_data["base64"])
                if len(base64_data) < 100:
                    continue
                if len(base64_data) > 10_000_000:
                    continue
                mime_type = img_data.get("mime_type", "")
                supported_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
                if mime_type not in supported_types:
                    continue
                if not base64_data.replace("+", "").replace("/", "").replace("=", "").isalnum():
                    continue
                validated_images[img_id] = img_data
            except Exception:
                continue
        return validated_images

    def is_vision_model(self, config_type: str = "chat") -> bool:
        if not config_provider.get_is_multimodal_enabled():
            return False
        config = self.chat_config if config_type == "chat" else self.inference_config
        model_name = config.model.lower()
        vision_models = [
            "gpt-4-vision",
            "gpt-4v",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o4-mini",
            "claude-3",
            "claude-3-sonnet",
            "claude-3-opus",
            "claude-3-haiku",
            "claude-sonnet-4",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "gemini-pro-vision",
            "gemini-1.5",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0",
            "gemini-2.0-flash",
            "gemini-2.5",
            "gemini-2.5-pro",
            "gemini-ultra",
            "deepseek-chat",
            "llama-3.3",
            "llama-3.3-70b",
            "llama-3.3-8b",
        ]
        return any(vision_model in model_name for vision_model in vision_models)

    def get_pydantic_model(self, provider: str | None = None, model: str | None = None) -> Model | None:
        target_model = model or self.chat_config.model
        config = self._build_config_for_model_identifier(target_model)
        if provider:
            config.provider = provider
            config.auth_provider = provider
        api_key = self._get_api_key(config.auth_provider)
        if not api_key and config.auth_provider == "ollama":
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", api_key)
        if not api_key and config.auth_provider not in {"ollama"}:
            raise UnsupportedProviderError(f"API key not found for provider '{config.auth_provider}'.")
        model_name = target_model.split("/", 1)[1] if "/" in target_model else target_model
        if not config.capabilities.get("supports_pydantic", False):
            raise UnsupportedProviderError(f"Model '{target_model}' does not support Pydantic-based agents.")
        provider_kwargs: Dict[str, Any] = {}
        if config.base_url:
            provider_kwargs["base_url"] = config.base_url
        if config.api_version:
            provider_kwargs["api_version"] = config.api_version
        openai_like_providers = {"openai", "openrouter", "azure", "ollama"}
        if config.auth_provider in openai_like_providers:
            if config.auth_provider == "ollama":
                base_url_root = config.base_url or os.environ.get("LLM_API_BASE") or "http://localhost:11434"
                provider_kwargs["base_url"] = base_url_root.rstrip("/") + "/v1"
            return OpenAIModel(model_name=model_name, provider=OpenAIProvider(api_key=api_key, **provider_kwargs))
        if config.provider == "anthropic":
            anthropic_kwargs = {key: value for key, value in provider_kwargs.items() if key != "api_version"}
            return AnthropicModel(model_name=model_name, provider=AnthropicProvider(api_key=api_key, **anthropic_kwargs))
        raise UnsupportedProviderError(f"Provider '{config.provider}' is not supported for Pydantic-based agents.")
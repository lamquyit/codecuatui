"""
LLM Provider Fallback Module
──────────────────────────────
Mặc định gọi OpenRouter. Nếu lỗi → tự động fallback sang OpenAI.
Không gọi API thừa để check — chỉ fallback khi request thực sự thất bại.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ── Provider Configuration ───────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Cấu hình cho một LLM provider."""
    name: str
    api_key: str
    base_url: Optional[str]
    llm_model: str           # model cho text completion
    vision_model: str        # model cho vision tasks
    embedding_model: str     # model cho embeddings
    embedding_dim: int = 3072


# Model mapping: OpenRouter yêu cầu prefix "openai/" trước tên model
OPENROUTER_MODELS = {
    "llm": "openai/gpt-4o-mini",
    "vision": "openai/gpt-4o",
    "embedding": "openai/text-embedding-3-large",
}

OPENAI_MODELS = {
    "llm": "gpt-4o-mini",
    "vision": "gpt-4o",
    "embedding": "text-embedding-3-large",
}


def get_providers() -> list[ProviderConfig]:
    """
    Trả về danh sách providers theo thứ tự ưu tiên:
      1. OpenRouter (nếu có OPENROUTER_API_KEY)
      2. OpenAI    (nếu có OPENAI_API_KEY)
    """
    providers = []

    openrouter_key = os.getenv("OPENAI_API_KEY_opr")
    if openrouter_key:
        providers.append(ProviderConfig(
            name="openrouter",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            llm_model=OPENROUTER_MODELS["llm"],
            vision_model=OPENROUTER_MODELS["vision"],
            embedding_model=OPENROUTER_MODELS["embedding"],
        ))

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers.append(ProviderConfig(
            name="openai",
            api_key=openai_key,
            base_url=None,
            llm_model=OPENAI_MODELS["llm"],
            vision_model=OPENAI_MODELS["vision"],
            embedding_model=OPENAI_MODELS["embedding"],
        ))

    if not providers:
        raise RuntimeError(
            "Không tìm thấy API key nào! "
            "Hãy set OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env"
        )

    return providers


def get_primary_provider() -> ProviderConfig:
    """Trả về provider đầu tiên (ưu tiên cao nhất)."""
    return get_providers()[0]


# ── Fallback Wrapper cho LLM Completion ──────────────────────────────────────

async def openai_complete_with_fallback(
    model_type: str,   # "llm" hoặc "vision"
    prompt: str,
    system_prompt=None,
    history_messages=None,
    **kwargs,
):
    """
    Gọi openai_complete_if_cache với fallback qua các providers.
    
    - model_type: "llm" → dùng llm_model, "vision" → dùng vision_model
    - Thử provider đầu tiên, nếu lỗi → thử provider tiếp theo
    - KHÔNG gọi API thừa để check — chỉ fallback khi request thất bại
    """
    from lightrag.llm.openai import openai_complete_if_cache

    providers = get_providers()
    history_messages = history_messages or []
    last_error = None

    for provider in providers:
        model_name = provider.llm_model if model_type == "llm" else provider.vision_model
        try:
            result = await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=provider.api_key,
                base_url=provider.base_url,
                **kwargs,
            )
            return result
        except Exception as e:
            last_error = e
            print(f"⚠️ [{provider.name}] LLM call failed: {e}")
            if provider != providers[-1]:
                next_provider = providers[providers.index(provider) + 1]
                print(f"   ↳ Falling back to {next_provider.name}...")
                # Nếu model name khác giữa providers, cần loại bỏ model cũ khỏi kwargs
                # để tránh conflict
            continue

    raise RuntimeError(
        f"Tất cả providers đều thất bại. Lỗi cuối: {last_error}"
    )


# ── Fallback Wrapper cho Embedding ───────────────────────────────────────────

async def openai_embed_with_fallback(texts: list[str]) -> list:
    """
    Gọi openai_embed với fallback qua các providers.
    Tự động xử lý None/invalid values trong texts.
    """
    from lightrag.llm.openai import openai_embed

    if texts is None:
        return []

    # Sanitize inputs
    safe_texts = [str(t) if t is not None else "" for t in texts]
    if not safe_texts:
        return []

    providers = get_providers()
    last_error = None

    for provider in providers:
        try:
            result = await openai_embed(
                safe_texts,
                model=provider.embedding_model,
                api_key=provider.api_key,
                base_url=provider.base_url,
            )
            if result is None:
                print(f"⚠️ [{provider.name}] Embedding returned None")
                continue
            return result
        except Exception as e:
            last_error = e
            print(f"⚠️ [{provider.name}] Embedding failed: {e}")
            import traceback
            traceback.print_exc()
            if provider != providers[-1]:
                next_provider = providers[providers.index(provider) + 1]
                print(f"   ↳ Falling back to {next_provider.name}...")
            continue

    # Nếu tất cả thất bại → trả zero vectors thay vì crash
    print(f"❌ All embedding providers failed. Returning zero vectors. Last error: {last_error}")
    dim = providers[0].embedding_dim if providers else 3072
    return [[0.0] * dim] * len(safe_texts)

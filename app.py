# app.py
# ------------------------------------------------------------
# Çoklu Yapay Zeka Yönetim Paneli
# Streamlit Cloud / Mobil / Masaüstü uyumlu tek dosyalık uygulama
# ------------------------------------------------------------
#
# ÖNERİLEN requirements.txt:
# streamlit
# openai
# google-genai
#
# STREAMLIT CLOUD SECRETS ÖRNEĞİ:
# Bu değerleri kodun içine yazmayın. Streamlit Cloud > App > Settings > Secrets
# bölümüne TOML formatında girin.
#
# GEMINI_API_KEY = "..."
# GROQ_API_KEY = "..."
# OPENROUTER_API_KEY = "..."
# OPENAI_API_KEY = "..."  # İsteğe bağlı. Yoksa ChatGPT seçeneği OpenRouter'a düşer.
#
# GEMINI_MODEL = "gemini-3.5-flash"
# LLAMA_MODEL = "llama-3.1-8b-instant"
# DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"
# OPENAI_MODEL = "gpt-5.4-mini"
# OPENROUTER_OPENAI_MODEL = "~openai/gpt-latest"
#
# ROUTER_PROVIDER = "groq"  # "groq" veya "gemini"
# ROUTER_GROQ_MODEL = "llama-3.1-8b-instant"
# ROUTER_GEMINI_MODEL = "gemini-3.5-flash"
#
# ENABLE_GEMINI_SEARCH = true
# OPENROUTER_SITE_URL = "https://your-streamlit-app-url.streamlit.app"
# OPENROUTER_APP_TITLE = "Coklu Yapay Zeka Yonetim Paneli"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import streamlit as st


# ------------------------------------------------------------
# Opsiyonel bağımlılıklar
# ------------------------------------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None


# ------------------------------------------------------------
# Streamlit sayfa ayarları
# ------------------------------------------------------------
st.set_page_config(
    page_title="Çoklu Yapay Zeka Yönetim Paneli",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Mobil / masaüstü arayüz CSS düzenlemeleri
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        [data-testid="stMarkdownContainer"] {
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.1rem;
            border-radius: 0.75rem;
            font-weight: 700;
            font-size: 1rem;
        }

        .stTextArea textarea {
            font-size: 1rem !important;
            line-height: 1.5 !important;
            border-radius: 0.75rem !important;
        }

        .stSelectbox, .stTextArea, .stSlider {
            width: 100%;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 0.9rem;
            }

            h1 {
                font-size: 1.65rem !important;
                line-height: 1.25 !important;
            }

            h2, h3 {
                font-size: 1.2rem !important;
            }

            .stTextArea textarea {
                min-height: 190px !important;
            }

            [data-testid="stSidebar"] {
                min-width: 80vw !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Veri modelleri ve özel hata sınıfları
# ------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    menu_label: str
    public_name: str
    provider: str
    model: str
    api_secret_name: str


class MissingSecretError(Exception):
    pass


class MissingDependencyError(Exception):
    pass


# ------------------------------------------------------------
# Secrets yardımcıları
# ------------------------------------------------------------
def secret_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Streamlit secrets değerini güvenli biçimde okur.
    Secrets dosyası yoksa veya ilgili key eksikse default döndürür.
    """
    try:
        value = st.secrets.get(key, default)
    except Exception:
        return default

    if value is None:
        return default

    return str(value).strip()


def bool_secret(key: str, default: bool = False) -> bool:
    raw = secret_value(key, str(default)).lower()
    return raw in {"1", "true", "yes", "on", "evet", "aktif"}


def has_secret(key: str) -> bool:
    value = secret_value(key)
    return bool(value)


def require_secret(key: str) -> str:
    value = secret_value(key)
    if not value:
        raise MissingSecretError(
            f"`{key}` bulunamadı. Streamlit Cloud Secrets içine bu anahtarı ekleyin."
        )
    return value


def redact_secret_values(text: str) -> str:
    """
    Teknik hata detaylarında API anahtarının görünmesini engellemek için basit maskeleme.
    """
    if not text:
        return ""

    known_secret_keys = [
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ]

    redacted = text
    for key in known_secret_keys:
        value = secret_value(key)
        if value and len(value) >= 8:
            redacted = redacted.replace(value, value[:4] + "***" + value[-4:])

    return redacted


# ------------------------------------------------------------
# Model yapılandırmaları
# ------------------------------------------------------------
def build_chatgpt_config() -> ModelConfig:
    """
    OpenAI API key varsa doğrudan OpenAI Responses API kullanılır.
    OPENAI_API_KEY yoksa ChatGPT seçeneği OpenRouter üzerinden çalışır.
    """
    if has_secret("OPENAI_API_KEY"):
        return ModelConfig(
            menu_label="OpenAI ChatGPT - OpenAI API",
            public_name="OpenAI ChatGPT",
            provider="openai",
            model=secret_value("OPENAI_MODEL", "gpt-5.4-mini"),
            api_secret_name="OPENAI_API_KEY",
        )

    return ModelConfig(
        menu_label="OpenAI ChatGPT - OpenRouter",
        public_name="OpenAI ChatGPT",
        provider="openrouter",
        model=secret_value("OPENROUTER_OPENAI_MODEL", "~openai/gpt-latest"),
        api_secret_name="OPENROUTER_API_KEY",
    )


def build_manual_models() -> Dict[str, ModelConfig]:
    """
    Sol menüdeki manuel model seçenekleri.
    Model slug'ları secrets üzerinden değiştirilebilir.
    """
    chatgpt_config = build_chatgpt_config()

    models = {
        "Google Gemini - Google AI Studio": ModelConfig(
            menu_label="Google Gemini - Google AI Studio",
            public_name="Google Gemini",
            provider="gemini",
            model=secret_value("GEMINI_MODEL", "gemini-3.5-flash"),
            api_secret_name="GEMINI_API_KEY",
        ),
        "Meta Llama 3.1 - Groq": ModelConfig(
            menu_label="Meta Llama 3.1 - Groq",
            public_name="Meta Llama 3.1",
            provider="groq",
            model=secret_value("LLAMA_MODEL", "llama-3.1-8b-instant"),
            api_secret_name="GROQ_API_KEY",
        ),
        "DeepSeek V3 / R1 - OpenRouter": ModelConfig(
            menu_label="DeepSeek V3 / R1 - OpenRouter",
            public_name="DeepSeek",
            provider="openrouter",
            model=secret_value("DEEPSEEK_MODEL", "deepseek/deepseek-r1:free"),
            api_secret_name="OPENROUTER_API_KEY",
        ),
        "OpenAI ChatGPT - OpenAI API veya OpenRouter": chatgpt_config,
    }

    return models


def build_route_targets() -> Dict[str, ModelConfig]:
    """
    Smart Router kararına göre çalıştırılacak nihai modeller.
    """
    return {
        "DEEPSEEK": ModelConfig(
            menu_label="DeepSeek",
            public_name="DeepSeek",
            provider="openrouter",
            model=secret_value("DEEPSEEK_MODEL", "deepseek/deepseek-r1:free"),
            api_secret_name="OPENROUTER_API_KEY",
        ),
        "GEMINI": ModelConfig(
            menu_label="Google Gemini",
            public_name="Google Gemini",
            provider="gemini",
            model=secret_value("GEMINI_MODEL", "gemini-3.5-flash"),
            api_secret_name="GEMINI_API_KEY",
        ),
        "LLAMA": ModelConfig(
            menu_label="Meta Llama",
            public_name="Meta Llama 3.1",
            provider="groq",
            model=secret_value("LLAMA_MODEL", "llama-3.1-8b-instant"),
            api_secret_name="GROQ_API_KEY",
        ),
    }


# ------------------------------------------------------------
# Promptlar
# ------------------------------------------------------------
DEFAULT_ASSISTANT_SYSTEM_PROMPT = """
Sen Türkçe yanıt veren, net, güvenilir ve yapılandırılmış bir yapay zeka asistanısın.
Kullanıcının sorusunu doğrudan yanıtla.
Gerektiğinde başlıklar, maddeler ve kısa örnekler kullan.
Bilmiyorsan veya emin değilsen bunu açıkça belirt.
""".strip()


ROUTER_SYSTEM_PROMPT = """
Kullanıcının yazdığı soruyu analiz et.

Eğer soru kodlama, yazılım, hata ayıklama, algoritma, API, veri tabanı,
framework, terminal, Git, Docker, DevOps veya teknik implementasyon ile ilgiliyse
sadece 'DEEPSEEK' cevabını ver.

Eğer soru güncel bilgi, internet araştırması, haber, hava durumu, finans,
fiyat, döviz, spor skoru, tarihsel olarak değişebilecek bilgi veya kaynaklı
araştırma gerektiriyorsa sadece 'GEMINI' cevabını ver.

Eğer soru yaratıcı yazarlık, felsefe, fikir üretme, genel sohbet, hikaye,
şiir, metin düzenleme veya gündelik tavsiye ile ilgiliyse sadece 'LLAMA'
cevabını ver.

Başka hiçbir kelime yazma.
Sadece şu üç etiketten birini döndür: DEEPSEEK, GEMINI, LLAMA.
""".strip()


# ------------------------------------------------------------
# OpenAI uyumlu istemciler
# ------------------------------------------------------------
def get_openai_compatible_client(provider: str) -> Any:
    if OpenAI is None:
        raise MissingDependencyError(
            "`openai` paketi yüklü değil. requirements.txt içine `openai` ekleyin."
        )

    if provider == "groq":
        return OpenAI(
            api_key=require_secret("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    if provider == "openrouter":
        headers = {}

        site_url = secret_value("OPENROUTER_SITE_URL", "https://streamlit.app")
        app_title = secret_value(
            "OPENROUTER_APP_TITLE",
            "Coklu Yapay Zeka Yonetim Paneli",
        )

        if site_url:
            headers["HTTP-Referer"] = site_url

        if app_title:
            headers["X-OpenRouter-Title"] = app_title

        return OpenAI(
            api_key=require_secret("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
        )

    if provider == "openai":
        return OpenAI(api_key=require_secret("OPENAI_API_KEY"))

    raise ValueError(f"Bilinmeyen sağlayıcı: {provider}")


def normalize_openai_content(content: Any) -> str:
    """
    OpenAI/OpenRouter/Groq chat completion message.content değerini metne çevirir.
    Bazı sağlayıcılarda content string, bazılarında parça listesi olabilir.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip() if content else ""


def call_openai_compatible_chat(
    provider: str,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1600,
) -> str:
    """
    Groq ve OpenRouter için OpenAI-compatible Chat Completions çağrısı.
    """
    client = get_openai_compatible_client(provider)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=max(0.01, float(temperature)),
        max_tokens=int(max_tokens),
    )

    content = completion.choices[0].message.content
    text = normalize_openai_content(content)

    if not text:
        return "_Model boş yanıt döndürdü._"

    return text


def call_openai_responses_api(
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1600,
) -> str:
    """
    Doğrudan OpenAI API için Responses API çağrısı.
    OPENAI_API_KEY varsa ChatGPT seçeneği burayı kullanır.
    """
    client = get_openai_compatible_client("openai")

    kwargs = {
        "model": model,
        "input": user_prompt,
        "max_output_tokens": int(max_tokens),
    }

    if system_prompt:
        kwargs["instructions"] = system_prompt

    # Bazı reasoning modelleri temperature parametresine kısıt koyabilir.
    # Bu yüzden hata halinde temperature olmadan bir kez daha denenir.
    try:
        kwargs["temperature"] = float(temperature)
        response = client.responses.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "temperature" in msg or "unsupported" in msg:
            kwargs.pop("temperature", None)
            response = client.responses.create(**kwargs)
        else:
            raise

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    # Eski/yeni SDK farklarına karşı yedek parse.
    parts: List[str] = []
    try:
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                text = getattr(content_item, "text", None)
                if text:
                    parts.append(text)
    except Exception:
        pass

    text = "\n".join(parts).strip()
    return text if text else "_OpenAI boş yanıt döndürdü._"


# ------------------------------------------------------------
# Gemini çağrısı
# ------------------------------------------------------------
def extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    parts: List[str] = []
    try:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(str(part_text))
    except Exception:
        pass

    return "\n".join(parts).strip()


def append_gemini_grounding_sources(text: str, response: Any) -> str:
    """
    Google Search grounding açıkken Gemini kaynak döndürürse yanıtın sonuna ekler.
    Kaynak bulunamazsa yanıtı değiştirmez.
    """
    sources: List[Tuple[str, str]] = []

    try:
        candidates = getattr(response, "candidates", []) or []
        if not candidates:
            return text

        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", []) or []

        seen_uris = set()

        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue

            uri = getattr(web, "uri", None)
            title = getattr(web, "title", None) or "Kaynak"

            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                sources.append((title.strip(), uri.strip()))

            if len(sources) >= 5:
                break

    except Exception:
        return text

    if not sources:
        return text

    source_lines = ["\n\n---\n**Kaynaklar:**"]
    for title, uri in sources:
        source_lines.append(f"- [{title}]({uri})")

    return text + "\n".join(source_lines)


def call_gemini(
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1600,
    use_google_search: bool = False,
) -> str:
    """
    Google Gemini çağrısı.
    Güncel bilgi gerektiren Smart Router kararlarında Google Search grounding açılabilir.
    """
    if genai is None or genai_types is None:
        raise MissingDependencyError(
            "`google-genai` paketi yüklü değil. requirements.txt içine `google-genai` ekleyin."
        )

    client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))

    config_kwargs: Dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens),
    }

    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    if use_google_search:
        # Gemini >= 2.0 modellerinde GoogleSearch tool desteklenir.
        # SDK eskiyse veya tool yoksa normal Gemini çağrısına düşer.
        try:
            config_kwargs["tools"] = [
                genai_types.Tool(google_search=genai_types.GoogleSearch())
            ]
        except Exception:
            pass

    config = genai_types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )

    text = extract_gemini_text(response)
    if not text:
        return "_Gemini boş yanıt döndürdü._"

    if use_google_search:
        text = append_gemini_grounding_sources(text, response)

    return text


# ------------------------------------------------------------
# Ortak model dispatch fonksiyonu
# ------------------------------------------------------------
def generate_with_model(
    config: ModelConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    use_google_search: bool = False,
) -> str:
    if config.provider == "gemini":
        return call_gemini(
            model=config.model,
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_google_search=use_google_search,
        )

    if config.provider in {"groq", "openrouter"}:
        return call_openai_compatible_chat(
            provider=config.provider,
            model=config.model,
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if config.provider == "openai":
        return call_openai_responses_api(
            model=config.model,
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Desteklenmeyen provider: {config.provider}")


# ------------------------------------------------------------
# Smart Router
# ------------------------------------------------------------
def fallback_route(question: str) -> str:
    """
    Router modeli çalışmazsa basit yerel anahtar kelime yönlendirmesi.
    Bu model çağrısı değildir; sadece emniyet yedeğidir.
    """
    q = question.lower()

    code_keywords = [
        "python", "javascript", "typescript", "java", "c#", "c++", "go ",
        "rust", "php", "sql", "html", "css", "react", "vue", "angular",
        "django", "flask", "fastapi", "streamlit", "kod", "yazılım",
        "program", "hata", "bug", "debug", "algoritma", "api", "sdk",
        "veritabanı", "database", "docker", "kubernetes", "git", "regex",
        "terminal", "deployment", "deploy", "class", "fonksiyon",
    ]

    current_info_keywords = [
        "güncel", "bugün", "yarın", "dün", "son dakika", "haber",
        "internet", "araştır", "kaynak", "hava durumu", "weather",
        "fiyat", "price", "kur", "dolar", "euro", "borsa", "kripto",
        "bitcoin", "ethereum", "maç", "skor", "seçim", "başkan",
        "ceo", "latest", "current", "2026",
    ]

    creative_keywords = [
        "hikaye", "şiir", "roman", "senaryo", "yaratıcı", "felsefe",
        "sohbet", "mizah", "deneme", "metin yaz", "reklam metni",
        "slogan", "kurgu",
    ]

    if any(keyword in q for keyword in code_keywords):
        return "DEEPSEEK"

    if any(keyword in q for keyword in current_info_keywords):
        return "GEMINI"

    if any(keyword in q for keyword in creative_keywords):
        return "LLAMA"

    return "LLAMA"


def clean_router_decision(raw: str) -> str:
    raw_upper = (raw or "").upper()

    if "DEEPSEEK" in raw_upper:
        return "DEEPSEEK"

    if "GEMINI" in raw_upper:
        return "GEMINI"

    if "LLAMA" in raw_upper or "LAMA" in raw_upper:
        return "LLAMA"

    return ""


def choose_router_provider() -> str:
    """
    Varsayılan router Groq Llama 3.1'dir.
    Groq key yoksa ve Gemini key varsa Gemini router olarak kullanılır.
    """
    configured = secret_value("ROUTER_PROVIDER", "groq").lower()

    if configured == "groq" and has_secret("GROQ_API_KEY"):
        return "groq"

    if configured == "gemini" and has_secret("GEMINI_API_KEY"):
        return "gemini"

    if has_secret("GROQ_API_KEY"):
        return "groq"

    if has_secret("GEMINI_API_KEY"):
        return "gemini"

    raise MissingSecretError(
        "Router için `GROQ_API_KEY` veya `GEMINI_API_KEY` bulunamadı."
    )


def route_question(question: str) -> Tuple[str, str, Optional[str]]:
    """
    Dönüş:
    - karar: DEEPSEEK / GEMINI / LLAMA
    - ham_router_yanıtı: modelin döndürdüğü ham çıktı
    - uyarı: varsa kullanıcıya gösterilecek açıklama
    """
    try:
        router_provider = choose_router_provider()

        if router_provider == "groq":
            router_model = secret_value(
                "ROUTER_GROQ_MODEL",
                secret_value("LLAMA_MODEL", "llama-3.1-8b-instant"),
            )

            raw = call_openai_compatible_chat(
                provider="groq",
                model=router_model,
                user_prompt=question,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.01,
                max_tokens=8,
            )

        else:
            router_model = secret_value(
                "ROUTER_GEMINI_MODEL",
                secret_value("GEMINI_MODEL", "gemini-3.5-flash"),
            )

            raw = call_gemini(
                model=router_model,
                user_prompt=question,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=8,
                use_google_search=False,
            )

        decision = clean_router_decision(raw)

        if decision:
            return decision, raw, None

        fallback = fallback_route(question)
        return (
            fallback,
            raw,
            "Router modeli beklenen tek kelimelik etiketi döndürmedi. Yerel yedek yönlendirme kullanıldı.",
        )

    except Exception as exc:
        fallback = fallback_route(question)
        return (
            fallback,
            "HEURISTIC_FALLBACK",
            f"Router modeli çalışmadı. Yerel yedek yönlendirme kullanıldı. Detay: {friendly_error_message(exc)}",
        )


# ------------------------------------------------------------
# Hata mesajları
# ------------------------------------------------------------
def friendly_error_message(exc: Exception) -> str:
    raw = str(exc)
    raw_lower = raw.lower()
    exc_name = exc.__class__.__name__.lower()

    if isinstance(exc, MissingSecretError):
        return str(exc)

    if isinstance(exc, MissingDependencyError):
        return str(exc)

    if "rate" in raw_lower or "quota" in raw_lower or "429" in raw_lower:
        return (
            "Seçilen modelin kota veya hız limiti dolmuş görünüyor. "
            "Biraz sonra tekrar deneyin veya farklı bir model seçin."
        )

    if "401" in raw_lower or "unauthorized" in raw_lower or "authentication" in raw_lower:
        return (
            "API anahtarı geçersiz veya yetkisiz görünüyor. "
            "Streamlit Secrets içindeki ilgili API key değerini kontrol edin."
        )

    if "403" in raw_lower or "permission" in raw_lower or "forbidden" in raw_lower:
        return (
            "Bu API anahtarının seçilen modele erişim izni yok. "
            "Model adını, sağlayıcı izinlerini ve hesap planını kontrol edin."
        )

    if "404" in raw_lower or "not found" in raw_lower or "model" in raw_lower:
        return (
            "Model adı veya API endpoint bulunamadı. "
            "Secrets içindeki model slug değerlerini kontrol edin."
        )

    if "timeout" in raw_lower or "api connection" in raw_lower or "connection" in exc_name:
        return (
            "API bağlantısı zaman aşımına uğradı veya sağlayıcıya ulaşılamadı. "
            "Ağ bağlantısını ve sağlayıcı durumunu kontrol edin."
        )

    return "Beklenmeyen bir hata oluştu. Teknik detayı aşağıdaki bölümden inceleyebilirsiniz."


def show_exception(exc: Exception) -> None:
    message = friendly_error_message(exc)

    if "kota" in message.lower() or "limit" in message.lower():
        st.warning(message)
    else:
        st.error(message)

    with st.expander("Teknik hata detayı"):
        st.code(redact_secret_values(str(exc))[:2500], language="text")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.header("Menü")

        mode = st.radio(
            "Çalışma modu",
            options=[
                "Manuel Model Seçimi",
                "Akıllı Yönlendirici (Smart Router)",
            ],
            index=0,
        )

        st.divider()

        st.subheader("API Key Durumu")
        st.caption("Sadece var/yok bilgisi gösterilir. Anahtarlar ekrana basılmaz.")

        key_rows = [
            ("Gemini", "GEMINI_API_KEY"),
            ("Groq", "GROQ_API_KEY"),
            ("OpenRouter", "OPENROUTER_API_KEY"),
            ("OpenAI", "OPENAI_API_KEY"),
        ]

        for label, key in key_rows:
            icon = "✅" if has_secret(key) else "⚠️"
            st.caption(f"{icon} {label}: `{key}`")

        with st.expander("Secrets örneği"):
            st.code(
                """
GEMINI_API_KEY = "..."
GROQ_API_KEY = "..."
OPENROUTER_API_KEY = "..."
OPENAI_API_KEY = "..."

GEMINI_MODEL = "gemini-3.5-flash"
LLAMA_MODEL = "llama-3.1-8b-instant"

# DeepSeek R1:
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"

# DeepSeek V3 kullanmak isterseniz:
# DEEPSEEK_MODEL = "deepseek/deepseek-chat:free"

OPENAI_MODEL = "gpt-5.4-mini"
OPENROUTER_OPENAI_MODEL = "~openai/gpt-latest"

ROUTER_PROVIDER = "groq"
ENABLE_GEMINI_SEARCH = true
                """.strip(),
                language="toml",
            )

        st.divider()
        st.caption("Mobil kullanım için butonlar ve metin alanları tam genişlikte ayarlandı.")

    return mode


# ------------------------------------------------------------
# Ana ekran parçaları
# ------------------------------------------------------------
def render_header() -> None:
    st.title("🤖 Çoklu Yapay Zeka Yönetim Paneli")
    st.caption(
        "Gemini, Llama, DeepSeek ve ChatGPT modellerini tek arayüzden kullanın. "
        "API anahtarları sadece Streamlit Secrets üzerinden okunur."
    )


def render_generation_controls() -> Tuple[float, int, str]:
    with st.expander("Gelişmiş ayarlar"):
        temperature = st.slider(
            "Yaratıcılık seviyesi",
            min_value=0.0,
            max_value=1.5,
            value=0.4,
            step=0.1,
            help="Düşük değerler daha tutarlı, yüksek değerler daha yaratıcı cevaplar üretir.",
        )

        max_tokens = st.slider(
            "Maksimum çıktı token sayısı",
            min_value=256,
            max_value=4096,
            value=1600,
            step=128,
        )

        system_prompt = st.text_area(
            "Opsiyonel sistem promptu",
            value=DEFAULT_ASSISTANT_SYSTEM_PROMPT,
            height=150,
        )

    return temperature, max_tokens, system_prompt.strip()


def render_answer(answer: str, config: ModelConfig) -> None:
    st.markdown("### Yanıt")
    st.caption(f"Model: `{config.public_name}` | Provider: `{config.provider}` | Slug: `{config.model}`")
    st.markdown(answer)


def render_manual_mode() -> None:
    st.subheader("1. Manuel Model Seçimi")

    models = build_manual_models()

    selected_label = st.selectbox(
        "Model seçin",
        options=list(models.keys()),
        index=0,
    )

    selected_config = models[selected_label]

    prompt = st.text_area(
        "Sorunuzu yazın",
        height=240,
        placeholder="Örn: Streamlit ile kullanıcı giriş sistemi nasıl kurulur?",
    )

    temperature, max_tokens, system_prompt = render_generation_controls()

    use_google_search = False
    if selected_config.provider == "gemini":
        use_google_search = st.checkbox(
            "Gemini için Google Search grounding kullan",
            value=bool_secret("ENABLE_GEMINI_SEARCH", True),
            help="Güncel bilgi, haber, fiyat veya hava durumu gibi sorularda faydalıdır.",
        )

    submitted = st.button("Soruyu Gönder", type="primary", use_container_width=True)

    if not submitted:
        return

    if not prompt.strip():
        st.warning("Lütfen bir soru yazın.")
        return

    try:
        with st.spinner(f"{selected_config.public_name} yanıt üretiyor..."):
            answer = generate_with_model(
                config=selected_config,
                prompt=prompt.strip(),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_google_search=use_google_search,
            )

        render_answer(answer, selected_config)

    except Exception as exc:
        show_exception(exc)


def render_smart_router_mode() -> None:
    st.subheader("2. Akıllı Yönlendirici (Smart Router)")

    prompt = st.text_area(
        "Sorunuzu yazın",
        height=240,
        placeholder=(
            "Örn: Bu Python hatasını düzeltir misin? / "
            "Bugünkü hava durumu nedir? / "
            "Bana kısa bir bilim kurgu hikayesi yaz."
        ),
    )

    temperature, max_tokens, system_prompt = render_generation_controls()

    submitted = st.button(
        "Akıllı Yönlendirici ile Gönder",
        type="primary",
        use_container_width=True,
    )

    if not submitted:
        return

    if not prompt.strip():
        st.warning("Lütfen bir soru yazın.")
        return

    route_targets = build_route_targets()

    with st.spinner("Router modeli soru tipini analiz ediyor..."):
        decision, raw_router_output, router_warning = route_question(prompt.strip())

    target_config = route_targets[decision]

    model_name_for_user = target_config.public_name
    st.info(f"Sistem Kararı: Bu soruyu en iyi **{model_name_for_user}** yanıtlar.")

    with st.expander("Router detayı"):
        st.write(f"Router kararı: `{decision}`")
        st.write(f"Ham router çıktısı: `{raw_router_output}`")

    if router_warning:
        st.warning(router_warning)

    use_google_search = (
        decision == "GEMINI" and bool_secret("ENABLE_GEMINI_SEARCH", True)
    )

    try:
        with st.spinner(f"{target_config.public_name} yanıt üretiyor..."):
            answer = generate_with_model(
                config=target_config,
                prompt=prompt.strip(),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_google_search=use_google_search,
            )

        render_answer(answer, target_config)

    except Exception as exc:
        show_exception(exc)


# ------------------------------------------------------------
# Uygulama başlangıcı
# ------------------------------------------------------------
def main() -> None:
    render_header()
    mode = render_sidebar()

    if mode == "Manuel Model Seçimi":
        render_manual_mode()
    else:
        render_smart_router_mode()


if __name__ == "__main__":
    main()
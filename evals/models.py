import re
import time
from datetime import date
from os import getenv
from pathlib import Path

import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from huggingface_hub import AsyncInferenceClient, HfApi
from joblib.memory import Memory
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)
from requests import HTTPError, get

# for development purposes, all languages will be evaluated on the fast models
# and only a sample of languages will be evaluated on all models
important_models = [
    "allenai/olmo-3.1-32b-instruct", 
    "meta-llama/llama-4-maverick",  # 0.6$
    "meta-llama/llama-3.3-70b-instruct",  # 0.3$
    "meta-llama/llama-3.1-70b-instruct",  # 0.3$
    "meta-llama/llama-3-70b-instruct",  # 0.4$
    # "meta-llama/llama-2-70b-chat", # 0.9$; not properly supported by OpenRouter
    "openai/gpt-5.4", # 15$
    # "openai/gpt-5.3", # 15$
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",  # 8$
    "openai/gpt-4o",  # 10$
    "openai/gpt-3.5-turbo", # $1.50
    "openai/gpt-oss-120b",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.1",  # 15$
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",  # 15$
    "anthropic/claude-3.5-sonnet",
    "mistralai/mistral-small-3.2-24b-instruct",  # 0.3$
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-saba",  # 0.6$
    "mistralai/mistral-nemo",  # 0.08$
    "google/gemini-3.1-pro-preview", #12$
    "google/gemini-3-pro-preview", # 12$
    "google/gemini-2.5-pro", # $10
    "google/gemini-2.5-flash",  # 0.6$
    "google/gemini-2.5-flash-lite",  # 0.3$
    "google/gemma-3-27b-it",  # 0.2$
    # "x-ai/grok-4", # $15
    # "minimax/minimax-m2.5",  # 1,1$; ~53% ok, content=None often
    # "moonshotai/kimi-k2.5",  # privacy filter or content=None 
    "cohere/command-a",
    # "qwen/qwen3-32b",
    # "qwen/qwen3-235b-a22b",
    "qwen/qwen3-30b-a3b",  # 0.29$
    "deepseek/deepseek-v3.2-exp",
    "microsoft/phi-4",  # 0.07$
    "amazon/nova-premier-v1", # 12.5$
    "amazon/nova-pro-v1",  # 0.09$
    "moonshotai/kimi-k2",  # 0.6$
    "baidu/ernie-4.5-300b-a47b",
    # Added 2026-05-19 — new-generation flagships (one per family; auto-discovery handles the rest)
    "openai/gpt-5.5",  # $30/M output; gpt-5.5-pro is $180/M, beyond cap
    "anthropic/claude-opus-4.7",
    "deepseek/deepseek-v4-pro",
    "x-ai/grok-4.20",
    "mistralai/mistral-medium-3.5",
    # "moonshotai/kimi-k2.6",  # removed 2026-06: ~58% errors + heavily
    # rate-limited (one model ran >50min); kimi-k2 represents Moonshot instead.
    "google/gemini-3.1-flash-lite",
]

blocklist = [
    "google/gemini-2.5-pro-preview",
    # "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash-lite-preview",
    "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-pro-preview-06-05",
    "google/gemini-2.5-pro-preview-05-06",
    "perplexity/sonar-deep-research",
    "perplexity/sonar-reasoning",
    "perplexity/sonar-reasoning-pro",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "alpindale/goliath-120b",
    "z-ai/glm-4.6",  # ~33% ok, content=None often
    "qwen/qwen3-235b-a22b",  # ~60% ok, content=None often
]

# Hard upper bound on per-token output cost. Models above this are dropped
# (validated in get_or_metadata + discover_new_models + load_models filter).
# Raised 2026-05-19 from $25 -> $30 to accommodate GPT-5.5 ($30/M output).
COST_CAP_PER_1M = 30.0

transcription_models = [
    "elevenlabs/scribe_v1",
    "openai/whisper-large-v3",
    # "openai/whisper-small",
    # "facebook/seamless-m4t-v2-large",
]

cache = Memory(location=".cache", verbose=0).cache


@cache
def load_or_metadata(date: date):
    """Fetch the OpenRouter model catalog, normalized to the shape the rest of
    this module expects (slug / permaslug / created_at / endpoint.pricing / ...).

    OpenRouter removed the undocumented /api/frontend/models endpoint (now 404),
    so we build from the official /api/v1/models. That catalog no longer carries
    per-provider data policy — privacy is instead enforced at REQUEST time via
    provider.data_collection="deny" in complete() (a strictly stronger guarantee:
    OpenRouter refuses to route a prompt to any training/retaining provider on
    every call, rather than us trusting a scraped field). Each normalized entry
    therefore reports dataPolicy.training=False so the existing downstream
    filters pass; the real enforcement lives in complete()."""
    headers = {"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"}
    last_error = None
    for attempt in range(4):
        try:
            resp = get(
                "https://openrouter.ai/api/v1/models", headers=headers, timeout=30
            )
            data = resp.json()["data"]
            break
        except Exception as e:  # transient network / non-JSON / outage
            last_error = e
            if attempt < 3:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"OpenRouter /api/v1/models fetch failed after retries: {last_error}"
        )

    normalized = []
    for m in data:
        slug = m.get("id")
        if not slug:
            continue
        name = m.get("name") or slug
        short_name = name.split(": ", 1)[1] if ": " in name else name
        try:
            completion = m["pricing"]["completion"]
        except (KeyError, TypeError):
            completion = None
        try:
            is_free = float(completion) == 0
        except (TypeError, ValueError):
            is_free = False
        try:
            created_at = pd.to_datetime(m.get("created"), unit="s", utc=True).isoformat()
        except (TypeError, ValueError):
            created_at = None
        normalized.append(
            {
                "slug": slug,
                "permaslug": m.get("canonical_slug") or slug,
                "short_name": short_name,
                "name": name,
                "created_at": created_at,
                # HuggingFace repo id (for open-weight size/license lookup);
                # the old endpoint called this hf_slug.
                "hf_slug": m.get("hugging_face_id") or None,
                "endpoint": {
                    "is_free": is_free,
                    "pricing": {"completion": completion},
                    # Privacy is enforced per-request in complete(); report
                    # compatible so the catalog-level filters pass.
                    "provider_info": {"dataPolicy": {"training": False}},
                },
            }
        )
    return normalized


def get_or_metadata(permaslug):
    models = load_or_metadata(date.today())
    slugs = [
        m
        for m in models
        if (m["permaslug"] == permaslug or m["slug"] == permaslug)
        and m["endpoint"]
        and not m["endpoint"]["is_free"]
        # Privacy is now enforced per-request in complete() via
        # provider.data_collection="deny" (the /api/v1/models catalog no longer
        # exposes per-provider data policy). This dataPolicy check is kept for
        # shape compatibility — load_or_metadata reports training=False — and
        # always passes; the real guarantee lives in complete().
        and m["endpoint"]["provider_info"]["dataPolicy"]["training"] is False
    ]
    if len(slugs) == 0:
        print(f"no appropriate model (not free) found for {permaslug}")
    return slugs[0] if len(slugs) >= 1 else None


# Strip numeric version tokens AND date-snapshot suffixes from a slug to
# derive a model "family" key: vendor + base product line, with version
# numbers, date snapshots, parameter-size tokens AND size-tier / variant
# suffixes (-mini, -flash, -pro, -8b, ...) all stripped. Everything that is
# "the same model at a different size or minor revision" collapses to one key,
# so auto-discovery keeps a SINGLE flagship per product line (highest cost
# wins) instead of a dozen near-duplicate variants. This is deliberately
# aggressive — the goal is to avoid flooding the cohort with size sweeps; a
# specific variant we actually want is added by hand to important_models.
#
# Examples:
#   openai/gpt-5.5-pro                -> openai/gpt
#   openai/gpt-5.4-mini               -> openai/gpt
#   anthropic/claude-opus-4.7         -> anthropic/claude
#   deepseek/deepseek-v4-pro          -> deepseek/deepseek
#   mistralai/ministral-8b-2512       -> mistralai/ministral
#   qwen/qwen3.6-flash                -> qwen/qwen
#   nvidia/nemotron-3-super-120b-a12b -> nvidia/nemotron
#   bytedance-seed/seed-1.6-20250625  -> bytedance-seed/seed
_DATE_SUFFIX_RE = re.compile(
    r"-(20\d{6}|20\d{2}-\d{2}-\d{2}|\d{2}-\d{2}|\d{4})$"
)
_VERSION_SUFFIX_RE = re.compile(r"[-]?v?\d+(\.\d+)*(-exp|-instruct)?(?=($|-))")
# Parameter-size tokens: -8b, -70b, -8x7b, -120b-a12b, -235b-a22b, -a3b ...
_PARAM_SIZE_RE = re.compile(r"-a?\d+(?:\.\d+)?x?\d*b(?:-a\d+b)?\b")
# Size-tier / variant words that denote "same line, different size or framing".
_SIZE_TIER_TOKENS = frozenset({
    "mini", "nano", "flash", "lite", "air", "turbo", "plus", "max", "pro",
    "micro", "small", "medium", "large", "xl", "xxl", "edge", "ultra", "super",
    "base", "fast", "chat", "instruct", "it", "hf", "opus", "sonnet", "haiku",
})


def _family_key(slug: str) -> str:
    vendor, _, name = slug.partition("/")
    # Strip trailing date snapshots first (so the version regex matches cleanly).
    while True:
        new = _DATE_SUFFIX_RE.sub("", name)
        if new == name:
            break
        name = new
    name = _VERSION_SUFFIX_RE.sub("", name)
    name = _PARAM_SIZE_RE.sub("", name)
    parts = [p for p in name.split("-") if p]
    while len(parts) > 1 and parts[-1] in _SIZE_TIER_TOKENS:
        parts.pop()
    return f"{vendor}/{'-'.join(parts)}" if parts else f"{vendor}/{name}"


# Providers we trust to ship general-purpose text LLMs. Adding a new vendor
# here is the explicit human gate for auto-discovery.
_DISCOVERY_PROVIDER_ALLOWLIST = frozenset({
    "openai", "anthropic", "google", "meta-llama", "mistralai", "deepseek",
    "x-ai", "qwen", "alibaba", "cohere", "amazon", "moonshotai", "baidu",
    "allenai", "microsoft", "liquid", "ibm-granite", "nvidia", "rekaai",
    "stepfun", "tencent", "z-ai", "bytedance-seed", "ai21", "nousresearch",
    "perplexity", "arcee-ai", "deepcogito", "prime-intellect", "writer",
    "upstage",
    # NOT "openrouter": its namespace holds routing meta-models and cloaked /
    # stealth test models (auto, bodybuilder, fusion, pareto-code, ...), not
    # real benchmarkable LLMs.
})

# Skip these substrings anywhere in the slug — covers transient snapshots,
# non-text modalities, and task-specialised variants.
_DISCOVERY_SKIP_TAGS = (
    "-preview", "-beta", "-experimental", ":free", "-latest",
    "-vision", "-vl", "-image", "-audio", "-tts", "-stt", "-embed",
    "-asr", "-transcribe", "-search", "rerank", "-ocr", "-edit",
    "-voice", "voice", "-build",  # voice/agent-build endpoints, not general text LLMs
    "coder", "codex", "devstral", "codestral",
    "-thinking", "-reasoning", "-think", "-deep-research", "deepresearch",
    "-multi-agent", "safeguard",
)

# Skip these whole product families (named non-text models).
_DISCOVERY_SKIP_PRODUCTS = (
    "whisper", "voxtral", "chirp", "kokoro", "orpheus", "zonos", "csm-",
    "parakeet", "canary",  # NVIDIA ASR speech models
    "sora", "veo-", "wan-", "seedance", "seedream", "flux.", "imagine",
    "kling", "hailuo", "riverflow", "recraft", "morph-",
    "bge-", "gte-", "e5-", "multilingual-e5",
)

# Vision variants often tack a bare "v" onto the version (glm-4.5v, glm-5v-turbo)
# rather than a "-vision"/"-vl" tag, so the substring filter above misses them.
_VISION_SUFFIX_RE = re.compile(r"\d+(\.\d+)?v(-|$)")


@cache
def discover_new_models(date: date) -> list[str]:
    """Surface OpenRouter models matching inclusion rules; pick the flagship per family.

    Flagship = highest-cost non-blocked variant within a family. If a model's
    flagship gets auto-blocklisted, the next-most-expensive variant takes its
    place on the next call (auto_blocklist is consulted before the dedupe step).
    """
    try:
        catalog = load_or_metadata(date)
    except Exception as e:
        print(f"[discover_new_models] OpenRouter catalog fetch failed: {e}; skipping")
        return []

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    blocked_families = {_family_key(s) for s in blocklist}
    blocked = set(blocklist) | set(load_auto_blocklist(date))

    # Newest curated release date per family. We skip a discovered model only
    # if we already hand-curate something in its family that is AT LEAST AS NEW.
    # This is what lets a genuinely newer flagship from a lab we already track
    # (e.g. a future Claude Opus 5 or GPT-6) still get surfaced — without it,
    # the coarse family key would permanently block every future model from
    # any curated lab.
    important_set = set(important_models)
    curated_dates: dict[str, pd.Timestamp] = {}
    for m in catalog:
        s = m.get("permaslug") or m.get("slug")
        if s in important_set or m.get("slug") in important_set:
            try:
                c = pd.to_datetime(m["created_at"], utc=True)
            except (TypeError, ValueError, KeyError):
                continue
            fam = _family_key(s)
            if fam not in curated_dates or c > curated_dates[fam]:
                curated_dates[fam] = c

    candidates = []
    for m in catalog:
        slug = m.get("permaslug") or m.get("slug")
        if not slug or slug in blocked:
            continue
        if slug.startswith("~"):
            continue  # OpenRouter alias slugs like "~anthropic/claude-opus-latest"
        provider = slug.split("/", 1)[0] if "/" in slug else ""
        if provider not in _DISCOVERY_PROVIDER_ALLOWLIST:
            continue
        if any(tag in slug for tag in _DISCOVERY_SKIP_TAGS):
            continue
        slug_lower = slug.lower()
        if any(prod in slug_lower for prod in _DISCOVERY_SKIP_PRODUCTS):
            continue
        if _VISION_SUFFIX_RE.search(slug_lower):
            continue  # bare-"v" vision variant (e.g. glm-4.5v, glm-5v-turbo)
        if not m.get("endpoint"):
            continue
        if m["endpoint"].get("is_free"):
            continue
        try:
            trains = m["endpoint"]["provider_info"]["dataPolicy"]["training"]
        except (TypeError, KeyError):
            continue
        if trains is not False:
            continue
        try:
            cost_per_1m = float(m["endpoint"]["pricing"]["completion"]) * 1_000_000
        except (TypeError, KeyError, ValueError):
            continue
        if cost_per_1m > COST_CAP_PER_1M:
            continue
        try:
            created = pd.to_datetime(m["created_at"], utc=True)
        except (TypeError, ValueError, KeyError):
            continue
        if created < cutoff:
            continue
        family = _family_key(slug)
        if family in curated_dates and created <= curated_dates[family]:
            continue  # we already curate a model in this family at least as new
        if family in blocked_families:
            continue  # date-suffixed snapshot of a blocklisted slug
        candidates.append((slug, created, family, cost_per_1m))

    # Dedupe: pick flagship per family (highest cost wins; newer wins on ties).
    by_family: dict[str, tuple[str, tuple]] = {}
    for slug, created, family, cost in candidates:
        rank = (-cost, -created.timestamp())
        if family not in by_family or rank < by_family[family][1]:
            by_family[family] = (slug, rank)
    return sorted(s for s, _ in by_family.values())


def get_translation_models():
    return pd.DataFrame(
        [
            {
                "id": "google/translate-v2",
                "name": "Google Translate",
                "provider_name": "Google",
                "cost": 20.0,
                "train_on_prompts": False,  # they don't do it in the API
                "size": None,
                "type": "closed-source",
                "license": None,
                "tasks": ["translation_from", "translation_to"],
            }
        ]
    )


load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

openrouter_rate_limit = AsyncLimiter(max_rate=20, time_period=1)
elevenlabs_rate_limit = AsyncLimiter(max_rate=2, time_period=1)
huggingface_rate_limit = AsyncLimiter(max_rate=5, time_period=1)
google_rate_limit = AsyncLimiter(max_rate=10, time_period=1)


class FatalAPIError(RuntimeError):
    """Account-level failure (auth, key-limit, payment). Abort the whole run.

    These errors apply to every subsequent call regardless of model/prompt, so
    continuing the eval just floods results-detailed with bogus errors and
    poisons the auto-blocklist. Raised by complete(); re-raised (not swallowed)
    by query() in tasks.py so it propagates out of the eval loop.
    """


# Account-level by TYPE, regardless of message: 401 (bad/revoked/mistyped key)
# and 403 (key lacks permission, or spend policy denies the request). Note
# RateLimitError (429) is deliberately absent, that one is transient and
# per-request, not account-level.
_FATAL_ERROR_TYPES = (AuthenticationError, PermissionDeniedError)

_FATAL_ERROR_MARKERS = (
    "key limit exceeded",
    "insufficient credits",
    "insufficient_quota",
    "invalid api key",
    "unauthorized",
    "payment required",
)


@cache
async def complete(**kwargs) -> str | None:
    # Privacy enforcement (paper §3.3): force OpenRouter to route only to
    # providers that do NOT train on or retain prompts, on every call. This
    # replaces the old catalog pre-filter with a stronger per-request guarantee
    # — the router cannot fall back to a training provider. Merge so a caller's
    # own provider/extra_body settings are preserved.
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body["provider"] = {**extra_body.get("provider", {}), "data_collection": "deny"}
    kwargs["extra_body"] = extra_body
    async with openrouter_rate_limit:
        try:
            response = await client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "filtered" in e.message:
                return None
            raise e
        except _FATAL_ERROR_TYPES as e:
            # Credential/permission failures are account-level by definition and
            # apply to every subsequent call. Matched BY TYPE because the string
            # markers below do not catch them: a revoked or mistyped key returns
            # 401 "User not found.", which contains none of the markers, so the
            # run would otherwise grind through the whole matrix logging every
            # combo as an error, flooding results-detailed and poisoning the
            # auto-blocklist, which is exactly what this guard exists to stop.
            raise FatalAPIError(
                f"OpenRouter account-level failure ({type(e).__name__}): {e}. "
                "Aborting run before results-detailed is polluted."
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if any(marker in msg for marker in _FATAL_ERROR_MARKERS):
                raise FatalAPIError(
                    f"OpenRouter account-level failure: {e}. "
                    "Aborting run before results-detailed is polluted."
                ) from e
            raise
    if not response.choices:
        raise Exception(response)
    return response.choices[0].message.content.strip()


# Lazy-init: building translate.Client() at import time tries to read
# GOOGLE_APPLICATION_CREDENTIALS and crashes the whole module if creds are
# missing — even for callers that never translate (backend, smoke tests).
# Defer until first use so import works without GCP creds.
_translate_client = None


def _get_translate_client():
    global _translate_client
    if _translate_client is None:
        _translate_client = translate.Client()
    return _translate_client


def get_google_supported_languages():
    return [l["language"] for l in _get_translate_client().get_languages()]


@cache
async def translate_google(text, source_language, target_language):
    async with google_rate_limit:
        response = _get_translate_client().translate(
            text, source_language=source_language, target_language=target_language
        )
    return response["translatedText"]


# @cache
# async def transcribe_elevenlabs(path, model):
#     modelname = model.split("/")[-1]
#     client = AsyncElevenLabs(api_key=getenv("ELEVENLABS_API_KEY"))
#     async with elevenlabs_rate_limit:
#         with open(path, "rb") as file:
#             response = await client.speech_to_text.convert(
#                 model_id=modelname, file=file
#             )
#     return response.text


# @cache
# async def transcribe_huggingface(path, model):
#     client = AsyncInferenceClient(api_key=getenv("HUGGINGFACE_ACCESS_TOKEN"))
#     async with huggingface_rate_limit:
#         output = await client.automatic_speech_recognition(model=model, audio=path)
#     return output.text


# async def transcribe(path, model="elevenlabs/scribe_v1"):
#     provider, modelname = model.split("/")
#     match provider:
#         case "elevenlabs":
#             return await transcribe_elevenlabs(path, modelname)
#         case "openai" | "facebook":
#             return await transcribe_huggingface(path, model)
#         case _:
#             raise ValueError(f"Model {model} not supported")


api = HfApi()


@cache
def get_hf_metadata(row):
    # get metadata from the HuggingFace API
    empty = {
        "hf_id": None,
        "creation_date": None,
        "size": None,
        "type": "closed-source",
        "license": None,
    }
    if not row:
        return empty
    id = row["hf_slug"] or row["slug"].split(":")[0]
    if not id:
        return empty
    try:
        info = api.model_info(id)
        license = ""
        if (
            info.card_data
            and hasattr(info.card_data, "license")
            and info.card_data.license
        ):
            license = (
                info.card_data.license.replace("-", " ").replace("mit", "MIT").title()
            )
        return {
            "hf_id": info.id,
            "creation_date": info.created_at,
            "size": info.safetensors.total if info.safetensors else None,
            "type": "open-source",
            "license": license,
        }
    except HTTPError:
        return empty


def get_cost(row):
    try:
        cost = float(row["endpoint"]["pricing"]["completion"])
        return round(cost * 1_000_000, 2)
    except (TypeError, KeyError):
        return None


def get_training_policy(row):
    # get openrouter info whether the provider may train on prompts
    # (this needs to be thoroughly avoided for our benchmark prompts!)
    return row["endpoint"]["provider_info"]["dataPolicy"]["training"]


# Auto-blocklist thresholds: a model is a blocklist CANDIDATE in a given run if
# it has attempted at least MIN_ATTEMPTS evaluations and FAIL_PCT_THRESHOLD% or
# more returned an error (content=None / filtered / etc.). Matches the manual
# blocklist's bar ("~33% ok, content=None often" -> 67% fail -> blocked).
AUTO_BLOCKLIST_MIN_ATTEMPTS = 100
AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD = 50.0
# ...but a single bad run does NOT exclude a model — a provider rate-limiting us
# during a long run would otherwise wrongly blocklist a healthy model (and once
# excluded it's never re-attempted, so it can't recover). A model must stay past
# the threshold for this many CONSECUTIVE runs before it is actually excluded.
# Strikes are persisted to HF (survives ephemeral CI runners) and maintained by
# update_blocklist_strikes(); a model that recovers in any run drops back to 0.
AUTO_BLOCKLIST_MIN_RUNS = 2
# Fast path: a model failing this badly is broken/unusable, not transiently
# rate-limited — exclude it after a SINGLE bad run rather than waiting out the
# grace period, so we don't keep spending on it.
AUTO_BLOCKLIST_IMMEDIATE_FAIL_PCT = 80.0
# The 2-run grace is only worth it when retrying is CHEAP. A model that is
# failing AND slow (heavily rate-limited) is expensive to retry and unlikely to
# recover, so it's excluded immediately too. "Slow" = wall-clock seconds per
# attempted sample above this, measured during the run. Healthy models run near
# the 20 req/s limit (~0.05 s/sample); a rate-limited one is many times slower
# (e.g. kimi-k2.6 ran ~0.7 s/sample). So the grace effectively only protects a
# FAST, moderately-failing model — exactly "retry only if cheap to retry".
AUTO_BLOCKLIST_SLOW_SEC_PER_SAMPLE = 0.3


def compute_model_health(detailed=None) -> pd.DataFrame:
    """Per-model success/failure stats. Uses the passed results-detailed frame if
    given (avoids re-downloading at end of a run), else loads it. Empty DF on miss."""
    from datasets_.util import load

    if detailed is None:
        detailed = load("results-detailed")
    if detailed.empty or "status" not in detailed.columns:
        return pd.DataFrame(
            columns=["model", "total", "failed", "failed_pct", "score_nonfailed"]
        )
    is_error = (detailed["status"] != "ok").rename("is_error")
    grouped = pd.DataFrame(
        {
            "total": detailed.groupby("model").size(),
            "failed": is_error.groupby(detailed["model"]).sum(),
            "score_nonfailed": detailed[~is_error].groupby("model")["score"].mean(),
        }
    ).reset_index()
    grouped["failed_pct"] = grouped["failed"] / grouped["total"] * 100
    return grouped.sort_values("failed_pct", ascending=False)


@cache
def load_auto_blocklist(date: date) -> list[str]:
    """Models excluded from the cohort: those that have stayed past the failure
    threshold for >= AUTO_BLOCKLIST_MIN_RUNS consecutive runs. This is a cheap
    READ of the persisted strike table (maintained by update_blocklist_strikes
    at the end of each eval run) — safe to call at backend startup, and a single
    bad run never excludes a model here. Empty before any strikes exist."""
    try:
        from datasets_.util import load

        strikes = load("model-health-strikes")
    except Exception as e:
        print(f"[auto_blocklist] failed to load strikes: {e}; using empty list")
        return []
    if strikes.empty or "strikes" not in strikes.columns:
        return []
    return sorted(
        strikes[strikes["strikes"] >= AUTO_BLOCKLIST_MIN_RUNS]["model"].tolist()
    )


def update_blocklist_strikes(detailed=None, slow_models=None,
                             not_attempted=None, dry_run=False) -> pd.DataFrame:
    """Recompute consecutive-bad-run strikes and persist them to HF. Called once
    per eval run (from main.py) after results are merged. A model currently past
    the failure threshold gets +1 strike; a model that has recovered (or never
    failed) drops to 0 and out of the table. Models reaching AUTO_BLOCKLIST_MIN_RUNS
    strikes are excluded by load_auto_blocklist on the NEXT run.

    The 2-run grace (one free re-attempt) only applies when retrying is cheap.
    A failing model that is ALSO egregiously bad (>=80%) or SLOW (in
    `slow_models`, measured this run) is excluded after a single run — we don't
    spend more time/money re-attempting an expensive failure.

    `not_attempted` are models this run never got to, deferred by
    MAX_NEW_MODELS_PER_RUN. Strikes are computed from the CUMULATIVE log, so
    without this they would keep accruing +1 per run on a stale record and be
    auto-blocklisted after AUTO_BLOCKLIST_MIN_RUNS runs without ever getting the
    re-attempt the grace period is supposed to buy them. Their prior count is
    held steady instead."""
    from datasets_.util import load, save, save_local_only

    slow_models = slow_models or set()
    not_attempted = not_attempted or set()
    health = compute_model_health(detailed)
    cols = ["model", "strikes", "failed_pct"]
    if health.empty:
        return pd.DataFrame(columns=cols)
    bad = health[
        (health["total"] >= AUTO_BLOCKLIST_MIN_ATTEMPTS)
        & (health["failed_pct"] >= AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD)
    ]
    try:
        prior = load("model-health-strikes")
        prior_map = (
            dict(zip(prior["model"], prior["strikes"]))
            if not prior.empty and "strikes" in prior.columns
            else {}
        )
    except Exception:
        prior_map = {}
    fail_map = dict(zip(bad["model"], bad["failed_pct"]))

    def _strikes_for(model, fail_pct):
        # Deferred by the per-run cap, it never ran, so it can't have earned a
        # strike. Hold the prior count so the record persists without advancing.
        if model in not_attempted:
            return int(prior_map.get(model, 0))
        # No grace if the failure is egregious OR slow-and-expensive-to-retry;
        # otherwise increment so a fast, moderately-failing model gets one free
        # re-attempt before exclusion.
        incremented = int(prior_map.get(model, 0)) + 1
        no_grace = fail_pct >= AUTO_BLOCKLIST_IMMEDIATE_FAIL_PCT or model in slow_models
        if no_grace:
            return max(incremented, AUTO_BLOCKLIST_MIN_RUNS)
        return incremented

    strikes_df = pd.DataFrame(
        [
            {
                "model": m,
                "strikes": _strikes_for(m, fail_map[m]),
                "failed_pct": round(float(fail_map[m]), 1),
            }
            for m in sorted(fail_map)
        ],
        columns=cols,
    )
    if dry_run:
        save_local_only(strikes_df, "dry-run/model-health-strikes")
    else:
        save(strikes_df, "model-health-strikes")
    blocked = sorted(
        strikes_df[strikes_df["strikes"] >= AUTO_BLOCKLIST_MIN_RUNS]["model"].tolist()
    )
    print(
        f"[strikes] {len(strikes_df)} model(s) failing this run; "
        f"{len(blocked)} now at >= {AUTO_BLOCKLIST_MIN_RUNS} consecutive strikes "
        f"(excluded next run): {blocked}"
    )
    return strikes_df


@cache
def load_models(date: date) -> pd.DataFrame:
    auto_discovered = discover_new_models(date)
    auto_blocked = set(load_auto_blocklist(date))

    # Manual curation wins: important_models override the auto-blocklist
    # (the warning gives a human a nudge to investigate the quality regression).
    override = set(important_models) & auto_blocked
    if override:
        print(
            f"[load_models] important_models override auto_blocklist (kept anyway): "
            f"{sorted(override)}"
        )
    if auto_blocked - override:
        print(
            f"[load_models] auto_blocklist excluding: "
            f"{sorted(auto_blocked - override)}"
        )
    if auto_discovered:
        print(f"[load_models] auto_discovered added: {auto_discovered}")

    all_model_candidates = (
        (set(important_models) | (set(auto_discovered) - auto_blocked))
        - set(blocklist)
    )

    # Snapshot health stats for inspection (small enough to track in git).
    try:
        health = compute_model_health()
        if not health.empty:
            Path("results").mkdir(exist_ok=True)
            health.to_json(
                "results/model_health.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )
    except Exception as e:
        print(f"[load_models] failed to snapshot model_health.json: {e}")

    # Validate models exist on OpenRouter before including them
    valid_models = []

    for model_id in all_model_candidates:
        metadata = get_or_metadata(model_id)
        if metadata is not None:
            valid_models.append(model_id)

    models = pd.DataFrame(sorted(valid_models), columns=["id"])
    or_metadata = models["id"].apply(get_or_metadata)  # TODO this is double-doubled
    hf_metadata = or_metadata.apply(get_hf_metadata)
    creation_date_hf = pd.to_datetime(hf_metadata.str["creation_date"]).dt.date
    creation_date_or = pd.to_datetime(
        or_metadata.str["created_at"].str.split("T").str[0]
    ).dt.date

    models = models.assign(
        name=or_metadata.str["short_name"]
        .str.replace(" (free)", "")
        .str.replace(" (self-moderated)", "")
        .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True),
        provider_name=or_metadata.str["name"].str.split(": ").str[0],
        # openrouter_metadata=or_metadata.astype(str),
        cost=or_metadata.apply(get_cost),
        train_on_prompts=or_metadata.apply(get_training_policy),
        hf_id=hf_metadata.str["hf_id"],
        size=hf_metadata.str["size"],
        type=hf_metadata.str["type"],
        license=hf_metadata.str["license"],
        creation_date=creation_date_hf.combine_first(creation_date_or),
    )
    models.to_json(
        "models_unfiltered.json", orient="records", indent=2, force_ascii=False
    )
    # Filter out expensive models to keep costs reasonable.
    # Log any manually-curated entries that get dropped here so the user knows why.
    too_expensive = models[models["cost"] > COST_CAP_PER_1M]
    important_dropped = too_expensive[too_expensive["id"].isin(important_models)]
    for _, row in important_dropped.iterrows():
        print(
            f"[load_models] dropping {row['id']} from cohort: "
            f"cost ${row['cost']}/M > cap ${COST_CAP_PER_1M}/M"
        )
    models = models[models["cost"] <= COST_CAP_PER_1M].reset_index(drop=True)
    models["tasks"] = [
        [
            "translation_from",
            "translation_to",
            "classification",
            "mmlu",
            "arc",
            "truthfulqa",
            "mgsm",
        ]
    ] * len(models)
    models = pd.concat([models, get_translation_models()])
    return models


models = load_models(date.today())

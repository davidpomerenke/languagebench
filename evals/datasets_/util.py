import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, get_dataset_config_names, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from joblib.memory import Memory
from langcodes import standardize_tag
from requests.exceptions import ConnectionError as RequestsConnectionError

cache = Memory(location=".cache", verbose=0).cache
TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

TRANSIENT_ERRORS = (RequestsConnectionError, ConnectionError, OSError)

# Macrolanguage mappings: when standardize_tag returns a macrolanguage,
# map it to the preferred specific variant for consistency across datasets.
# This ensures results from different benchmarks use the same language code.
MACROLANGUAGE_MAPPINGS = {
    "no": "nb",  # Norwegian -> Norwegian Bokmål (most widely used variant)
    # Add more mappings here if they cause duplicate entries in the languages table:
    # "ms": "zsm",  # Malay -> Standard Malay (if both appear in population data)
    # "ar": "arb",  # Arabic -> Standard Arabic (if both appear in population data)
    # "zh": "cmn",  # Chinese -> Mandarin Chinese (if both appear in population data)
    # Check LANGUAGE_SPEAKING_POPULATION to see which macrolanguages need mapping
}


def standardize_bcp47(tag: str, macro: bool = True) -> str:
    """Standardize a BCP-47 tag with consistent macrolanguage handling."""
    
    standardized = standardize_tag(tag, macro=macro)
    return MACROLANGUAGE_MAPPINGS.get(standardized, standardized)


@cache
def _get_dataset_config_names(dataset, **kwargs):
    return get_dataset_config_names(dataset, **kwargs)


def _load_dataset_impl(dataset, subset, **kwargs):
    return load_dataset(dataset, subset, **kwargs)


@cache
def _load_dataset(dataset, subset, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            return _load_dataset_impl(dataset, subset, **kwargs)
        # ValueError covers HF's "Couldn't find cache for config X" — a flaky
        # download/build that left no cache; a retry usually succeeds. Bounded,
        # so a genuinely-missing config still raises after 4 attempts (and the
        # caller treats that one combo as an error rather than crashing).
        except (*TRANSIENT_ERRORS, ValueError) as e:
            last_error = e
            if attempt < 3:
                time.sleep(2**attempt)
    raise last_error


# Cache individual dataset items to avoid reloading entire datasets
@cache
def _get_dataset_item(dataset, subset, split, index, **kwargs):
    """Load a single item from a dataset efficiently"""
    ds = _load_dataset(dataset, subset, split=split, **kwargs)
    return ds[index] if index < len(ds) else None


def load(fname: str):
    try:
        ds = load_dataset(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
        return ds["train"].to_pandas()
    except (DatasetNotFoundError, RepositoryNotFoundError, KeyError):
        return pd.DataFrame()


def save(df: pd.DataFrame, fname: str):
    df = df.drop(columns=["__index_level_0__"], errors="ignore")
    # Write the local snapshot first so progress is on disk even if the HF push
    # later fails — cheap insurance for long, checkpoint-heavy runs.
    Path("results").mkdir(exist_ok=True)
    df.to_json(f"results/{fname}.json", orient="records", force_ascii=False, indent=2)
    ds = Dataset.from_pandas(df)
    # Retry the push with backoff: per-model checkpointing pushes often, and
    # HF intermittently 429s / drops connections. A transient failure must not
    # kill a multi-hour run; a persistent one still raises so we don't silently
    # lose durability (the next run resumes from whatever did land on HF).
    last_error = None
    for attempt in range(5):
        try:
            ds.push_to_hub(f"fair-forward/evals-for-every-language-{fname}", token=TOKEN)
            return
        except (HfHubHTTPError, *TRANSIENT_ERRORS) as e:
            last_error = e
            if attempt < 4:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40s
                print(f"[save] HF push of '{fname}' failed (attempt {attempt + 1}/5): "
                      f"{e}; retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def save_local_only(df: pd.DataFrame, fname: str):
    """Write the snapshot to results/{fname}.json without pushing to HF.

    Used during partial-scale eval runs (smoke tests, local development) so
    the public dataset isn't truncated by a filtered aggregate. The next
    full-scale run will push the canonical version.

    `fname` may contain a subdirectory (e.g. "dry-run/results"), which DRY_RUN
    uses to keep test output away from the tracked results/*.json files.
    """
    df = df.drop(columns=["__index_level_0__"], errors="ignore")
    out = Path("results") / f"{fname}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", force_ascii=False, indent=2)


def get_valid_task_languages(task_name: str) -> set:
    """Return set of bcp_47 codes that have data available for the given task."""
    from datasets_.flores import flores, splits
    from datasets_.mmlu import tags_afrimmlu, tags_global_mmlu, tags_mmlu_autotranslated
    from datasets_.arc import tags_uhura_arc_easy, tags_uhura_arc_easy_translated
    from datasets_.truthfulqa import tags_uhura_truthfulqa
    from datasets_.mgsm import tags_mgsm, tags_afrimgsm, tags_gsm8kx, tags_gsm_autotranslated
    
    if task_name in ["translation_from", "translation_to", "classification"]:
        return set(flores["bcp_47"])
    elif task_name == "mmlu":
        return set([*tags_afrimmlu.keys(), *tags_global_mmlu.keys(), *tags_mmlu_autotranslated.keys()])
    elif task_name == "arc":
        return set([*tags_uhura_arc_easy.keys(), *tags_uhura_arc_easy_translated.keys()])
    elif task_name == "truthfulqa":
        return set(tags_uhura_truthfulqa.keys())
    elif task_name == "mgsm":
        return set([*tags_mgsm.keys(), *tags_afrimgsm.keys(), *tags_gsm8kx.keys(), *tags_gsm_autotranslated.keys()])
    return set()

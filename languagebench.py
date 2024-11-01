import asyncio
import json
import os
import random
from os import getenv

import evaluate
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from joblib.memory import Memory
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# config
models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-70b-instruct",  # lots of slow repetitions for LRLs
    "mistralai/mistral-nemo",
    # "google/gemini-flash-1.5",  # very fast
    "qwen/qwen-2.5-72b-instruct",  # somewhat slow
]
fast_model = "anthropic/claude-3.5-sonnet"
original_language = "eng_Latn"
dataset = "floresp-v2.0-rc.3/dev"
random.seed(42)
target_languages = [f.split(".")[1] for f in os.listdir(dataset)]
detailed_target_languages = random.choices(target_languages, k=5)

# setup
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("sacrebleu")
rate_limit = AsyncLimiter(max_rate=15, time_period=1)


def check_rate_limit():
    print(
        requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"},
        ).json()
    )
    models = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"},
    ).json()["data"]
    model = next((m for m in models if m["id"] == "google/gemini-flash-1.5"), None)
    print(model)


@cache
async def complete(**kwargs):
    async with rate_limit:
        response = await client.chat.completions.create(**kwargs)
    if not response.choices:
        raise Exception(response)
    return response


def reorder(language_name):
    if "," in language_name and "(" not in language_name:
        return language_name.split(",")[1] + " " + language_name.split(",")[0]
    return language_name


language_names = pd.read_csv("LanguageCodes.tab", sep="\t")
language_names["Name"] = language_names["Name"].apply(reorder)
language_stats = pd.read_csv("languages.tsv", sep="\t")
script_names = pd.read_csv("ScriptCodes.csv")


@cache
async def translate(model, target_language, target_script, sentence):
    reply = await complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to the {target_language} language; use the {target_script} script; reply only with the translation:\n\n{sentence}",
            }
        ],
        temperature=0,
        max_tokens=1024,
    )
    return reply.choices[0].message.content


def get_language_stats(language_code):
    lang, script = language_code.split("_", 1)
    script = script.split("_", 1)[0]
    stats = language_stats[language_stats["iso639_3"] == lang]
    if not stats.empty:
        stats = stats.iloc[0].to_dict()
    else:
        stats = dict()
    stats["script"] = script_names[script_names["Code"] == script]["English Name"].iloc[
        0
    ]
    name_series = language_names[language_names["LangID"] == lang]["Name"]
    stats["name"] = (
        name_series.iloc[0]
        if not name_series.empty
        else stats.get("itemLabel_en") or stats.get("itemLabel", lang)
    )
    return stats


async def main():
    n = 30
    results = []
    original_sentences = open(f"{dataset}/dev.{original_language}").readlines()
    for target_language in target_languages:
        if target_language == original_language:
            continue
        target_sentences = open(f"{dataset}/dev.{target_language}").readlines()
        for model in models:
            if model != fast_model and target_language not in detailed_target_languages:
                continue
            stats = get_language_stats(target_language)
            print(f"{model} -> {stats['name']}")
            predictions = [
                translate(model, stats["name"], stats["script"], sentence)
                for sentence in original_sentences[:n]
            ]
            predictions = await tqdm_asyncio.gather(*predictions, miniters=1)
            metrics = bleu.compute(
                predictions=predictions,
                references=target_sentences[:n],
                tokenize="char",
            )

            results.append(
                {
                    "model": model,
                    "original_language": original_language,
                    "target_language": target_language,
                    "target_language_name": stats["name"],
                    "speakers": int(stats.get("maxSpeakers", 0)),
                    "bleu": metrics["score"],
                }
            )
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            pd.DataFrame(results).groupby("target_language_name").agg(
                {"bleu": "mean", "speakers": "mean"}
            ).reset_index().to_json("results_summary.json", indent=2, orient="records")


if __name__ == "__main__":
    # check_rate_limit()
    asyncio.run(main())

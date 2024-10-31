import asyncio
import json
import os
import random
from os import getenv

import evaluate
import pandas as pd
from dotenv import load_dotenv
from joblib.memory import Memory
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

# config
models = [
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "anthropic/claude-3.5-sonnet",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]
# models = ["gpt-4o-mini"]
original_language = "eng_Latn"
dataset = "floresp-v2.0-rc.3/dev"
random.seed(42)
target_languages = [f.split(".")[1] for f in os.listdir(dataset)]
target_languages = random.choices(target_languages, k=9)
# target_languages = [
#     "eng_Latn",
#     "deu_Latn",
#     "fra_Latn",
#     "spa_Latn",
#     "cmn_Hans",
#     "cmn_Hant",
# ]

# setup
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
    # api_key=getenv("OPENAI_API_KEY"),
)
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("sacrebleu")


@cache
async def complete(**kwargs):
    return await client.chat.completions.create(**kwargs)


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
                "content": f"Translate the following text to {target_language} (script: {target_script}):\n\n{sentence}",
            }
        ],
        temperature=0,
    )
    return reply.choices[0].message.content


def get_language_stats(language_code):
    lang, script = language_code.split("_")
    stats = language_stats[language_stats["iso639_3"] == lang]
    if not stats.empty:
        stats = stats.iloc[0].to_dict()
    else:
        stats = dict()
    stats["script"] = script_names[script_names["Code"] == script]["English Name"].iloc[
        0
    ]
    stats["name"] = language_names[language_names["LangID"] == lang]["Name"].iloc[0]
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
            stats = get_language_stats(target_language)
            print(f"{model} -> {stats['name']}")
            # predictions = [
            #     await translate(model, stats["name"], stats["script"], sentence)
            #     for sentence in tqdm(original_sentences[:n])
            # ]
            predictions = [
                translate(model, stats["name"], stats["script"], sentence)
                for sentence in tqdm(original_sentences[:n])
            ]
            predictions = await tqdm_asyncio.gather(*predictions)
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
            # compute mean bleu for each target language
            pd.DataFrame(results).groupby("target_language_name").agg(
                {"bleu": "mean", "speakers": "mean"}
            ).reset_index().to_json("results_summary.json", indent=2, orient="records")


if __name__ == "__main__":
    asyncio.run(main())

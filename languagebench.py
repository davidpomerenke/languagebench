import asyncio
import json
import os
from os import getenv

import evaluate
import pandas as pd
from dotenv import load_dotenv
from joblib.memory import Memory
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

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
target_languages = sorted([f.split(".")[1] for f in os.listdir(dataset)][:10])
target_languages = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "spa_Latn",
    "cmn_Hans",
    "cmn_Hant",
]

# setup
load_dotenv()
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
    # api_key=getenv("OPENAI_API_KEY"),
)
cache = Memory(location=".cache", verbose=0).cache
bleu = evaluate.load("sacrebleu")
language_stats = pd.read_csv("languages.tsv", sep="\t")


@cache
async def translate(model, target_language, sentence):
    reply = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text from {original_language} to {target_language}:\n\n{sentence}",
            }
        ],
        temperature=0,
    )
    return reply.choices[0].message.content


def get_language_stats(language_code):
    lang, script = language_code.split("_")
    stats = language_stats[language_stats["iso639_3"] == lang]
    if stats.empty:
        return dict()
    return stats.iloc[0].to_dict()


async def main():
    n = 30
    results = []
    original_sentences = open(f"{dataset}/dev.{original_language}").readlines()
    for target_language in target_languages:
        target_sentences = open(f"{dataset}/dev.{target_language}").readlines()
        for model in models:
            print(f"{model} -> {target_language}")
            predictions = await tqdm_asyncio.gather(
                *[
                    translate(model, target_language, sentence)
                    for sentence in original_sentences[:n]
                ],
            )
            metrics = bleu.compute(
                predictions=predictions, references=target_sentences[:n]
            )
            stats = get_language_stats(target_language)
            results.append(
                {
                    "model": model,
                    "original_language": original_language,
                    "target_language": target_language,
                    "target_language_name": stats.get("itemLabel_en", target_language),
                    "speakers": stats.get("maxSpeakers"),
                    "bleu": metrics["score"],
                }
            )
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())

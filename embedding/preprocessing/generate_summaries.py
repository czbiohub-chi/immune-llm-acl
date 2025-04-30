import argparse
import json
import re

import pandas as pd
from openai import OpenAI
from pqdm.threads import pqdm
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--input_col", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    return args


def worker(term, prompt_template, client, model):
    prompt = json.loads(re.sub("\{.*\}", term, prompt_template))
    completion = client.chat.completions.create(
        messages=prompt,
        model=model,
        seed=42,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    return completion.choices[0].message.content


def generate_summaries(
    *,  # enforce kwargs
    terms: list[str],
    prompt_file: str,
    api_url: str,
    api_key: str,
    model: str,
    num_workers: int = 0,
) -> list[str]:
    client = OpenAI(base_url=api_url, api_key=api_key)

    with open(prompt_file) as f:
        prompt_template = f.read()

    if num_workers == 0:
        summaries = []
        for term in tqdm(terms):
            summaries.append(worker(term, prompt_template, client, model))
    else:
        summaries = pqdm(
            terms,
            lambda term: worker(term, prompt_template, client, model),
            n_jobs=num_workers,
        )

    return summaries


def main(args):
    input_df = pd.read_csv(args.input_file)
    terms = input_df[args.input_col]

    summaries = generate_summaries(
        terms=terms,
        prompt_file=args.prompt_file,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.num_workers,
    )

    input_df[args.input_col + "_summarized"] = summaries
    input_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

import argparse
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from pqdm.threads import pqdm
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--input_col", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_col", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    args = parser.parse_args()
    return args


def worker(batch: list[str], client: OpenAI, model: str) -> list[list[float]]:
    out = client.embeddings.create(
        input=batch,
        model=model,
    )
    return [d.embedding for d in out.data]


def generate_embeddings(
    *,  # enforce kwargs
    terms: list[str],
    api_url: str,
    api_key: str,
    model: str,
    num_workers: int = 0,
    batch_size: int = 1024,
) -> list[np.ndarray]:
    client = OpenAI(base_url=api_url, api_key=api_key)
    batched_terms = [
        terms[i : i + batch_size] for i in range(0, len(terms), batch_size)
    ]
    if num_workers == 0:
        batched_embeddings: list[list[list[float]]] = []
        for batch in tqdm(batched_terms):
            batched_embeddings.append(worker(batch, client, model))
    else:
        batched_embeddings = pqdm(
            batched_terms,
            lambda batch: worker(batch, client, model),
            n_jobs=num_workers,
        )
    embeddings = [np.asarray(e) for batch in batched_embeddings for e in batch]
    return embeddings


def main(args):
    assert not os.path.exists(args.output_file)

    input_df = pd.read_csv(args.input_file)
    terms = input_df[args.input_col]

    embeddings = generate_embeddings(
        terms=terms,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    embeddings = {k: v for k, v in zip(input_df[args.output_col], embeddings)}
    np.save(args.output_file, embeddings)


if __name__ == "__main__":
    args = parse_args()
    main(args)

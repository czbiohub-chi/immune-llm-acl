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


def main(args):
    assert not os.path.exists(args.output_file)
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)

    input_df = pd.read_csv(args.input_file)
    terms = input_df[args.input_col]
    batched_terms = [
        terms[i : i + args.batch_size] for i in range(0, len(terms), args.batch_size)
    ]
    if args.num_workers == 0:
        batched_embeddings: list[list[list[float]]] = []
        for batch in tqdm(batched_terms):
            batched_embeddings.append(worker(batch, client, args.model))
    else:
        batched_embeddings = pqdm(
            batched_terms,
            lambda batch: worker(batch, client, args.model),
            n_jobs=args.num_workers,
        )

    embeddings = [np.asarray(e) for batch in batched_embeddings for e in batch]
    embeddings = {k: v for k, v in zip(input_df[args.output_col], embeddings)}
    np.save(args.output_file, embeddings)


if __name__ == "__main__":
    args = parse_args()
    main(args)

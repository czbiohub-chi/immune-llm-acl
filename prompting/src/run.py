import argparse
import os

import pandas as pd
from tqdm import trange

# fmt: off
# isort: off
from benchmark import read_screen_results
from screen import (
    DEFAULT_ANSWER_CHOICES,
    DEFAULT_ANSWER_DELIMITER,
    DEFAULT_NUM_PARALLEL_REQUESTS,
    run_screen,
)
from utils import read_pregenerated_outputs
# isort: on
# fmt: on


def run_benchmark(
    *,  # enforce kwargs
    prompt_file: str,
    benchmark_tsv: str,
    screens_dir: str,
    outputs_dir: str,
    api_url: str,
    api_key: str = "EMPTY",
    model: str = "",
    iterative_cot: bool,
    as_text: bool,
    pregenerated_outputs_dir: str | None = None,
    answer_delimiter: str = DEFAULT_ANSWER_DELIMITER,
    answer_choices: list[str] = DEFAULT_ANSWER_CHOICES,
    num_parallel_requests: int = DEFAULT_NUM_PARALLEL_REQUESTS,
):
    experiments = pd.read_csv(benchmark_tsv, sep="\t")
    with open(prompt_file) as f:
        prompt_template = f.read()

    if pregenerated_outputs_dir is not None:
        pregenerated_outputs_files = os.listdir(pregenerated_outputs_dir)
        for screen_file in experiments["screen_results"]:
            if not screen_file in pregenerated_outputs_files:
                raise ValueError(
                    f"{screen_file} not present in pregenerated_outputs_dir"
                )

    os.makedirs(outputs_dir)

    for i in trange(len(experiments), desc="Screens"):
        screen_file = experiments.loc[i, "screen_results"]
        organism = None
        if "organism" in experiments:
            organism = experiments.loc[i, "organism"]
        screen_results = read_screen_results(
            os.path.join(screens_dir, screen_file),
            organism=organism,
        )
        genes = screen_results["OFFICIAL_SYMBOL"].to_list()
        pregenerated_outputs = None
        if pregenerated_outputs_dir is not None:
            pregenerated_outputs = read_pregenerated_outputs(
                outputs_file=os.path.join(pregenerated_outputs_dir, screen_file),
                screen_results=screen_results,
            )
        _, outputs, answers = run_screen(
            genes=genes,
            crispr_strategy=experiments.loc[i, "crispr_strategy"],
            cell_type=experiments.loc[i, "cell_type"],
            target_phenotype=experiments.loc[i, "target_phenotype"],
            prompt_template=prompt_template,
            api_url=api_url,
            api_key=api_key,
            model=model,
            iterative_cot=iterative_cot,
            as_text=as_text,
            outputs=pregenerated_outputs,
            answer_delimiter=answer_delimiter,
            answer_choices=answer_choices,
            num_parallel_requests=num_parallel_requests,
        )
        out_df = pd.DataFrame({"gene": genes, "output": outputs, "answer": answers})
        out_df.to_csv(os.path.join(outputs_dir, screen_file), index=False, sep="\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        required=True,
        help="File with text or json formatted prompt template; should contain format fields for {gene}, {crispr_strategy}, {cell_type}, and {phenotype}",
    )
    parser.add_argument(
        "--benchmark_tsv",
        required=True,
        help="TSV with screen metadata",
    )
    parser.add_argument(
        "--screens_dir",
        required=True,
        help="Folder containing screens to benchmark",
    )
    parser.add_argument(
        "--outputs_dir",
        required=True,
        help="Folder to save results",
    )
    parser.add_argument(
        "--api_url",
        required=True,
        help="Base URL for API e.g. https://api.openai.com/v1",
    )
    parser.add_argument(
        "--api_key",
        default="EMPTY",
        help="API key",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model to use for inference, defaults to first in API model list",
    )
    parser.add_argument(
        "--iterative_cot",
        action="store_true",
        help="Use constrained iterative chain of thought to generate the final answer",
    )
    parser.add_argument(
        "--as_text",
        action="store_true",
        help="Serialize prompt as text, removing chat structure",
    )
    parser.add_argument(
        "--pregenerated_outputs_dir",
        help="If using constrained iterative chain of thought, final answer can be derived using pregenerated outputs; folder containing pregenerated outputs TSVs",
    )
    parser.add_argument(
        "--answer_delimiter",
        default=DEFAULT_ANSWER_DELIMITER,
        help="Section header for extracting final answer",
    )
    parser.add_argument(
        "--answer_choices",
        nargs="+",
        default=DEFAULT_ANSWER_CHOICES,
        help="If using constrained iterative chain of thought, possible answer choices for constrained decoding",
    )
    parser.add_argument(
        "--num_parallel_requests",
        default=DEFAULT_NUM_PARALLEL_REQUESTS,
        type=int,
        help="Number of requests that can be made to the API in parallel",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        prompt_file=args.prompt_file,
        benchmark_tsv=args.benchmark_tsv,
        screens_dir=args.screens_dir,
        outputs_dir=args.outputs_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        iterative_cot=args.iterative_cot,
        as_text=args.as_text,
        pregenerated_outputs_dir=args.pregenerated_outputs_dir,
        answer_delimiter=args.answer_delimiter,
        answer_choices=args.answer_choices,
        num_parallel_requests=args.num_parallel_requests,
    )

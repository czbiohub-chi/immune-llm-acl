import argparse
import json
import os

import pandas as pd
from openai import OpenAI
from pqdm.threads import pqdm
from tqdm import tqdm

# fmt: off
# isort: off
from answer import (
    answer_with_constrained_iterative_cot,
    send_prompt,
    string_extract_answer,
)
from utils import read_pregenerated_outputs
# isort: on
# fmt: on

DEFAULT_ANSWER_DELIMITER = "Final Answer:"
DEFAULT_ANSWER_CHOICES = ["Yes", "No"]
DEFAULT_NUM_PARALLEL_REQUESTS = 32


def _process_prompt(
    *,  # enforce kwargs
    prompt_template: str,
    gene: str,
    crispr_strategy: str,
    cell_type: str,
    target_phenotype: str,
    output: str | None,
    model: str,
    client: OpenAI,
    answer_delimiter: str,
    answer_choices: list[str],
    iterative_cot: bool,
    as_text: bool,
) -> tuple[str, str, str]:
    prepared_prompt = (
        prompt_template.replace("{gene}", gene, 1)
        .replace("{crispr_strategy}", crispr_strategy, 1)
        .replace("{cell_type}", cell_type, 1)
        .replace("{phenotype}", target_phenotype, 1)
    )

    # require initial prompt template in chat format
    prepared_prompt = json.loads(prepared_prompt)
    if (
        not isinstance(prepared_prompt, list)
        or len(prepared_prompt) < 1
        or not isinstance(prepared_prompt[0], dict)
    ):
        # assumes if deserializes as list[dict], dict keys and values will be str
        raise ValueError("Prompt decoded as valid JSON but is not in chat format")

    # serialize as text
    if as_text:
        prepared_prompt = (
            "\n\n".join([msg["content"] for msg in prepared_prompt]) + "\n"
        )

    # generate output
    if output is None:
        output = send_prompt(
            client=client,
            model=model,
            prompt=prepared_prompt,
        )

    # extract answer
    if iterative_cot:
        answer = answer_with_constrained_iterative_cot(
            client=client,
            model=model,
            prompt=prepared_prompt,
            output=output,
            answer_delimiter=answer_delimiter,
            answer_choices=answer_choices,
        )
    else:
        answer = string_extract_answer(
            output=output,
            answer_delimiter=answer_delimiter,
        )

    return gene, output, answer


def run_screen(
    *,  # enforce kwargs
    genes: list[str],
    crispr_strategy: str,
    cell_type: str,
    target_phenotype: str,
    prompt_template: str,
    api_url: str,
    api_key: str = "EMPTY",
    model: str = "",
    iterative_cot: bool,
    as_text: bool,
    outputs: list[str] | None = None,
    answer_delimiter: str = DEFAULT_ANSWER_DELIMITER,
    answer_choices: list[str] = DEFAULT_ANSWER_CHOICES,
    num_parallel_requests: int = DEFAULT_NUM_PARALLEL_REQUESTS,
) -> tuple[list[str], list[str]]:
    if outputs is not None and not iterative_cot:
        raise ValueError(
            "Passing pregenerated outputs without iterative chaint of thought is redundant, do you just need to reextract answers?"
        )
    client = OpenAI(base_url=api_url, api_key=api_key)
    if not model:
        model = client.models.list().data[0].id

    answers = []
    if outputs is not None:
        assert len(genes) == len(outputs)

    kwargs = [
        {
            "prompt_template": prompt_template,
            "gene": gene,
            "crispr_strategy": crispr_strategy,
            "cell_type": cell_type,
            "target_phenotype": target_phenotype,
            "output": None if outputs is None else outputs[i],
            "model": model,
            "client": client,
            "answer_delimiter": answer_delimiter,
            "answer_choices": answer_choices,
            "iterative_cot": iterative_cot,
            "as_text": as_text,
        }
        for i, gene in enumerate(genes)
    ]

    if num_parallel_requests == 0:
        outs = []
        for kw in tqdm(kwargs):
            out = _process_prompt(**kw)
            outs.append(out)
    else:
        # errors in any subproc get returned in list, fails run when trying to sort
        outs = pqdm(
            array=kwargs,
            function=_process_prompt,
            n_jobs=num_parallel_requests,
            argument_type="kwargs",
            desc="Genes",
        )

    sort_genes = {gene: i for i, gene in enumerate(genes)}
    outs.sort(key=lambda goa: sort_genes[goa[0]])
    genes, outputs, answers = zip(*outs)
    return list(genes), list(outputs), list(answers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        required=True,
        help="File with text or json formatted prompt template; should contain format fields for {gene}, {crispr_strategy}, {cell_type}, and {phenotype}",
    )
    parser.add_argument(
        "--genes_tsv",
        required=True,
        help="TSV with candidate genes",
    )
    parser.add_argument(
        "--crispr_strategy",
        required=True,
        help="CRISPR screen strategy, e.g. CRISPR KO",
    )
    parser.add_argument(
        "--cell_type",
        required=True,
        help="CRISPR screen cell line, e.g. T cells",
    )
    parser.add_argument(
        "--target_phenotype",
        required=True,
        help="CRISPR screen target phenotype, e.g. increased IL2 secretion",
    )
    parser.add_argument(
        "--outputs_tsv",
        required=True,
        help="TSV to save results to",
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
        "--pregenerated_outputs_tsv",
        help="If using constrained iterative chain of thought, final answer can be derived using pregenerated outputs; TSV containing pregenerated outputs",
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
    assert not os.path.exists(args.outputs_tsv)

    with open(args.prompt_file) as f:
        prompt_template = f.read()

    genes = pd.read_csv(args.genes_tsv, sep="\t")["gene"]

    pregenerated_outputs = None
    if args.pregenerated_outputs_tsv is not None:
        pregenerated_outputs = read_pregenerated_outputs(
            outputs_file=args.pregenerated_outputs_tsv,
            screen_results=None,
        )

    _, outputs, answers = run_screen(
        genes=genes,
        crispr_strategy=args.crispr_strategy,
        cell_type=args.cell_type,
        target_phenotype=args.target_phenotype,
        prompt_template=prompt_template,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        iterative_cot=args.iterative_cot,
        as_text=args.as_text,
        outputs=pregenerated_outputs,
        answer_delimiter=args.answer_delimiter,
        answer_choices=args.answer_choices,
        num_parallel_requests=args.num_parallel_requests,
    )
    out_df = pd.DataFrame({"gene": genes, "output": outputs, "answer": answers})
    out_df.to_csv(args.outputs_tsv, index=False, sep="\t")

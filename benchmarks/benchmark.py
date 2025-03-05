import os
from pathlib import Path

import pandas as pd
from tqdm import trange


def load_benchmark(
    benchmark_path: str,
    *,  # enforce kwargs
    metadata_file: str = "screens.tsv",
) -> pd.DataFrame:
    benchmark_path = Path(benchmark_path)
    experiments = pd.read_csv(benchmark_path / metadata_file, sep="\t")
    hits = []

    for i in trange(len(experiments), desc="Screens"):
        screen_file = experiments.loc[i, "screen_results"]
        organism = None
        if "organism" in experiments:
            organism = experiments.loc[i, "organism"]
        screen_results = read_screen_results(
            screen_file=benchmark_path / screen_file,
            organism=organism,
        )
        assert (screen_results["HIT"] == "YES").all()
        scores = screen_results[experiments.loc[i, "binary_hit_col"].upper()]
        sign = experiments.loc[i, "target_hit"]
        if sign == "+":
            mask = scores > 0
        elif sign == "-":
            mask = scores < 0
        else:
            raise ValueError(f"unknown sign {sign}")
        _hits = list(
            zip(screen_results["OFFICIAL_SYMBOL"].to_list(), mask.astype(int).to_list())
        )
        for gene, hit in _hits:
            hits.append(
                {
                    "screen_file": screen_file,
                    "organism": organism,
                    "perturbation": experiments.loc[i, "crispr_strategy"],
                    "gene": gene,
                    "cell": experiments.loc[i, "cell_type"],
                    "phenotype": experiments.loc[i, "target_phenotype"],
                    "hit": hit,
                }
            )
    return pd.DataFrame(hits)


def read_screen_results(screen_file: str, organism: str | None = None) -> pd.DataFrame:
    screen_results = pd.read_csv(screen_file, sep="\t")
    if "IDENTIFIER_TYPE" not in screen_results:
        if organism is None:
            raise ValueError(
                "Must provide organism if screen file does not contain unique gene identifier"
            )
        if not os.path.exists("genomes"):
            raise ValueError("Cannot find reference genomes to align screened genes")
        genome_path_map = {
            "human": "genomes/genome_homo_sapien.tsv",
            "mouse": "genomes/genome_mus_musculus.tsv",
        }
        if organism not in genome_path_map:
            raise ValueError(
                f"Organism must be one of: {list(genome_path_map.keys())}, got {organism}"
            )
        genome = pd.read_csv(genome_path_map[organism], sep="\t")
        genome = genome[genome["Gene_Type"] == "PROTEIN_CODING"].reset_index(drop=True)
        gene_mask = (
            screen_results["OFFICIAL_SYMBOL"]
            .str.lower()
            .isin(genome["OFFICIAL_SYMBOL"].str.lower())
        )
    else:
        gene_mask = screen_results["IDENTIFIER_TYPE"] == "ENTREZ_GENE"
    dupe_mask = screen_results["OFFICIAL_SYMBOL"].duplicated(keep=False)
    mask = gene_mask & ~dupe_mask
    return screen_results[mask].reset_index(drop=True).copy()

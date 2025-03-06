import os

import pandas as pd


def read_screen_results(screen_file: str, organism: str | None = None) -> pd.DataFrame:
    screen_results = pd.read_csv(screen_file, sep="\t")
    if "IDENTIFIER_TYPE" not in screen_results:
        if organism is None:
            raise ValueError(
                "Must provide organism if screen file does not contain unique gene identifier"
            )
        if not os.path.exists("../training/data/genomes"):
            raise ValueError("Cannot find reference genomes to align screened genes")
        genome_path_map = {
            "human": "../training/data/genomes/genome_homo_sapien.tsv",
            "mouse": "../training/data/genomes/genome_mus_musculus.tsv",
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


def read_pregenerated_outputs(
    *,  # enforce kwargs
    outputs_file: str,
    screen_results: pd.DataFrame | None = None,
) -> list[str]:
    outputs_df = pd.read_csv(outputs_file, sep="\t")
    outputs_df = outputs_df.set_index("gene")
    if screen_results is not None:
        outputs_df = outputs_df.loc[screen_results["OFFICIAL_SYMBOL"]]
        assert screen_results["OFFICIAL_SYMBOL"].equals(pd.Series(outputs_df.index))
    outputs = outputs_df["output"]
    return outputs.to_list()

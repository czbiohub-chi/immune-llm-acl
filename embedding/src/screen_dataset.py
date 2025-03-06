from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# 5 randomly selected screens for consistent validation
VALIDATION_SCREENS = [1545, 613, 316, 153, 1906]

CELL_COL = "CELL_TYPE"
PHENOTYPE_COL = "PHENOTYPE_NOTES"
METHOD_COL = "LIBRARY_METHODOLOGY"
GENE_COL = "OFFICIAL_SYMBOL"
GENE_ID_COL = "IDENTIFIER_ID"
GENE_ID_TYPE_COL = "IDENTIFIER_TYPE"
HIT_COL = "HIT"
ORGANISM_COL = "ORGANISM_OFFICIAL"
SCREEN_COL = "#SCREEN_ID"

EMB_T = dict[str, np.ndarray]
ITEM_T = dict[str, str | int | np.ndarray]


class ScreenDataset(Dataset):
    def __init__(
        self,
        *,  # enforce kwargs
        data_dir: str,
        use_summarized: bool,
        hits_only: bool,
        blacklist: list[int] | None = None,
        whitelist: list[int] | None = None,
    ):
        data_dir = Path(data_dir)

        screens = load_screen_metadata(data_dir)
        if blacklist is not None and whitelist is not None:
            raise ValueError("Redundant to pass both blacklist and whitelist.")
        mask = screens.index.notna()
        if blacklist is not None:
            mask = ~screens[SCREEN_COL].isin(blacklist)
        elif whitelist is not None:
            mask = screens[SCREEN_COL].isin(whitelist)
        print(f"Filtered to {mask.sum()} screens using user defined filters")
        screens = screens.loc[mask].reset_index(drop=True)

        human_genome, mouse_genome = load_reference_genome(data_dir)

        genome_map = {
            "H. sapiens": human_genome,
            "M. musculus": mouse_genome,
        }

        embs = load_embeddings(data_dir, use_summarized)
        gene_embs_map = {
            "H. sapiens": embs["gene_embs_human"],
            "M. musculus": embs["gene_embs_mouse"],
        }

        counts = dict()
        count_all_no = 0
        self.data: list[ITEM_T] = []
        for _, screen_metadata in tqdm(
            screens.iterrows(),
            total=len(screens),
            desc="Loading data",
        ):
            screen_id, screen_counts, screen_data, all_no = extract_screen(
                data_dir=data_dir,
                screen_metadata=screen_metadata,
                genome_map=genome_map,
                method_embs=embs["method_embs"],
                cell_embs=embs["cell_embs"],
                phenotype_embs=embs["phenotype_embs"],
                gene_embs_map=gene_embs_map,
                hits_only=hits_only,
            )
            if all_no:
                count_all_no += 1
            counts[screen_id] = screen_counts
            self.data.extend(screen_data)

        # sanity checks
        recovered_screen_ids = {x["screen_id"] for x in self.data}
        correction = 0
        if hits_only:
            correction = count_all_no
        assert (len(recovered_screen_ids) + correction) == len(screens)

        counts = pd.DataFrame.from_dict(counts, orient="index")
        counts.index.name = SCREEN_COL

        assert len(self.data) == counts["filtered"].sum()

        counts["diff_n"] = counts["original"] - counts["filtered"]
        counts["diff_frac"] = counts["diff_n"] / counts["original"]

        print("Extracted genes:")
        print(counts[["filtered", "original"]].sum())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> ITEM_T:
        return self.data[i]


def load_screen_metadata(data_dir: Path) -> pd.DataFrame:
    human_screens = pd.read_csv(data_dir / "screens/index_homo_sapiens.tsv", sep="\t")
    mouse_screens = pd.read_csv(data_dir / "screens/index_mus_musculus.tsv", sep="\t")

    # FULL_SIZE column is not accurate to actual data, e.g. see screen #25
    all_screens = pd.concat([human_screens, mouse_screens])
    print(f"Originally {len(all_screens)} screens")

    screens = all_screens[~all_screens["SIGNIFICANCE_CRITERIA"].str.contains("OR")]
    screens = screens.reset_index(drop=True)
    print(f"Filtered to {len(screens)} screens using default filters")

    screens[PHENOTYPE_COL] = screens["PHENOTYPE"] + ". " + screens["NOTES"]
    return screens


def load_reference_genome(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    human_genome = pd.read_csv("../genomes/genome_homo_sapiens.tsv", sep="\t")
    mouse_genome = pd.read_csv("../genomes/genome_mus_musculus.tsv", sep="\t")

    human_genome = human_genome[human_genome["Gene_Type"] == "PROTEIN_CODING"]
    mouse_genome = mouse_genome[mouse_genome["Gene_Type"] == "PROTEIN_CODING"]

    assert not human_genome[GENE_COL].duplicated().any()
    assert not mouse_genome[GENE_COL].duplicated().any()

    return human_genome, mouse_genome


def load_embeddings(data_dir: Path, use_summarized: bool) -> dict[str, EMB_T]:
    load = lambda x: np.load(x, allow_pickle=True).item()
    if use_summarized:
        method_embs = load(data_dir / "embeddings/summarized_methods.npy")
        cell_embs = load(data_dir / "embeddings/summarized_cells.npy")
        phenotype_embs = load(data_dir / "embeddings/summarized_phenotypes.npy")
        gene_embs_human = load(data_dir / "embeddings/summarized_genes_human.npy")
        gene_embs_mouse = load(data_dir / "embeddings/summarized_genes_mouse.npy")
    else:
        method_embs = load(data_dir / "embeddings/methods.npy")
        cell_embs = load(data_dir / "embeddings/cells.npy")
        phenotype_embs = load(data_dir / "embeddings/phenotypes.npy")
        gene_embs_human = load(data_dir / "embeddings/genes_human.npy")
        gene_embs_mouse = load(data_dir / "embeddings/genes_mouse.npy")

    return {
        "method_embs": method_embs,
        "cell_embs": cell_embs,
        "phenotype_embs": phenotype_embs,
        "gene_embs_human": gene_embs_human,
        "gene_embs_mouse": gene_embs_mouse,
    }


def extract_screen(
    *,  # enforce kwargs
    data_dir: Path,
    screen_metadata: pd.Series,
    genome_map: dict[str, pd.DataFrame],
    method_embs: EMB_T,
    cell_embs: EMB_T,
    phenotype_embs: EMB_T,
    gene_embs_map: dict[str, EMB_T],
    hits_only: bool,
) -> tuple[int, dict[str, int], list[ITEM_T], bool]:
    screen_id = screen_metadata[SCREEN_COL]
    method_term = screen_metadata[METHOD_COL]
    phenotype_term = screen_metadata[PHENOTYPE_COL]
    cell_term = screen_metadata[CELL_COL]
    organism_term = screen_metadata[ORGANISM_COL]

    method_emb = method_embs[method_term]
    phenotype_emb = phenotype_embs[phenotype_term]
    cell_emb = cell_embs[cell_term]

    genome = genome_map[organism_term]
    gene_embs = gene_embs_map[organism_term]

    raw_screen = pd.read_csv(
        data_dir / f"screens/BIOGRID-ORCS-SCREEN_{screen_id}-1.1.16.screen.tab.txt",
        sep="\t",
    )
    screen = raw_screen

    # only select genes with ENTREZ_GENE identifiers
    screen = screen[screen[GENE_ID_TYPE_COL] == "ENTREZ_GENE"].reset_index(drop=True)

    # convert identifiers to int
    if screen[GENE_ID_COL].dtype != int:
        screen = screen[screen[GENE_ID_COL].str.isnumeric()].reset_index(drop=True)
        screen[GENE_ID_COL] = screen[GENE_ID_COL].astype(int)

    # may have some mismatch between reference genome version and genome used to align screen
    mask = screen[GENE_ID_COL].isin(genome[GENE_ID_COL])
    screen = screen[mask].reset_index(drop=True)

    data = []
    gene_ids = screen[GENE_ID_COL]
    gene_terms = screen[GENE_COL]
    hits = screen[HIT_COL]
    for gene_id, gene_term, hit in zip(gene_ids, gene_terms, hits):
        gene_emb = gene_embs[gene_id]
        if hits_only and hit != "YES":
            continue
        data.append(
            {
                "screen_id": screen_id,
                "organism_term": organism_term,
                "method_term": method_term,
                "gene_id": gene_id,
                "gene_term": gene_term,
                "cell_term": cell_term,
                "phenotype_term": phenotype_term,
                "method_emb": method_emb,
                "gene_emb": gene_emb,
                "cell_emb": cell_emb,
                "phenotype_emb": phenotype_emb,
                "hit": hit,
            }
        )

    counts = {"original": len(raw_screen), "filtered": len(data)}
    all_no = (hits == "NO").all()
    return screen_id, counts, data, all_no

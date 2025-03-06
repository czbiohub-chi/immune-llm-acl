import pandas as pd


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

import polars as pl
from typing import Iterator


def str_to_date(*dfs:pl.DataFrame, col:str, format="%Y-%m-%d") -> Iterator[pl.DataFrame]:
    for df in dfs:
        df = df.with_columns(
            pl.col(col).str.strptime(pl.Date, format="%Y-%m-%d")
        )
        yield df

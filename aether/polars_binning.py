from IPython.display import display
import polars as pl
from typing import Iterator
import matplotlib.ticker as mticker
import math


def get_col_bin_auto(
    *dfs: pl.DataFrame,
    col: str,
    num_n_bins: int = 10,
    num_sig_digits: int = 3,
    dt_truncate_unit: str = "1mo",
    verbose: bool = False
) -> tuple[pl.DataFrame, ...]:
    """
    
    Examples:
    ```python
    # df_original, df_test_original = df.clone(), df_test.clone()

    df_new, df_test_new, df_bin_detail_info = aether.get_col_bin_auto(df_original, df_test_original, col='date')
    df, df_test = df.with_columns(df_new), df_test.with_columns(df_test_new)
    display(df, df_test)
    ```
    """
    dfs_with_col = [df for df in dfs if col in df.columns]
    if not dfs_with_col:
        raise ValueError(f"指定された列 `{col}` を持つDataFrameが1つもありません")
    
    dtype = dfs_with_col[0].schema[col]

    if dtype.is_numeric():
        return get_col_bin_numeric(*dfs, col=col, num_n_bins=num_n_bins, num_sig_digits=num_sig_digits, verbose=verbose)
    elif dtype in (pl.Date, pl.Datetime):
        return get_col_bin_datetime(*dfs, col=col, dt_truncate_unit=dt_truncate_unit, verbose=verbose)
    elif dtype in (pl.Utf8, pl.Categorical):
        return get_col_bin_categorical(*dfs, col=col, verbose=verbose)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_col_bin_numeric(
    *dfs: pl.DataFrame,
    col: str,
    num_n_bins: int = 10,
    num_sig_digits: int = 3,
    verbose: int = 0,
) -> tuple[pl.DataFrame, ...]:

    col_bin = f"{col}_bin"
    dfs_with_col = [df for df in dfs if col in df.columns]
    df_concat = pl.concat([df.select(col) for df in dfs_with_col])
    min_val = df_concat.select(col).to_series().min()
    max_val = df_concat.select(col).to_series().max()
    locator = mticker.MaxNLocator(nbins=num_n_bins)
    bins = locator.tick_values(min_val, max_val)
    bins = sorted(set(_round_sig(b, num_sig_digits) for b in bins))# 有効数字で丸める⇒重複を排す(丸めた影響でかぶりが出る可能性がある)⇒並びが崩れるので並べ直す
    breaks = bins[1:-1]
    labels = _make_bin_labels(bins)

    starts = bins[:-1]
    ends = bins[1:]
    centers = [(s + e) / 2 for s, e in zip(starts, ends)]

    dfs_bin = []
    for df in dfs:
        if col in df.columns:
            df_bin = df.select(pl.col(col).cut(breaks=breaks, labels=labels).alias(col_bin))
        else:
            # 元の列がない場合、全部値がNullの列を返す
            df_bin = pl.DataFrame({col_bin: [None] * df.height})
        dfs_bin.append(df_bin)
        if verbose:
            print("df_bin:")
            display(df_bin)

    df_bin_detail_info = pl.DataFrame({
        col_bin: labels,
        f"{col_bin}_start": starts,
        f"{col_bin}_end": ends,
        f"{col_bin}_median": centers
    }).with_columns([pl.col(col_bin).cast(pl.Categorical)])

    if verbose:
        print("df_bin_detail_info:")
        display(df_bin_detail_info)

    return (*dfs_bin, df_bin_detail_info)


def get_col_bin_datetime(
    *dfs: pl.DataFrame,
    col: str,
    dt_truncate_unit: str = "1mo",
    verbose: int = 0,
) -> tuple[pl.DataFrame, ...]:
    
    col_bin = f"{col}_bin"
    dfs_with_col = [df for df in dfs if col in df.columns]
    df_concat = pl.concat([df.select(col) for df in dfs_with_col])

    # minをbin始まりとしてtruncate(切り捨て)して、maxに1期間分余分に足したもの(ケツのbinを切り捨てずに切り出すため。後の処理で使う)
    min_truncated = df_concat.select(pl.col(col).min().dt.truncate(dt_truncate_unit)).item()
    max_plus_1_unit = df_concat.select(pl.col(col).max().dt.offset_by(dt_truncate_unit)).item()

    # ビンの生成
    is_date = df_concat[col].dtype == pl.Date
    range_fn = pl.date_range if is_date else pl.datetime_range
    bin_starts_plus1 = range_fn(start=min_truncated, end=max_plus_1_unit, interval=dt_truncate_unit, eager=True) # ケツに1期間を足したリスト
    bin_starts = bin_starts_plus1[:-1].to_list() # ケツの1期間は削る
    bin_ends = bin_starts_plus1[1:].to_list() # 最初の1期間は削る
    bin_medians = [s + (e - s) // 2 for s, e in zip(bin_starts, bin_ends)]

    # ビンの詳細テーブル
    df_bin_detail_info = pl.DataFrame({
        col_bin: bin_starts,
        f"{col_bin}_start": bin_starts,
        f"{col_bin}_end": bin_ends,
        f"{col_bin}_median": bin_medians
    })

    dfs_bin = []
    for df in dfs:
        if col in df.columns:
            col_expr = pl.col(col).dt.truncate(dt_truncate_unit)
            if not is_date:
                # only Datetime needs cast
                unit = df_bin_detail_info.schema[col_bin].time_unit
                col_expr = col_expr.dt.cast_time_unit(unit)
            df_bin = df.select(col_expr.alias(col_bin))
        else:
            df_bin = pl.DataFrame({col_bin: [None] * df.height})
        dfs_bin.append(df_bin)
        if verbose:
            print("df_bin:")
            display(df_bin)

    if verbose:
        print("df_bin_detail_info:")
        display(df_bin_detail_info)

    return (*dfs_bin, df_bin_detail_info)


def get_col_bin_categorical(
    *dfs: pl.DataFrame,
    col: str,
    verbose: int = 0,
) -> tuple[pl.DataFrame, ...]:

    col_bin = f"{col}_bin" # カテゴリカルの場合何もしないので意味がないが、他と揃えるためにbin列として新設する
    dfs_bin = []
    for df in dfs:
        if col in df.columns:
            df_bin = df.select(pl.col(col).alias(col_bin))
        else:
            df_bin = pl.DataFrame({col_bin: [None] * df.height})
        dfs_bin.append(df_bin)
        if verbose:
            print("df_bin:")
            display(df_bin)
    return (*dfs_bin, None)


def _round_sig(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _make_bin_labels(bins: list[float]) -> list[str]:
    return [f"{start}–{end}" for start, end in zip(bins[:-1], bins[1:])]

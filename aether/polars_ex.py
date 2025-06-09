from IPython.display import display, Markdown

import polars as pl
import altair as alt
import matplotlib.pyplot as plt

import types
from typing import Sequence, Optional, Union

from tqdm import tqdm


alt.themes.enable('dark')
plt.style.use('dark_background')


"""
★read_csv_or_parquet関数 ※csvをparquetにしといて次から早く読む自作関数
"""
def read_csv_or_parquet(csv_path: str, **kwargs) -> pl.DataFrame:
    """
    CSVまたは対応するParquetファイルを読み込み、PolarsのDataFrameを返す。

    - Parquetファイルが存在すればそれを優先して読み込む（高速）
    - Parquetファイルが存在しなければCSVを読み込み、Parquetとして保存

    Parameters
    ----------
    csv_path : str
        読み込むCSVファイルのパス

    Returns
    -------
    pl.DataFrame
        読み込まれたPolarsのDataFrame
    """
    import os


    # 拡張子を.parquetに差し替える
    base, _ = os.path.splitext(csv_path)
    parquet_path = base + ".parquet"

    if os.path.exists(parquet_path):
        df = pl.read_parquet(parquet_path)
    else:
        df = pl.read_csv(csv_path, **kwargs)
        df.write_parquet(parquet_path)

    return df


"""
◆_format_sig関数
"""
def _format_sig(x: float, sig_digits: int) -> str:
    """
    数値 `x` を指定された有効数字や桁数で文字列として整形する。
    
    - None → 空文字列
    - float かつ int と等しければ整数文字列（例: 5.0 → '5'）
    - それ以外は 'g' フォーマット（有効数字）を使いつつ、
      指数表記が出た場合は 'f' にフォールバックして固定小数点表示
    - 不要な小数点・末尾ゼロを除去
    """
    if x is None:
        return ""
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        s = f"{x:.{sig_digits}g}"
        if "e" in s or "E" in s:
            s = f"{x:.{sig_digits}f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return s
    return str(x)


"""
★describe_ex関数
"""
def describe_ex(df: pl.DataFrame, detailed: bool = True, sig_digits: int = 2) -> pl.DataFrame:
    """
    Polars DataFrame に対して拡張統計要約を行う関数。

    デフォルトの `df.describe()` に加え、以下の情報も表示する：
    - 欠損数（missing）、非欠損数（non-missing）
    - 平均、標準偏差、中央値、最頻値（top）とその件数（top_count）
    - ユニーク値数（n_unique）
    - データ型（dtype）

    パラメータ
    ----------
    df : pl.DataFrame
        対象の Polars データフレーム。

    detailed : bool, optional
        True の場合、全統計情報を出力。False または None の場合、`df.describe()` 相当の出力を整形して返す。
        None のときはグローバル変数 `detailed` を参照（未定義時は False 扱い）。

    sig_digits : int, optional (default=2)
        数値の有効桁数。None の場合は丸め処理を行わない。

    戻り値
    -------
    pl.DataFrame
        統計情報を文字列型でまとめた DataFrame。各列に対する統計値を行方向に表示。

    使用例
    -------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1, 2, 2, None], "y": ["a", "b", "a", "a"]})
    >>> describe_ex(df, detailed=True)

    備考
    ----
    - detailed=False の場合、`pl.DataFrame.describe()` に近い形式（整形済）を返す。
    - 非数値列（文字列など）でも top やユニーク件数などを集計可能。
    - 型情報や欠損状況なども一緒に確認したい場合に便利。
    """
    # if detailed is None:
    #     detailed = globals().get("detailed", False)

    describe_schema = df.schema

    if not detailed:
        df_simple = df.describe(percentiles=[0.5])
        if sig_digits is not None:
            df_simple = df_simple.with_columns([
                pl.col(col)
                .map_elements(lambda x: _format_sig(x, sig_digits), return_dtype=pl.String)
                .alias(col)
                for col in df_simple.columns if col != "statistic"
            ])
        return df_simple.cast(pl.Utf8)

    # === 統計行の収集 ===
    stat_labels = [
        ("non-missing", lambda s: s.len()),
        ("missing", lambda s: s.null_count()),
        ("mean", lambda s: s.mean()),
        ("std", lambda s: s.std()),
        ("min", lambda s: s.min()),
        ("median", lambda s: s.median()),
        ("max", lambda s: s.max()),
    ]

    stats_rows = []
    for label, func in stat_labels:
        row = {"statistic": label}
        for col in df.columns:
            try:
                val = df.select(func(pl.col(col)))[0, 0]
                row[col] = _format_sig(val, sig_digits)
            except:
                row[col] = ""
        stats_rows.append(row)

    # === 補足統計（dtype, top, top_count, n_unique） ===
    stats_df = df.select([
        pl.col(col).n_unique().alias(f"{col}_n_unique") for col in df.columns
    ]).to_dict(as_series=False)

    rows = {
        "dtype": {"statistic": "dtype"},
        "top": {"statistic": "top"},
        "top_count": {"statistic": "top_count"},
        "n_unique": {"statistic": "n_unique"},
    }

    for col in df.columns:
        dtype = describe_schema[col]
        n_val = stats_df[f"{col}_n_unique"][0]
        assert isinstance(n_val, (int, float)), f"n_unique for column '{col}' must be numeric"

        try:
            top_row = (
                df.select(pl.col(col))
                .group_by(col)
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .limit(1)
            )
            top_val = top_row[0, col] if top_row.height > 0 else None
            top_count = top_row[0, "count"] if top_row.height > 0 else 0
        except:
            top_val = None
            top_count = None

        rows["dtype"][col] = str(dtype)
        rows["top"][col] = _format_sig(top_val, sig_digits)
        rows["top_count"][col] = str(top_count)
        rows["n_unique"][col] = str(int(n_val))

    extra_rows = [rows[k] for k in ["dtype", "top", "top_count", "n_unique"]]
    df_result = pl.DataFrame(stats_rows + extra_rows).cast(pl.Utf8)

    # 行の並び替え
    desired_order = [
        "dtype", "non-missing", "missing", "n_unique",
        "mean", "std", "min", "median", "max", "top", "top_count"
    ]
    actual_stats = df_result.select("statistic").to_series().to_list()
    sorted_stats = [s for s in desired_order if s in actual_stats] + [s for s in actual_stats if s not in desired_order]

    df_result = df_result.with_columns(
        pl.col("statistic").map_elements(lambda s: sorted_stats.index(s), return_dtype=pl.Int32).alias("__sort_order")
    ).sort("__sort_order").drop("__sort_order")

    return df_result

# Monkey patch（任意）
pl.DataFrame.describe_ex = describe_ex


"""
★get_bin_columns関数
"""
def get_bin_column(
    *dfs: pl.DataFrame,
    col: str,
    num_n_bins: int = 10,
    num_sig_digits: int = 3,
    dt_truncate_unit: str = "1mo",
    verbose:bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    指定列に対してビニング（bin列の追加）を行い、各DataFrameに対応するbin列とbin情報を返す関数。

    数値・日時・カテゴリ型に対応し、ビニングの方法を自動的に判別する。

    パラメータ
    ----------
    *dfs : pl.DataFrame
        処理対象の複数の Polars DataFrame。対象列が存在するものだけを処理対象とする。

    col : str
        ビニング対象の列名。

    num_n_bins : int, optional (default=10)
        数値列のときに作成するビンの最大数。`matplotlib.ticker.MaxNLocator` を用いて自動で区切る。

    num_sig_digits : int, optional (default=3)
        数値列のビン境界値を有効数字で丸める桁数。

    dt_truncate_unit : str, optional (default="1mo")
        日付・日時列に対して `.dt.truncate()` する際の単位（例: "1mo", "1d" など）。

    verbose : bool, optional (default=False)
        中間処理の表示を行う（Jupyterなどで `display()` を使用）。

    戻り値
    -------
    tuple[pl.DataFrame, ..., pl.DataFrame]
        - 各入力 DataFrame に対応する bin 列だけを持つ DataFrame のタプル（入力の順に並ぶ）。
        - 最後の要素は bin の詳細情報（開始、終了、中央値など）を持つ DataFrame（数値・日付列のときのみ）。

    使用例
    -------
    >>> df1 = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
    >>> df2 = pl.DataFrame({"val": [6, 7, 8, 9, 10]})
    >>> df1_bin, df2_bin, df_bin_info = get_bin_column(df1, df2, col="val")

    備考
    ----
    - 数値列の場合：等間隔ビンに分割し、`cut` によってラベルを付けたカテゴリ列を生成。
    - 日付・日時列の場合：`dt.truncate()` により期間単位でビンを作成。
    - 文字列・カテゴリ列の場合：元の値をそのまま bin として利用。
    - 元の列が存在しない DataFrame に対しては、すべて `None` の bin 列を返す。
    - bin詳細（`_start`, `_end`, `_median`）は最終返却要素に格納。
    """
    import matplotlib.ticker as mticker


    col_bin = col + '_bin'
    # # 最初のdfから型を取得（アサート込み）
    # assert all(col in df.columns for df in dfs), f"列 `{col}` が存在しないDataFrameがあります"
    # dtype = dfs[0].schema[col]
    # assert all(df.schema[col] == dtype for df in dfs), f"`{col}` の型がDataFrame間で一致していません"
    # df_concat = pl.concat([df.select(col) for df in dfs])
    # colを持っているDataFrameのみ抽出
    dfs_with_col = [df for df in dfs if col in df.columns]
    if not dfs_with_col:
        raise ValueError(f"指定された列 `{col}` を持つDataFrameが1つもありません")
    # 最初にcolを持っているdfから型取得（型不一致アサートは省略可）
    dtype = dfs_with_col[0].schema[col]
    # min/max を取るための結合用
    df_concat = pl.concat([df.select(col) for df in dfs_with_col])

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
        min_val = df_concat.select(col).to_series().min()
        max_val = df_concat.select(col).to_series().max()
        locator = mticker.MaxNLocator(nbins=num_n_bins)
        bins = locator.tick_values(min_val, max_val)
        if verbose:
            print("bins(mticker.MaxNLocator):")
            display(bins)
        bins = [_round_sig(bin, num_sig_digits) for bin in bins] # 有効数字で丸める
        bins = sorted(list(set(bins))) # 重複を排す(丸めた影響でかぶりが出る可能性がある)⇒並びが崩れるので並べ直す
        breaks = bins[1:-1]
        labels = _make_bin_labels(bins)
        starts = bins[:-1]
        ends = bins[1:]
        centers = [(s + e) / 2 for s, e in zip(starts, ends)]

        dfs_bin = []
        for df in dfs:
            # df_bin = df.select([
            #     pl.col(col).cut(breaks=breaks, labels=labels).alias(f"{col}_bin")
            # ])
            # dfs_bin.append(df_bin)
            if col in df.columns:
                df_bin = df.select([
                    pl.col(col).cut(breaks=breaks, labels=labels).alias(col_bin)
                ])
            else:
                # 元の列がない場合、全部値がNullの列を返す
                df_bin = pl.DataFrame({col_bin: pl.Series(name=col_bin, values=[None] * df.height, dtype=dtype)})
            dfs_bin.append(df_bin)
            if verbose:
                print("df_bin:")
                display(df_bin)
            assert df.height == df_bin.height, (
                f"Row count mismatch: df={df.height}, df_bin={df_bin.height}"
            )

        df_bin_detail_info = pl.DataFrame({
            f"{col_bin}": labels,
            f"{col_bin}_start": starts,
            f"{col_bin}_end": ends,
            f"{col_bin}_median": centers
        }).with_columns([
            pl.col(f"{col}_bin").cast(pl.Categorical)
        ])
        if verbose:
            print("df_bin_detail_info:")
            display(df_bin_detail_info)

    elif dtype in (pl.Date, pl.Datetime):
        # bin列を追加（truncate処理）
        dfs_bin = []
        for df in dfs:
            if col in df.columns:
                df_bin = df.select(
                    pl.col(col).dt.truncate(dt_truncate_unit).alias(f"{col}_bin")
                )
            else:
                # 元の列がない場合、全部値がNullの列を返す
                df_bin = pl.DataFrame({col_bin: pl.Series(name=col_bin, values=[None] * df.height, dtype=dtype)})
            dfs_bin.append(df_bin)
            if verbose:
                print("df_bin:")
                display(df_bin)

        # 最小・最大日時を取得
        min_date = df_concat.select(pl.col(col).min().dt.truncate(dt_truncate_unit)).item()
        # max_date = df.select(pl.col(col).max()).item()
        max_date_plus_1_unit = df_concat.select(pl.col(col).max().dt.offset_by(dt_truncate_unit)).item()

        # 連続したbin開始点の生成（例：1日ごとの月初）
        range_fn = pl.date_range if dtype == pl.Date else pl.datetime_range
        bin_starts_plus_1_unit = range_fn(
            start=min_date,
            end=max_date_plus_1_unit,
            interval=dt_truncate_unit,
            eager=True
        ) # ケツに1期間を足したリスト

        # bin_end列（1つ先のstart）
        bin_starts = bin_starts_plus_1_unit[:-1].to_list() # ケツの1期間は削る
        bin_ends = bin_starts_plus_1_unit[1:].to_list() # 最初の1期間は削る

        bin_medians = [start + (end - start) // 2 for start, end in zip(bin_starts, bin_ends)]

        # DataFrame化
        # df_bin_detail_info = pl.DataFrame({
        #     f"{col_bin}": bin_starts,
        #     f"{col_bin}_start": bin_starts,
        #     f"{col_bin}_end": bin_ends,
        #     f"{col_bin}_median": bin_medians
        # })
        # 時刻ビン詳細テーブルを先に作る（μs精度）
        df_bin_detail_info = pl.DataFrame({
            f"{col_bin}": bin_starts,
            f"{col_bin}_start": bin_starts,
            f"{col_bin}_end": bin_ends,
            f"{col_bin}_median": bin_medians
        })

        # ビン列の精度を取得（"us", "ms", etc）
        target_unit = df_bin_detail_info.schema[f"{col_bin}"].time_unit

        # 各dfに bin列を追加（truncate & cast）
        dfs_bin = []
        for df in dfs:
            if col in df.columns:
                df_bin = df.select(
                    pl.col(col).dt.truncate(dt_truncate_unit).dt.cast_time_unit(target_unit).alias(col_bin)
                )
            else:
                df_bin = pl.DataFrame({col_bin: pl.Series(name=col_bin, values=[None] * df.height, dtype=pl.Datetime("μs"))})
            dfs_bin.append(df_bin)

        if verbose:
            print("df_bin_detail_info:")
            display(df_bin_detail_info)

    elif dtype in (pl.Utf8, pl.Categorical):
        dfs_bin = []
        for df in dfs:
            if col in df.columns:
                df_bin = df.select([pl.col(col).alias(f"{col_bin}")])#[f"{col}_bin"]
            else:
                # 元の列がない場合、全部値がNullの列を返す
                df_bin = pl.DataFrame({col_bin: pl.Series(name=col_bin, values=[None] * df.height, dtype=dtype)})
            dfs_bin.append(df_bin)
            if verbose:
                print("df_bin:")
                display(df_bin)
        df_bin_detail_info = None

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return (*dfs_bin, df_bin_detail_info)


def _round_sig(x: float, sig: int) -> float:
    import math


    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _make_bin_labels(bins: list[float]) -> list[str]:
    labels = [
        f"{start}–{end}"
        for start, end in zip(bins[:-1], bins[1:])
    ]
    return labels


"""
★plot_histogram関数
- ↑のbinning関数を内部で使って、それを使って集計したものをHistogramにする処理
"""
def plot_histogram(
    df: pl.DataFrame,
    col_bin: str,
    df_bin_detail_info: pl.DataFrame = None,
    col_color: str = None,
    bar_opacity: float = 0.5,
    num_x_scale_zero: bool = False,
    normalize_histogram: bool = False,
    title: str = None,
    verbose: bool = False
) -> alt.Chart:
    """
    Altair でヒストグラム（棒グラフ）を描画する関数。

    ビン列（カテゴリ、数値、または日付のビン）をもとに集計し、棒グラフとして表示する。
    日付・数値ビンの場合は `df_bin_detail_info` を渡すことで棒の幅を明示的に設定可能。

    パラメータ
    ----------
    df : pl.DataFrame
        入力データ。ビン列を含んでいる必要がある。

    col_bin : str
        ビニング済みの列名（"xxx_bin" など）。棒グラフのX軸となる。

    df_bin_detail_info : pl.DataFrame, optional
        ビンの詳細情報（`_start`, `_end` 列など）を含むDataFrame。
        これが指定されている場合、Altair の `x` と `x2` を用いて棒の幅を連続値として描画する。

    col_color : str, optional
        グループ毎に色分けしたい列名。None の場合は色分けしない。

    bar_opacity : float, optional (default=0.5)
        棒の透明度。

    num_x_scale_zero : bool, optional (default=False)
        X軸（連続値）のスケールにゼロを含めるかどうか。

    normalize_histogram : bool, optional (default=False)
        ヒストグラムを相対頻度（割合）として正規化するかどうか。

    title : str, optional
        グラフタイトル。None の場合は `col_bin` から自動推定。

    verbose : bool, optional
        処理途中のDataFrameを `display()` で表示するデバッグモード。

    戻り値
    -------
    alt.Chart
        Altair による棒グラフチャートオブジェクト。

    使用例
    -------
    >>> chart = plot_histogram(df, col_bin="age_bin", df_bin_detail_info=bin_info)
    >>> chart.display()

    備考
    ----
    - `df_bin_detail_info` を渡すと、棒の幅や位置が連続的に表示され、視認性が向上する。
    - カテゴリビンの場合は `x` のみ、連続ビンの場合は `x` + `x2` によるレンジ指定を行う。
    - Polars と Altair を組み合わせて軽量な可視化が可能。
    """    
    if verbose:
        print(f"col_bin: {col_bin}")
        print('df:')
        display(df)
    
    if title is None:
        if col_bin.endswith("_bin"):
            title = col_bin.removesuffix("_bin")
        else:
            title = col_bin

    col_y = 'count'
    group_keys = [col_bin]
    if col_color and (col_bin != col_color):
        group_keys.append(col_color)
    if normalize_histogram == False:
        expr = pl.len().alias(col_y)
    else:
        expr = (pl.len() / pl.lit(len(df))).alias(col_y)
    if verbose:
        print(f'group_keys: {group_keys}')
    df_agg = df.group_by(pl.col(group_keys)).agg(expr)
    if verbose:
        print('df_agg:')
        display(df_agg)

    if df_bin_detail_info is not None:
        if verbose:
            print('df_bin_detail_info:')
            display(df_bin_detail_info)

        df_agg = df_agg.join(df_bin_detail_info, on=col_bin, how='left')
        if verbose:
            print('df_agg(joint):')
            display(df_agg)

        col_bin_start = f"{col_bin}_start"
        col_bin_end = f"{col_bin}_end"
        assert col_bin_start in df_agg.columns, f"{col_bin_start} が df_agg に存在しません"
        assert col_bin_end in df_agg.columns, f"{col_bin_end} が df_agg に存在しません"

        df_plot = df_agg.with_columns([
            pl.col(col_bin_start).alias("bin_left"),
            pl.col(col_bin_end).alias("bin_right"),
            pl.col(col_y).alias("bin_top"),
            pl.lit(0).alias("bin_bottom")
        ])
        if verbose:
            print('df_plot:')
            display(df_plot)

        dtype = df_plot.schema["bin_left"]
        bin_type = "temporal" if dtype in (pl.Datetime, pl.Date) else "quantitative"

        chart = alt.Chart(df_plot).encode(
            x=alt.X("bin_left", type=bin_type, title=None, scale=alt.Scale(zero=num_x_scale_zero)),
            x2="bin_right",
            y=alt.Y("bin_bottom:Q", title="count"),
            y2=alt.Y2("bin_top:Q")
        )

    else:
        chart = alt.Chart(df_agg).encode(
            x=alt.X(f"{col_bin}:N", title="category"),
            y=alt.Y(f"{col_y}:Q", title="count")
        )

    chart = chart.mark_bar(opacity=bar_opacity, stroke='gray', strokeWidth=1).properties(title=title)
    if col_color:
        chart = chart.encode(color=alt.Color(f"{col_color}:N"))
    
    return chart


"""
★plot_line_point_over_bin関数
"""
def plot_line_point_over_bin(
    df: pl.DataFrame,
    col_bin: str,
    col_target: str,
    col_color: str = None,
    agg_func = pl.mean,
    df_bin_detail_info: pl.DataFrame = None,
    num_y_scale_zero: bool = False,
    point_size: int = 50,
    verbose: bool = False,
    color_scale_scheme: str = None,
) -> alt.Chart:
    """
    ビンごとにターゲット列を集約し、線＋点プロットを Altair で描画する関数。

    集約されたターゲット値をビンの中央値またはカテゴリでX軸に、線と点の重ね合わせで視覚化する。
    カラーグループでの比較や、bin情報の中間点指定にも対応。

    パラメータ
    ----------
    df : pl.DataFrame
        入力データ。ビン列・ターゲット列・（オプションで）カラー列を含む。

    col_bin : str
        ビニング列の名前。カテゴリまたは bin 情報に基づく列。

    col_target : str
        Y軸にプロットする対象列。数値型を想定。

    col_color : str, optional
        グループごとに線・点を色分けする列名。None の場合は単一色。

    agg_func : Callable, optional (default=pl.mean)
        ターゲット列の集約関数（例: `pl.mean`, `pl.median`, `pl.max` など）。

    df_bin_detail_info : pl.DataFrame, optional
        ビンの詳細情報を含む DataFrame。列 `{col_bin}_median` が含まれている場合、それをX軸に使用。

    num_y_scale_zero : bool, optional
        Y軸スケールに 0 を含めるかどうか（Altair の `scale.zero` オプション）。

    point_size : int, optional (default=50)
        プロットされる点のサイズ。

    verbose : bool, optional
        処理中の DataFrame を `display()` で表示するデバッグモード。

    color_scale_scheme : str, optional
        Altair のカラースキーム名（例: "category10", "tableau20" など）。指定しない場合はデフォルト。

    戻り値
    -------
    alt.Chart
        線（line）と点（point）を重ねた Altair の複合チャート。

    使用例
    -------
    >>> chart = plot_line_point_over_bin(df, col_bin="age_bin", col_target="score", col_color="gender")
    >>> chart.display()

    備考
    ----
    - `df_bin_detail_info` が指定されていて `{col_bin}_median` が存在する場合、X軸にはその中央値を使う。
    - 色と形は `col_color` によって自動的に変化し、凡例も自動表示される。
    - `resolve_scale` により y 軸は共有され、color/shape は独立。
    """
    col_target_agg = f'{col_target} ({agg_func.__name__})'

    # 集約
    group_keys = [col_bin] + ([col_color] if col_color else [])
    df_agg = df.group_by(group_keys).agg(agg_func(col_target).alias(col_target_agg))

    if df_bin_detail_info is not None:
        df_agg = df_agg.join(df_bin_detail_info, on=col_bin, how="left")

    if verbose:
        print('df_agg:')
        display(df_agg)

    col_bin_median = f"{col_bin}_median"
    col_x = col_bin_median if col_bin_median in df_agg.columns else col_bin

    if verbose:
        print(f"col_x: {col_x}, col_color: {col_color}")

    base = alt.Chart(df_agg)
    enc_x = alt.X(col_x)
    enc_y = alt.Y(f"{col_target_agg}:Q", axis=alt.Axis(orient="right"), scale=alt.Scale(zero=num_y_scale_zero))

    chart_line = base.mark_line().encode(
        x=enc_x, y=enc_y, color=alt.Color(f"{col_color}:N", legend=None, scale=alt.Scale(scheme=color_scale_scheme))
    )
    chart_point = base.mark_point(size=point_size).encode(
        x=enc_x, y=enc_y, 
        color=alt.Color(f"{col_color}:N", scale=alt.Scale(scheme=color_scale_scheme)), 
        shape=alt.Shape(f"{col_color}:N")
    )

    return alt.layer(chart_line, chart_point).resolve_scale(y='shared', color='independent', shape='independent')


def _align_all_columns(*dfs: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    if not dfs:
        raise ValueError("少なくとも1つのDataFrameが必要です")

    # --- 1. 全列の型をチェックして一貫性を確認 ---
    col_types: dict[str, pl.DataType] = {}
    for df in dfs:
        for col, dtype in df.schema.items():
            if col in col_types:
                assert col_types[col] == dtype, f"列 '{col}' の型が一致しません: {col_types[col]} ≠ {dtype}"
            else:
                col_types[col] = dtype

    # --- 2. 全列名を収集（順序：df1の列順＋他にしかない列） ---
    base_cols = list(dfs[0].columns)
    other_cols = [col for df in dfs[1:] for col in df.columns if col not in base_cols]
    final_col_order = base_cols + sorted(set(other_cols), key=other_cols.index)  # 他の列は登場順

    # --- 3. 各DataFrameに足りない列を追加し、順番を揃える ---
    aligned_dfs = []
    for df in dfs:
        missing_cols = [col for col in final_col_order if col not in df.columns]
        df_extended = df.with_columns(
            [pl.lit(None).cast(col_types[col]).alias(col) for col in missing_cols]
        )
        df_aligned = df_extended.select(final_col_order)
        aligned_dfs.append(df_aligned)

    return tuple(aligned_dfs)


def _standardize_columns_by_first_df(
    *dfs: pl.DataFrame,
    col_list: list[str],
    verbose: bool = False
) -> list[pl.DataFrame]:
    """
    各列ごとに、最初に有効値を持つDataFrameを基準として標準化。
    変換後のnp.nanはPolarsのNoneに変換して返す。
    """
    from sklearn.preprocessing import StandardScaler


    if not dfs:
        raise ValueError("少なくとも1つのDataFrameが必要です")

    if verbose:
        print(">>> 標準化対象の列:", col_list)
        print(">>> 入力DataFrameの数:", len(dfs))

    # 各列ごとのStandardScalerを用意
    scalers: dict[str, StandardScaler] = {}
    for col in col_list:
        # 最初にそのcolに「Noneでない」値が含まれるdfを探す
        for df in dfs:
            if col in df.columns:
                # print(f"col: {col}, df.select(pl.col(col).is_not_null().len()).item(): {df.select(pl.col(col).is_not_null().len()).item()}")
                if df.select(pl.col(col).is_not_null().sum()).item() > 0:
                    # 全部NaNの列を除外する
                    df = df.filter(pl.col(col).is_not_null())
                    # print(f"len(df): {len(df)}")
                    scaler = StandardScaler()
                    scaler.fit(df.select(col))
                    scalers[col] = scaler
                    if verbose:
                        print(f">>> '{col}' fit元 df の先頭5行:\n", df.select(col).head())
                    break
        else:
            raise ValueError(f"全てのDataFrameにおいて列 '{col}' に有効な値が存在しません")

    # 各DataFrameに対して標準化
    result_dfs = []
    for i, df in enumerate(dfs):
        df_scaled_cols = []
        for col in col_list:
            if col in df.columns:
                col_vals = df.select(col)
                transformed = scalers[col].transform(col_vals)
                col_df = pl.DataFrame(transformed, schema=[col], orient="row")
                col_df = col_df.fill_nan(None)  # ← NaNをPolarsのNoneに
            else:
                col_df = pl.Series(name=col, values=[None] * df.height).to_frame()
            df_scaled_cols.append(col_df)

        df_scaled = pl.concat(df_scaled_cols, how="horizontal")
        df_rest = df.drop(col_list)
        df_final = df_rest.hstack(df_scaled)

        # 行数チェック
        assert df.shape[0] == df_final.shape[0], f"行数不一致 at df {i}"

        if verbose:
            print(f"\n>>> df[{i}] 標準化後の先頭5行:\n", df_final.head())

        result_dfs.append(df_final)

    return result_dfs


"""
★plot_venn関数
"""
def plot_venn(
    df: pl.DataFrame,
    col_entity: str,
    col_category: str,
    category_order: list[str] | None = ['train', 'test', 'detail'],
    category_colors: dict[str, str] | None = {'train': 'royalblue', 'test': 'indianred', 'detail': 'gold'},
    subtitle_label_fontsize: int = 28,
    category_label_fontsize: int = 24,
    count_label_fontsize: int = 20,
    title: str = None,
    verbose: bool = False,
) -> alt.Chart:
    """
    AltairでVenn図（集合の重なり）を描画する関数。

    与えられた Polars DataFrame を基に、エンティティ（例: ID）がどのカテゴリ（例: train/test/detail）に
    含まれているかを可視化する。実際の描画は matplotlib を用いてVenn図を生成し、それを画像として Altair に埋め込む。

    パラメータ
    ----------
    df : pl.DataFrame
        入力データ。`col_entity` と `col_category` 列を含む必要がある。

    col_entity : str
        集合対象のエンティティ（例: ID）を示す列名。

    col_category : str
        各エンティティが所属するカテゴリ（例: train, test, detail）を示す列名。

    category_order : list[str] or None, optional
        Venn図に描画するカテゴリの順序。最大3カテゴリまで対応。

    category_colors : dict[str, str] or None, optional
        各カテゴリに割り当てる色（matplotlibのカラー名など）。

    subtitle_label_fontsize : int, optional (default=28)
        サブタイトル（重なりの説明）のフォントサイズ。

    category_label_fontsize : int, optional (default=24)
        各カテゴリ名のフォントサイズ。

    count_label_fontsize : int, optional (default=20)
        カウント（重なりの要素数）のフォントサイズ。

    title : str, optional
        グラフのタイトル。指定しない場合は `col_entity` が使用される。

    verbose : bool, optional
        デバッグ出力を有効にする。

    戻り値
    -------
    alt.Chart
        Venn図を埋め込んだ Altair の画像チャート。

    使用例
    -------
    ```python
    df_combined = pl.concat([
        df_train.with_columns(pl.lit("train").alias("src")).select(["PID", "src"]),
        df_test.with_columns(pl.lit("test").alias("src")).select(["PID", "src"]),
        df_detail.with_columns(pl.lit("detail").alias("src")).select(["PID", "src"]),
    ])
    plot_venn(df_combined, col_entity="PID", col_category="src")    
    ```

    使用例2
    -------
    ```python
    def compare_id(
        df_train: pl.DataFrame,
        df_test: pl.DataFrame,
        df_detail: pl.DataFrame,
        col_compare: str,
        subtitle_label_fontsize: int = 18,
        category_label_fontsize: int = 16,
        count_label_fontsize: int = 12,
        width=600,
        height=200,
    ) -> alt.Chart:
        col_label = 'DataFrame'

        # 各DataFrameにDataFrame名を示す列を追加し、PIDとその列だけを選択
        df_train_labeled = df_train.with_columns(pl.lit("train").alias(col_label)).select([col_compare, col_label])
        df_test_labeled = df_test.with_columns(pl.lit("test").alias(col_label)).select([col_compare, col_label])
        df_detail_labeled = df_detail.with_columns(pl.lit("detail").alias(col_label)).select([col_compare, col_label])

        df_combined = pl.concat([df_train_labeled, df_test_labeled, df_detail_labeled], how="vertical")

        venn_chart = plot_venn(
            df=df_combined,
            col_entity=col_compare,
            col_category=col_label,
            subtitle_label_fontsize = subtitle_label_fontsize,
            category_label_fontsize = category_label_fontsize,
            count_label_fontsize = count_label_fontsize,
        ).properties(width=width, height=height)
        return venn_chart
    ```

    備考
    ----
    - matplotlib による描画を Altair に埋め込む特殊処理を行っているため、データとしての直接描画ではない。
    - `_draw_venn_matplotlib_dual()` と `_matplotlib_to_altair()` に依存しており、これらの補助関数が必要。
    - 最大3カテゴリまで対応（matplotlibのVenn制約による）。
    """
    if verbose:
        print(f"category_colors: {category_colors}")

    fig_matplotlib = _draw_venn_matplotlib_dual(
        df,
        col_entity=col_entity,
        col_category=col_category,
        category_order=category_order,
        category_colors=category_colors,
        subtitle_label_fontsize=subtitle_label_fontsize,
        category_label_fontsize=category_label_fontsize,
        count_label_fontsize=count_label_fontsize,
        verbose=verbose,
    )

    chart_altair = _matplotlib_to_altair(
        fig_matplotlib, 
        )
    if title is None:
        title = col_entity
    return chart_altair.properties(title=title)


def _draw_venn_matplotlib_dual(
    df: pl.DataFrame,
    col_entity: str,
    col_category: str,
    subtitle_label_fontsize: int,
    category_label_fontsize,
    count_label_fontsize,
    category_order: list[str] | None = ['train', 'test'],
    category_colors: dict[str, str] | None = None,
    figsize: tuple[int, int] = None, # (6, 10)
    verbose: bool = False,
) -> plt.Figure:
    """
    全値とユニーク値のVenn図を上下に1枚にまとめて描画し、Figureを返す。
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

    _draw_venn_matplotlib(
        df=df,
        col_entity=col_entity,
        col_category=col_category,
        subtitle_label_fontsize=subtitle_label_fontsize,
        category_label_fontsize=category_label_fontsize,
        count_label_fontsize=count_label_fontsize,
        ax=ax1,
        use_unique=False,
        category_order=category_order,
        category_colors=category_colors,
        subtitle="all",
        verbose=verbose,
    )

    _draw_venn_matplotlib(
        df=df,
        col_entity=col_entity,
        col_category=col_category,
        subtitle_label_fontsize=subtitle_label_fontsize,
        category_label_fontsize=category_label_fontsize,
        count_label_fontsize=count_label_fontsize,
        ax=ax2,
        use_unique=True,
        category_order=category_order,
        category_colors=category_colors,
        subtitle="unique",
        verbose=verbose,
    )

    return fig


def _draw_venn_matplotlib(
    df: pl.DataFrame,
    col_entity: str,
    col_category: str | None,
    subtitle_label_fontsize: int,
    category_label_fontsize: int,
    count_label_fontsize: int,
    ax: plt.Axes,
    use_unique: bool = True,
    category_order: list[str] | None = ['train', 'test'],
    category_colors: dict[str, str] | None = None,
    subtitle: str = None,
    verbose: bool = False,
    offset_shared_area: float = 0.15,
) -> None:
    """
    指定したaxにVenn図を描画する関数。
    use_unique=Trueならユニーク値集合。Falseなら多重集合。
    subtitle が指定されていれば、ax.set_title でサブタイトルを付ける。
    """
    from matplotlib_venn import venn2, venn3
    from collections import Counter


    # ✅ col_category が None の場合はダミーカラムを作成
    # if (col_category is None) or (col_category not in df.columns):
    # カラムが存在しない or None or 全部 null の場合は dummy モードに
    if (
        col_category is None or
        col_category not in df.columns or
        df.select(pl.col(col_category).is_not_null().sum()).item() == 0
        ):
        dummy_col = "__dummy_category__"
        df = df.with_columns([
            pl.lit("").alias(dummy_col)
        ])
        col_category = dummy_col
        is_dummy_category = True  # ← フラグを立てる
    else:
        is_dummy_category = False
        
    all_categories = df.select(pl.col(col_category)).unique().to_series().to_list()

    # category_order に基づき選択
    if category_order is not None:
        ordered = [cat for cat in category_order if cat in all_categories]
        remaining = sorted(set(all_categories) - set(ordered))
        selected_categories = tuple(ordered + remaining)
    else:
        selected_categories = tuple(sorted(all_categories))

    if len(selected_categories) > 3:
        raise ValueError(f"カテゴリが3種類を超えています: {selected_categories}")

    sets = []
    for cat in selected_categories:
        values = df.filter(pl.col(col_category) == cat).select(col_entity).to_series().to_list()
        sets.append(set(values) if use_unique else Counter(values))

    while len(sets) < 3:
        sets.append(set() if use_unique else Counter())
        selected_categories += ("",)

    # 描画
    non_empty_count = sum(bool(s) for s in sets)
    if non_empty_count == 1:
        empty = Counter() if not use_unique else set()  # ← 型を合わせる
        venn = venn2(subsets=(sets[0], empty), set_labels=selected_categories[:2], ax=ax)
        venn_type = 2
    elif non_empty_count == 2:
        venn = venn2(subsets=sets[:2], set_labels=selected_categories[:2], ax=ax)
        venn_type = 2
    else:
        venn = venn3(subsets=sets, set_labels=selected_categories, ax=ax)
        venn_type = 3

    for label in venn.set_labels:
        if label:
            label.set_fontsize(category_label_fontsize)

    # fontsizeを調整する
    # 1つの円を描くためにvenn2を使ってる場合、ダミーの円の値である0が表示されてしまうので空文字にして消す
    for id_, idx in venn.id2idx.items():
        if idx < len(venn.subset_labels):
            label = venn.subset_labels[idx]
            if label:
                if (
                    label.get_text() == "0"
                    and is_dummy_category
                    and id_ == '010'
                ):
                    label.set_text("")
                else:
                    label.set_fontsize(count_label_fontsize)
    
    # 共通部分の値テキストが重なってしまうことが多いので上にずらす
    # 2領域の重なりは1段階、3領域の重なりは2段階上にずらす
    if venn_type == 3:
        overlap_ids = ['110', '101', '011', '111']
    elif venn_type == 2:
        overlap_ids = ['11']  # 2集合で重なるのは1つだけ

    for vid in overlap_ids:
        try:
            label = venn.get_label_by_id(vid)
        except IndexError:
            label = None

        if label is not None:
            x, y = label.get_position()
            offset = offset_shared_area * 2 if vid == '111' else offset_shared_area
            label.set_position((x, y + offset))

    # ダミーカテゴリで1つの円が描かれた場合、色はグレーにする
    if category_colors or is_dummy_category:
        id_map = ['100', '010', '001']
        for id_, cat in zip(id_map, selected_categories):
            idx = venn.id2idx.get(id_)
            if idx is not None and idx < len(venn.patches):
                patch = venn.patches[idx]
                if patch:
                    if is_dummy_category:
                        patch.set_color("lightgray")
                    elif category_colors and cat in category_colors:
                        patch.set_color(category_colors[cat])

    # サブタイトル
    if subtitle:
        ax.set_title(subtitle, fontsize=subtitle_label_fontsize,
        )


def _matplotlib_to_altair(
    fig: plt.Figure,
) -> alt.Chart:
    """
    MatplotlibのFigureをAltairの画像チャートに変換する。
    """
    import base64
    from io import BytesIO


    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    image_url = "data:image/png;base64," + encoded

    return alt.Chart().mark_image(
        url=image_url,
    )


"""
★profile関数
"""
def profile(
        *dfs,
        col_target=None,
        num_n_bins=10,
        width_chart=200,
        height_chart=200,
        str_col_bin_unique_limit: int = 100,
        standardize_line=True,
        normalize_histogram=True,
        tabulate_dfs_color: list[str] = ['lightblue', 'lightpink'],
        verbose=False,
        width_panel: int = 350,
        table_compact_fn=None,
        render_panel_fn=None,
    ):
    """
    複数のDataFrameを対象に、全列に対するプロファイリング可視化とサマリ表を表示する関数。
    各列に対して、Altairチャート＋great_tables表を横並びに表示する。

    各列に対してヒストグラムまたは折れ線グラフを描画し、比較可能な形式で表示。
    数値・カテゴリ・文字列などの型に応じて適切なビニングや標準化・正規化処理を行う。
    また、各DataFrameに対する `describe_ex()` の結果を色分けして表形式で表示。

    パラメータ
    ----------
    *dfs : DataFrame(s)
        プロファイリング対象となる Polars データフレーム（複数可）。

    col_target : str, optional
        折れ線グラフとしてプロットする際のターゲット列（例：目的変数）。None の場合は省略。

    num_n_bins : int, optional (default=10)
        数値列のビン数。ビニングは `_draw_profile_graph()` 内部で自動処理される。

    width_chart : int, optional
        各チャートの横幅（ピクセル単位）。

    height_chart : int, optional
        各チャートの高さ（ピクセル単位）。

    columns_concat_chart : int, optional
        表示時に1行に並べるチャートの数（Altairの `alt.concat(..., columns=...)` に対応）。

    str_col_bin_unique_limit : int, optional (default=100)
        文字列・カテゴリ列で表示するユニーク値の最大数。超えるとスキップされる。

    standardize_line : bool, optional (default=True)
        折れ線グラフを標準化（平均0, 分散1）して比較するかどうか。

    normalize_histogram : bool, optional (default=True)
        ヒストグラムを正規化（相対頻度）するかどうか。

    tabulate_dfs_color : list[str], optional
        `describe_ex()` の出力に対して色を付けるためのリスト。DataFrameの順番に対応。

    verbose : bool, optional
        デバッグ出力を有効にする。

    戻り値
    -------
    None
        Altair チャートおよび HTML テーブルが notebook 上で `display()` によって表示される。

    使用例
    -------
    >>> profile(df_train, df_test, col_target="target")

    備考
    ----
    - チャートは Altair を使用しており、Jupyter Notebook / JupyterLab 上での可視化を想定。
    - `_draw_profile_graph()` や `_draw_profile_table()` などの補助関数に依存。
    - `describe_ex()` を使って拡張サマリを生成しており、型や欠損、最頻値なども表で確認可能。
    """
    from IPython.display import display
    from tqdm import tqdm

    columns = _get_ordered_unique_columns(dfs)

    # デフォルトのパネル描画関数（chart + GT を横並びに表示）
    if render_panel_fn is None:
        def render_panel_fn(chart, gt):
            import io
            import base64
            from IPython.display import HTML

            buf = io.BytesIO()
            chart.save(fp=buf, format="png")
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
            # GTの上部マージンに合わせて少しマージンを取る
            chart_img_html = f'<img src="data:image/png;base64,{chart_base64}" style="max-width: 100%; margin-top: 10px;">'

            html = f"""
            <div style="display: flex; justify-content: flex-start; gap: 20px; margin-bottom: 0px; align-items: flex-start;">
                <div style="min-width: {width_panel}px; max-width: 500px; overflow-x: auto;">
                    {gt._repr_html_()}
                </div>
                <div style="width: {width_panel}px; text-align: left;">
                    {chart_img_html}
                </div>
            </div>
            """

            display(HTML(html))

    # 各列ごとにグラフ＋テーブルを作って表示
    # pbar = tqdm(columns, desc="Profiling Columns", leave=False)
    # for col in pbar:
    for i, col in enumerate(columns, 1):
        # pbar.set_description(f"Processing... (col: {col})")
        # display(Markdown(f"### 📊 [{i}/{len(columns)}] {col}"))
        icon = _get_dtype_icon(dfs[0], col)
        display(Markdown(f"### {icon} `{col}` _(Column {i} of {len(columns)})_"))

        # チャート作成
        chart = _draw_profile_graph(
            *dfs, col=col, col_target=col_target, num_n_bins=num_n_bins,
            str_col_bin_unique_limit=str_col_bin_unique_limit,
            standardize_line=standardize_line,
            normalize_histogram=normalize_histogram,
            verbose=verbose
        )

        if chart is None:
            continue

        # describeテーブル作成
        # dfs_describe = [df.select(col).describe_ex() for df in dfs]
        dfs_describe = []
        for df in dfs:
            if col in df.columns:
                dfs_describe.append(df.select(col).describe_ex())
        table = _draw_profile_table(*dfs_describe, dfs_color=tabulate_dfs_color)

        # オプションで compact 表示適用
        if table_compact_fn is not None:
            table = table_compact_fn(table)

        # 表とグラフを横並びで表示
        render_panel_fn(chart.properties(width=width_chart, height=height_chart), table)


def _get_dtype_icon(df: pl.DataFrame, col: str) -> str:
    dtype = df.schema[col]
    if dtype == pl.Int64 or dtype == pl.Int32 or dtype == pl.Float64 or dtype == pl.Float32:
        return "🔟"  # 数値
    elif dtype == pl.Utf8:
        return "🆎"  # 文字列
    elif dtype == pl.Boolean:
        return "✅"  # 真偽値
    elif dtype == pl.Datetime:
        return "🕒"  # 日時
    elif dtype == pl.Date:
        return "📅"  # 日時
    elif dtype == pl.Timedelta:
        return "⌛"  # 日時
    else:
        return "❓"  # 不明型


def _draw_profile_graph(
    *dfs:pl.DataFrame,
    col:str,
    col_target: Optional[Union[str, list[str]]] = None, # Union[str, list[str]],
    dfs_name:list[str] = ['train', 'test'],
    dfs_color_histogram:list[str] = ['royalblue', 'indianred'],
    dfs_color_line:list[str] = ['blues', 'reds'],
    col_dataframe_name = 'DataFrame',
    col_target_unpivot_name: str = 'column',
    col_target_unpivot_value: str = 'target',
    normalize_histogram:bool = True,
    standardize_line:bool = True,
    num_n_bins: int = 10,
    dt_truncate_unit: str = "1mo",
    str_col_bin_unique_limit:int = 100,
    verbose:bool=False,
):
    """
    - profile関数(グラフ＋表を全列分表示)のグラフ描画処理
    - plot_histogram, plot_line_point_over_bin, plot_venn関数などを使う
    """
    from typing import Union, Optional


    assert len(dfs_name) >= len(dfs), f"dfs_nameの要素数が足りません (必要数: {len(dfs)})"
    assert len(dfs_color_histogram) >= len(dfs), f"dfs_color_histogramの要素数が足りません (必要数: {len(dfs)})"
    assert len(dfs_color_line) >= len(dfs), f"dfs_color_lineの要素数が足りません (必要数: {len(dfs)})"

    # dfsが単品かつdfsがメンテされてなさそうな場合、DataFrameの名前を消す
    if len(dfs) == 1 and len(dfs_name) > 1:
        dfs_name = [None] 

    # col_targetはstr, list[str]どっちでもいけるようにする(内部ではlistで統一)
    # col_target_list = [col_target] if isinstance(col_target, str) else col_target
    if col_target is None:
        col_target_list = []
    elif isinstance(col_target, str):
        col_target_list = [col_target]
    else:
        col_target_list = col_target

    # 列名: DataFrame, 値: train, testみたいな列を追加する(色分け用)
    dfs = [
        df.with_columns(pl.lit(name).alias(col_dataframe_name))
        for df, name in zip(dfs, dfs_name)
    ]

    # 列を揃える(ない場合は値がすべてNullの列となる)
    dfs_aligned = _align_all_columns(*dfs)

    # bin列を作る(未結合)
    *dfs_bin, df_bin_detail_info = get_bin_column(
        *dfs_aligned, col=col, 
        num_n_bins=num_n_bins, dt_truncate_unit=dt_truncate_unit, verbose=False
    )
    col_bin = dfs_bin[0].to_series().name

    # ↓これは呼び出す側にもっていくかも★

    # カテゴリ列のクラスが制限以上なら描画しない(そのbin列はすべてNullとする→描画されない)
    # ユニーク数を確認
    n_uniques = [
        df_bin.select(pl.col(col_bin).n_unique()).item()
        for df_bin in dfs_bin
    ]
    # bin数の最大値を取る
    n_unique_max = max(n_uniques)
    # チェックNGフラグ
    col_bin_check_ng = n_unique_max > str_col_bin_unique_limit


    # NG(クラス数多すぎ)ならスキップ ⇒ ベン図
    if col_bin_check_ng:
        # print(f"列: {col} のbin数が多すぎるためスキップします (bin数: {n_unique_max}, 上限: {str_col_bin_unique_limit})")
        # return None
        if verbose:
            print(f"列: {col} のbin数が多すぎるためベン図を描画します (bin数: {n_unique_max}, 上限: {str_col_bin_unique_limit})")
        
        # 結合してplot_vennに渡す
        dfs_labeled = [
            df.select([col, col_dataframe_name])
            for df in dfs
        ]
        df_combined = pl.concat(dfs_labeled, how="vertical")
        category_colors = dict(zip(dfs_name, dfs_color_histogram))
        venn_chart = plot_venn(
            df=df_combined,
            col_entity=col,
            col_category=col_dataframe_name,
            # width=width,
            # height=height,
            category_colors=category_colors,
            title=f"{col}",
            verbose=verbose,
        )
        return venn_chart


    # # bin列を結合する
    if col_bin != col:
        dfs_with_bin = [
            df_aligned.hstack(df_bin)
            for df_aligned, df_bin in zip(dfs_aligned, dfs_bin)
        ]
    else:
        dfs_with_bin = dfs_aligned

    # 標準化する(オプション)
    if standardize_line and col_target_list:
        dfs_with_bin = _standardize_columns_by_first_df(
            *dfs_with_bin, col_list=col_target_list
        )

    # ---- サブ関数（ループの前に置く） ----
    # domain = dfs_name
    # range = dfs_color_histogram
    def create_histogram_chart(df_with_bin: pl.DataFrame):
        # color_scale_bar = alt.Scale(domain=['train', 'test'], range=['royalblue', 'indianred'])
        color_scale_bar = alt.Scale(domain=dfs_name, range=dfs_color_histogram)

        chart_histogram = plot_histogram(
            df_with_bin,
            col_bin=col_bin,
            col_color=col_dataframe_name,
            df_bin_detail_info=df_bin_detail_info,
            normalize_histogram=normalize_histogram,
            verbose=False
        )
        is_all_name_missing = df_with_bin.select(pl.col(col_dataframe_name).is_null().all()).item()
        if is_all_name_missing:
            legend = None
        else:
            legend = alt.Legend(title=f'{col_dataframe_name}')
        chart_histogram = chart_histogram.encode(
            color=alt.Color(
                f"{col_dataframe_name}:N",
                scale=color_scale_bar,
                legend=legend
            )
        )
        return chart_histogram#, train_or_test

    name_to_color_line = dict(zip(dfs_name, dfs_color_line))
    def create_line_point_chart(df_with_bin: pl.DataFrame):
        dataframe_name = df_with_bin.select(col_dataframe_name).unique().item()
        # color_scale_scheme_line = 'blues' if dataframe_name == 'train' else 'reds'
        color_scale_scheme_line = name_to_color_line[dataframe_name]

        # target列が複数でもいけるように、unpivot(melt)してロングフォーマットに直す
        df_unpivot_target = df_with_bin.unpivot(
            on=col_target_list,
            index=col_bin,
            variable_name=col_target_unpivot_name,
            value_name=col_target_unpivot_value
        )

        chart_line_point = plot_line_point_over_bin(
            df_unpivot_target,
            col_bin=col_bin,
            col_target=col_target_unpivot_value,
            col_color=col_target_unpivot_name,
            df_bin_detail_info=df_bin_detail_info,
            color_scale_scheme=color_scale_scheme_line,
            verbose=False
        )

        # 折れ線グラフが1つもない場合、折れ線用のレジェンドタイトル(凡例のグループ名)を表示しない
        is_all_target_missing = df_unpivot_target.select(pl.col(col_target_unpivot_value).is_null().all()).item()
        if is_all_target_missing:
            legend = None
        else:
            legend = alt.Legend(title=f"target{f' ({dataframe_name})' if dataframe_name else ''}")

        chart_line_point = chart_line_point.encode(
            color=alt.Color(legend=legend)
        )
        return chart_line_point

    # ---- ここからループ本体 ----
    chart_histogram_list = []
    chart_line_point_list = []

    for df_with_bin in dfs_with_bin:
        is_all_histogram_bin_missing = df_with_bin.select(pl.col(col_bin).is_null().all()).item()
        if is_all_histogram_bin_missing:
            continue

        chart_histogram = create_histogram_chart(df_with_bin)
        chart_histogram_list.append(chart_histogram)

        if col_target_list:
            chart_line_point = create_line_point_chart(df_with_bin)
            chart_line_point_list.append(chart_line_point)

    # ---- 最後にまとめる ----
    chart_histogram = alt.layer(*chart_histogram_list)

    if col_target_list:
        chart_line_point = alt.layer(*chart_line_point_list)
        chart = alt.layer(chart_histogram, chart_line_point).resolve_scale(
            y='independent', color='independent', shape='independent'
        )
    else:
        chart = chart_histogram

    return chart


def _draw_profile_table(
    *dfs: pl.DataFrame,
    dfs_name: list[str] = ['train', 'test'],
    dfs_color: list[str] = ['lightblue', 'lightpink'],
    use_dark_theme: bool = True,
    use_compact_style: bool = True,
    label_columns: list[str] | None = ['statistic'],
    sig_digits: int | None = 3,
    show_dtype_row: bool = False  # ← 追加
):
    """
    複数の Polars DataFrame を比較しやすい表形式に整形して返す。
    型情報行と視覚的なスタイルオプション付き。
    """
    from IPython.display import HTML
    from great_tables import GT, style, loc, px


    num_dfs = len(dfs)
    assert num_dfs >= 1, "1つ以上のDataFrameが必要です"

    if dfs_name is None:
        dfs_name = [f"df{i+1}" for i in range(num_dfs)]
    if dfs_color is None:
        dfs_color = ["lightgray"] * num_dfs

    dfs_name = dfs_name[:num_dfs]
    dfs_color = dfs_color[:num_dfs]
    label_columns = label_columns or []

    # ラベル列の処理
    if label_columns:
        for col in label_columns:
            for df in dfs:
                assert col in df.columns, f"{col} が全てのDataFrameに必要です"
        label_df = dfs[0].select(label_columns)
        dfs_raw = tuple(df.drop(label_columns) for df in dfs)
    else:
        label_df = None
        dfs_raw = dfs

    schema_info = [df.schema for df in dfs_raw]
    label_schema = dfs[0].schema if label_columns else {}

    # sig_digits の適用
    if sig_digits is not None:
        dfs = tuple(
            df.with_columns([
                pl.col(col).map_elements(lambda x: _format_sig(x, sig_digits), return_dtype=pl.String).alias(col)
                for col in df.columns
            ]) for df in dfs_raw
        )
    else:
        dfs = dfs_raw

    # カラム名に df名 を追加
    dfs_named = []
    column_to_color, column_to_df, column_to_full = {}, {}, {}

    for i, (df, name, color) in enumerate(zip(dfs, dfs_name, dfs_color)):
        renamed_cols = {}
        for col in df.columns:
            new_col = col if num_dfs == 1 else f"{col} ({name})"
            renamed_cols[col] = new_col
            column_to_color[new_col] = color
            column_to_df.setdefault(col, []).append(i)
            column_to_full.setdefault(col, []).append(new_col)
        dfs_named.append(df.rename(renamed_cols))

    df_combined = pl.concat(dfs_named, how="horizontal")

    if label_df is not None:
        df_combined = label_df.hstack(df_combined)
        if num_dfs > 1:
            df_combined = _reorder_columns_by_df_map(df_combined, label_columns, column_to_full, column_to_df)
    else:
        if num_dfs > 1:
            df_combined = _reorder_columns_by_df_map(df_combined, [], column_to_full, column_to_df)

    # 型情報の1行目を追加
    if show_dtype_row:
        dtype_row = {}
        for schema, name in zip(schema_info, dfs_name):
            for col in schema:
                full_col = col if num_dfs == 1 else f"{col} ({name})"
                dtype_row[full_col] = str(schema[col])
        for col in label_columns:
            dtype_row[col] = str(label_schema.get(col, ""))

        dtype_row_filled = {col: dtype_row.get(col, "") for col in df_combined.columns}
        df_combined = pl.concat([pl.DataFrame([dtype_row_filled]), df_combined], how="vertical")

    # GreatTables スタイル適用
    table = GT(df_combined)
    if use_dark_theme:
        table = table.tab_options(
            table_background_color="#1e1e1e",
            heading_background_color="#2e2e2e",
            row_group_background_color="#2e2e2e",
            table_border_top_color="#444444",
            table_border_bottom_color="#444444"
        )
    if use_compact_style:
        table = table.tab_options(
            data_row_padding=px(2),
            row_group_padding=px(2),
            heading_title_font_size="small",
            heading_subtitle_font_size="small",
            table_font_size="small"
        )
        table = table.opt_vertical_padding(scale=0.5)
        table = table.opt_horizontal_padding(scale=0.7)

    for col in df_combined.columns:
        if col in label_columns:
            continue
        color = column_to_color.get(col)
        if color:
            table = table.tab_style(style=style.text(color=color), locations=loc.body(columns=col))
            table = table.tab_style(style=style.text(color=color), locations=loc.column_labels(columns=col))

    if show_dtype_row:
        table = table.tab_style(style=style.borders(sides="top", color="#888", style="solid", weight="2px"), locations=loc.body(rows=[0]))
        table = table.tab_style(style=style.borders(sides="bottom", color="#888", style="solid", weight="2px"), locations=loc.body(rows=[0]))
        table = table.tab_style(style=style.text(color="#bbb", weight="bold"), locations=loc.body(rows=[0]))

    # return table
    # 他と合わせるため左寄せにする
    table_html = HTML(f"""
        <div style="text-align: left; display: inline-block;">
        {table._repr_html_()}
        </div>
        """)
    return table_html


def _reorder_columns_by_df_map(
    df: pl.DataFrame,
    label_columns: list[str],
    column_to_full: dict[str, list[str]],
    column_to_df: dict[str, list[int]]
) -> pl.DataFrame:
    """
    各 base列（括弧の前）を、dfの順に揃えて交互に並べる。
    label_columns はそのまま先頭に維持。

    例：
        df1: x, y   df2: x  → x (train), x (test), y (train)
    """
    base_order = [base for base in column_to_full if any(col in df.columns for col in column_to_full[base])]
    ordered_cols = label_columns[:]
    for base in base_order:
        for idx in column_to_df[base]:
            if idx < len(column_to_full[base]):
                col = column_to_full[base][idx]
                if col in df.columns:
                    ordered_cols.append(col)
    return df.select(ordered_cols)


def _get_ordered_unique_columns(dfs: Sequence[pl.DataFrame]) -> list[str]:
    seen = set()
    ordered_cols = []
    for df in dfs:
        for col in df.columns:
            if col not in seen:
                seen.add(col)
                ordered_cols.append(col)
    return ordered_cols


"""
__all__ を動的に生成（このモジュール内の関数だけを対象にする）
"""
__all__ = [
    name for name, val in globals().items()
    if (
        (isinstance(val, types.FunctionType) or isinstance(val, type))  # 関数 or クラス
        and 
        val.__module__ == __name__  # このモジュール内のものに限定
    )
]
from IPython.display import display, Markdown

import polars as pl
import altair as alt

from .polars_binning import get_col_bin_auto


alt.themes.enable('dark')


def plot_histogram_line_after_binning(
    *dfs: pl.DataFrame,
    col: str,
    col_target: str | list[str] = "sales",

    # binning
    num_n_bins: int = 10,
    num_sig_digits: int = 3,
    dt_truncate_unit: str = "1mo",

    # histogram
    col_color_hist: str = None, # 優先
    color_hist: list[str] = ['royalblue', 'indianred'],
    bar_opacity_hist: float = 0.5,
    num_x_scale_zero_hist: bool = False,
    normalize_hist: bool = False,

    # line
    color_line: list[str] = ['gold', 'orange'], # col_colorが指定されてない時に使われる固定色
    agg_func_line = pl.mean,
    col_color_line: str = None,
    color_scale_scheme_line: str = 'blues', # 'reds' # col_colorが指定された場合の色の系統
    num_y_scale_zero_line: bool = False,
    line_size: int = 1,
    line_opacity: float = 0.7,
    point_size: int = 5,

    # degug
    verbose: int = 0,
) -> alt.Chart:

    # bin処理（共通のbinを作る）
    *dfs_binned, df_bin_detail_info = get_col_bin_auto(
        *dfs, 
        col=col,
        num_n_bins=num_n_bins,
        num_sig_digits=num_sig_digits,
        dt_truncate_unit=dt_truncate_unit,
        verbose=verbose,
        )
    col_bin = f"{col}_bin"

    # binnedを適用した DataFrame を取得
    dfs = [df.with_columns(df_binned) for df, df_binned in zip(dfs, dfs_binned)]

    # ヒストグラムチャート群
    hist_charts = []
    for i, df in enumerate(dfs):
        color_hist_item = color_hist[i] if color_hist is not None and i < len(color_hist) else None
        chart = _plot_histogram_over_bin(
            df,
            col_bin=col_bin,
            df_bin_detail_info=df_bin_detail_info,
            color=color_hist_item,
            bar_opacity=bar_opacity_hist,
            num_x_scale_zero=num_x_scale_zero_hist,
            normalize_histogram=normalize_hist,    
        )
        if verbose >= 2:
            display(chart)
        hist_charts.append(chart)

    # ヒストグラムをオーバーレイ
    chart_hist_overlay = alt.layer(*hist_charts).resolve_scale(
        y='shared', color='shared'
    )
    if verbose >= 2:
        display(chart_hist_overlay)

    line_charts = []
    for i, df in enumerate(dfs):
        if col_target not in df.columns:
            continue
        color_line_item = color_line[i] if color_line is not None and i < len(color_line) else None
        chart = _plot_line_over_bin(
            df,
            col_bin=col_bin,
            col_target=col_target,
            col_color=col_color_line,
            color_scale_scheme=color_scale_scheme_line, # 'reds' # col_colorが指定された場合の色の系統
            color=color_line_item,
            df_bin_detail_info=df_bin_detail_info,
            agg_func=agg_func_line,
            num_y_scale_zero=num_y_scale_zero_line,
            line_size=line_size,
            line_opacity=line_opacity,
            point_size=point_size,
        )
        if verbose >= 2:
            display(chart)
        line_charts.append(chart)
    chart_line_overlay = alt.layer(*line_charts).resolve_scale(
        y='shared', color='shared'
    )
    if verbose >= 2:
        display(chart_line_overlay)

    # 総合チャート（ヒストグラム＋折れ線群）
    chart = alt.layer(chart_hist_overlay, chart_line_overlay).resolve_scale(
        y='independent', color='independent'
    )
    return chart


def plot_histogram_after_binning(
        
) -> alt.Chart:
    pass


def plot_line_after_binning(
        
) -> alt.Chart:
    pass


"""
★plot_histogram関数
- ↑のbinning関数を内部で使って、それを使って集計したものをHistogramにする処理
"""
def _plot_histogram_over_bin(
    df: pl.DataFrame,
    col_bin: str,
    df_bin_detail_info: pl.DataFrame = None,
    col_color: str = None, # 優先
    color: str = 'royalblue',  # col_colorが指定されてない場合に使われる固定色
    bar_opacity: float = 0.5,
    num_x_scale_zero: bool = False,
    normalize_histogram: bool = False,
    title: str = None,
    verbose: int = 0,
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

    # mark_bar 後の色指定
    if col_color:
        chart = chart.encode(color=alt.Color(f"{col_color}:N"))
    elif color:
        chart = chart.encode(color=alt.value(color))

    return chart


"""
★plot_line_point_over_bin関数
"""
def _plot_line_over_bin(
    df: pl.DataFrame,
    col_bin: str,
    col_target: str,
    df_bin_detail_info: pl.DataFrame = None,
    agg_func = pl.mean,
    col_color: str = None,
    color_scale_scheme: str = 'blues', # 'reds' # col_colorが指定された場合の色の系統
    color: str = 'gold', # col_colorが指定されてない時に使われる固定色
    line_size: int = 1,
    line_opacity: float = 0.7,
    point_size: int = 5,
    num_y_scale_zero: bool = False,
    verbose: int = 0,
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

    if col_color:
        enc_color = alt.Color(f"{col_color}:N", legend=None, scale=alt.Scale(scheme=color_scale_scheme))
        enc_shape = alt.Shape(f"{col_color}:N")
    elif color:
        enc_color = alt.value(color)
        enc_shape = alt.value("circle")  # shapeも固定（任意）
    else:
        enc_color = alt.Undefined
        enc_shape = alt.Undefined

    chart_line = base.mark_line(size=line_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y, color=enc_color
    )

    chart_point = base.mark_point(size=point_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y, color=enc_color, shape=enc_shape
    )
    return alt.layer(chart_line, chart_point).resolve_scale(y='shared', color='independent', shape='independent')

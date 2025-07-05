from IPython.display import display, Markdown

import polars as pl
import altair as alt
from typing import Literal

from .polars_binning import get_col_bin_auto


alt.themes.enable('dark')


def plot_histogram_line_after_binning(
    *dfs: pl.DataFrame,
    col: str,
    col_target: str | list[str] = "sales",
    col_color_scale_domain: list[str] = ['train', 'test'], # map元(値)
    chart_title: str = None,

    # binning - numeric
    num_n_bins: int = 10,
    num_sig_digits: int = 3,
    # binning - datetime
    dt_truncate_unit: str = "1mo",

    # histogram
    col_color_hist: str = 'data_name', # 優先
    col_color_scale_mode_hist: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_hist: list[str] = ['royalblue', 'indianred'], # map先(色)
    col_color_scale_scheme_hist: str = 'category10',
    col_color_legend_title_hist: str = 'data',
    bar_opacity_hist: float = 0.5,
    normalize_hist: bool = False,
    # histogram - numeric
    num_x_scale_zero_hist: bool = False,

    # line
    data_color_line: list[str] = ['gold', 'orange'], # col_colorが指定されてない時に使われる固定色
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
    # hist_charts = []
    # scaleがレジェンドとして一式吐かれるっぽいので、先頭のやつだけ出してまとめるのがいいかも。(2回目からは出さない)
    # その場合、dfに対象列があるかどうかのチェックは必須
    # つかそこまで頑張るならループよりもconcatしてからのほうが楽な気もする。その場合col_colorを自然に使えるし
    # いやその場合normalizeとかめんどいかも？少なくとも前もっての処理がめんどくはなりそう。
    # groupbyを丁寧にやりゃいいだけか？⇒と思ったら既に対応済みで草。concatが楽かも…処理はわかりにくいけどコードはスッキリしそう
    # 必要な列だけ select + data_name 列を追加（if必要）
    

    # # グラフ描画で必要な列
    # cols_needed = [col_bin]
    # if col_color_hist and col_color_hist not in cols_needed:
    #     cols_needed.append(col_color_hist)
    # if col_target and col_target not in cols_needed:
    #     cols_needed.append(col_target)

    # # グラフ描画で必要な列だけをselect
    # # col_color_histを持っているDataFrameが1つもない場合、col_color_histを外付けする(複数dfの場合などでdf自体への色付け指定と見做す)
    # any_df_has_col_color_hist = any(col_color_hist in df.columns for df in dfs)
    # dfs_selected = [
    #     df.select([col for col in cols_needed if col in df.columns]).with_columns(pl.lit(col_color_scale_domain[i]).alias(col_color_hist)) if not any_df_has_col_color_hist else
    #     df.select([col for col in cols_needed if col in df.columns])
    #     for i, df in enumerate(dfs)
    # ]

    # # 結合
    # df_concat = pl.concat(dfs_selected)
    # if verbose:   
    #     display(df_concat)

    # # 色のスケール(変換ルール)を作成する
    # if col_color_scale_mode_hist == 'domain_range':
    #     col_color_scale_hist=alt.Scale(domain=col_color_scale_domain, range=col_color_scale_range_hist)
    # else:
    #     col_color_scale_hist=alt.Scale(scheme=col_color_scale_scheme_hist) 

    # for i, df in enumerate(dfs):
        # color_hist_item = color_hist[i] if color_hist is not None and i < len(color_hist) else None
        # data_name_item = data_name[i] if data_name is not None and i < len(data_name) else None
    chart = _plot_histogram_over_bin(
        *dfs,
        col_bin=col_bin,
        # data_name=data_name_item,
        df_bin_detail_info=df_bin_detail_info,
        chart_title=chart_title,
        col_color_scale_domain=col_color_scale_domain,
        # color=color_hist_item,
        col_color_hist=col_color_hist,
        col_color_scale_mode_hist=col_color_scale_mode_hist,
        col_color_scale_range_hist=col_color_scale_range_hist, # map先(色)
        col_color_scale_scheme_hist=col_color_scale_scheme_hist,
        col_color_legend_title_hist=col_color_legend_title_hist,
        bar_opacity_hist=bar_opacity_hist,
        normalize_hist=normalize_hist,
        num_x_scale_zero_hist=num_x_scale_zero_hist,
        verbose=verbose,
    )
    if verbose >= 2:
        display(chart)
        # hist_charts.append(chart)

    # # ヒストグラムをオーバーレイ
    # chart_hist_overlay = alt.layer(*hist_charts).resolve_scale(
    #     y='shared', color='independent'
    # )
    # if verbose >= 2:
    #     display(chart_hist_overlay)

    # line_charts = []
    # for i, df in enumerate(dfs):
    #     if col_target not in df.columns:
    #         continue
    #     color_line_item = data_color_line[i] if data_color_line is not None and i < len(data_color_line) else None
    #     data_name_item = data_name[i] if data_name is not None and i < len(data_name) else None
    #     print(data_name_item)
    #     chart = _plot_line_over_bin(
    #         df,
    #         col_bin=col_bin,
    #         col_target=col_target,
    #         data_name=data_name_item,
    #         col_color=col_color_line,
    #         color_scale_scheme=color_scale_scheme_line, # 'reds' # col_colorが指定された場合の色の系統
    #         color=color_line_item,
    #         df_bin_detail_info=df_bin_detail_info,
    #         agg_func=agg_func_line,
    #         num_y_scale_zero=num_y_scale_zero_line,
    #         line_size=line_size,
    #         line_opacity=line_opacity,
    #         point_size=point_size,
    #     )
    #     if verbose >= 2:
    #         display(chart)
    #     line_charts.append(chart)
    # chart_line_overlay = alt.layer(*line_charts).resolve_scale(
    #     y='shared', color='independent'
    # )
    # if verbose >= 2:
    #     display(chart_line_overlay)

    # # 総合チャート（ヒストグラム＋折れ線群）
    # chart = alt.layer(chart_hist_overlay, chart_line_overlay).resolve_scale(
    #     y='independent', color='independent'
    # )
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
    *dfs: pl.DataFrame,
    col_bin: str,
    df_bin_detail_info: pl.DataFrame = None,
    col_color_scale_domain: list[str] = ['train', 'test'], # map元(値)
    chart_title: str = None,

    # histogram
    col_color_hist: str = 'data_name', # 優先
    col_color_scale_mode_hist: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_hist: list[str] = ['royalblue', 'indianred'], # map先(色)
    col_color_scale_scheme_hist: str = 'category10',
    col_color_legend_title_hist: str = 'data',
    bar_opacity_hist: float = 0.5,
    normalize_hist: bool = False,
    # histogram - numeric
    num_x_scale_zero_hist: bool = False,

    verbose: int = 0,
) -> alt.Chart:

    if chart_title is None:
        if col_bin.endswith("_bin"):
            chart_title = col_bin.removesuffix("_bin")
        else:
            chart_title = col_bin
    
    # 色のスケール(変換ルール)を作成する
    if col_color_scale_mode_hist == 'domain_range':
        col_color_scale_hist=alt.Scale(domain=col_color_scale_domain, range=col_color_scale_range_hist)
    else:
        col_color_scale_hist=alt.Scale(scheme=col_color_scale_scheme_hist) 

    # グラフ描画で必要な列
    cols_needed = [col_bin]
    if col_color_hist and col_color_hist not in cols_needed:
        cols_needed.append(col_color_hist)

    # グラフ描画で必要な列だけをselect
    # col_color_histを持っているDataFrameが1つもない場合、col_color_histを外付けする(複数dfの場合などでdf自体への色付け指定と見做す)
    any_df_has_col_color_hist = any(col_color_hist in df.columns for df in dfs)
    dfs_selected = [
        df.select([col for col in cols_needed if col in df.columns]).with_columns(pl.lit(col_color_scale_domain[i]).alias(col_color_hist)) if not any_df_has_col_color_hist else
        df.select([col for col in cols_needed if col in df.columns])
        for i, df in enumerate(dfs)
    ]

    # binとcolorごとに集約する
    # ついでに正規化(normalize)も行う
    col_y = 'count'
    def _agg(df:pl.DataFrame) -> pl.DataFrame:
        group_keys = [col_bin]
        if col_color_hist and (col_bin != col_color_hist):
            group_keys.append(col_color_hist)
        if normalize_hist == False:
            expr = pl.len().alias(col_y)
        else:
            expr = (pl.len() / pl.lit(len(df))).alias(col_y)
        if verbose:
            print(f'group_keys: {group_keys}')
        df_agg = df.group_by(pl.col(group_keys)).agg(expr)
        return df_agg
    dfs_agg = [_agg(df) for df in dfs_selected]

    # 結合
    df_agg_concat = pl.concat(dfs_agg)
    if verbose:   
        display(df_agg_concat)

    if df_bin_detail_info is not None:

        df_agg_concat = df_agg_concat.join(df_bin_detail_info, on=col_bin, how='left')

        col_bin_start = f"{col_bin}_start"
        col_bin_end = f"{col_bin}_end"

        df_plot = df_agg_concat.with_columns([
            pl.col(col_bin_start).alias("bin_left"),
            pl.col(col_bin_end).alias("bin_right"),
            pl.col(col_y).alias("bin_top"),
            pl.lit(0).alias("bin_bottom")
        ])

        dtype = df_plot.schema["bin_left"]
        bin_type = "temporal" if dtype in (pl.Datetime, pl.Date) else "quantitative"

        chart = alt.Chart(df_plot).encode(
            x=alt.X("bin_left", type=bin_type, title=None, scale=alt.Scale(zero=num_x_scale_zero_hist)),
            x2="bin_right",
            y=alt.Y("bin_bottom:Q", title="count"),
            y2=alt.Y2("bin_top:Q")
        )

    else:
        chart = alt.Chart(df_agg_concat).encode(
            x=alt.X(f"{col_bin}:N", title="category"),
            y=alt.Y(f"{col_y}:Q", title="count")
        )

    chart = chart.mark_bar(opacity=bar_opacity_hist, stroke='gray', strokeWidth=1).properties(title=chart_title)

    # mark_bar 後の色指定
    if col_color_scale_hist:
        chart = chart.encode(color=alt.Color(f'{col_color_hist}:N', legend=alt.Legend(title=col_color_legend_title_hist), scale=col_color_scale_hist))
    # if col_color:
    #     chart = chart.encode(color=alt.Color(f"{col_color}:N"))
    # elif color:
    #     legend_label = data_name
    #     # chart = chart.encode(color=alt.Color(f"legend_label:N", legend=alt.Legend(title=None), scale=alt.Scale(domain=[legend_label], range=[color])))
    #     chart = chart.encode(color=alt.Color(f"legend_label:N", legend=alt.Legend(title=None), scale=scale))

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
    data_name: str = 'train',
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

    # 固定色用のダミー列を仕込んでおく
    # 色設定のための仮の列を追加する(legend_title列、値はdata_name(trainなど)固定、色をそれにマッピング)
    if not col_color and color:
        df_agg = df_agg.with_columns(pl.lit(data_name).alias('legend_label'))

    if df_bin_detail_info is not None:
        df_agg = df_agg.join(df_bin_detail_info, on=col_bin, how="left")

    if verbose:
        print('df_agg:')
        display(df_agg)

    col_bin_median = f"{col_bin}_median"
    col_x = col_bin_median if col_bin_median in df_agg.columns else col_bin

    if verbose:
        print(f"col_x: {col_x}, col_color: {col_color}")

    chart_base = alt.Chart(df_agg)
    enc_x = alt.X(col_x)
    enc_y = alt.Y(f"{col_target_agg}:Q", axis=alt.Axis(orient="right"), scale=alt.Scale(zero=num_y_scale_zero))
    if col_color:
        enc_color_line = alt.Color(f"{col_color}:N", legend=alt.Legend(title=None), scale=alt.Scale(scheme=color_scale_scheme))
        enc_color_point = alt.Color(f"{col_color}:N", legend=None, scale=alt.Scale(scheme=color_scale_scheme))
    elif color:
        legend_label = data_name
        enc_color_line = alt.Color(f"legend_label:N", legend=alt.Legend(title=None), scale=alt.Scale(domain=[legend_label], range=[color]))
        enc_color_point = alt.Color(f"legend_label:N", legend=None, scale=alt.Scale(domain=[legend_label], range=[color]))
    else:
        chart_base = alt.Chart(df_agg)
        enc_color_line = alt.Undefined
        enc_color_point = alt.Undefined

    chart_line = chart_base.mark_line(size=line_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y, color=enc_color_line
    )

    chart_point = chart_base.mark_point(size=point_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y, color=enc_color_point#, shape=enc_shape

    )

    return alt.layer(chart_line, chart_point).resolve_scale(y='shared', color='independent', shape='independent')

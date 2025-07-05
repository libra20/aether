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
    col_color: str = 'data_name', # 優先

    # histogram
    col_color_scale_mode_hist: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_hist: list[str] = ['royalblue', 'indianred'], # map先(色)
    col_color_scale_scheme_hist: str = 'category10',
    col_color_legend_title_hist: str = None, #'data',
    bar_opacity_hist: float = 0.5,
    normalize_hist: bool = False,
    # histogram - numeric
    num_x_scale_zero_hist: bool = False,

    # line
    agg_func_line = pl.mean,
    num_y_scale_zero_line: bool = False,
    line_size: int = 1,
    line_opacity: float = 0.7,
    point_size: int = 5,

    # col_color: str = None, # 優先
    col_color_scale_mode_line: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_line: list[str] = ['gold', 'orange'], # map先(色)
    col_color_scale_scheme_line: str = 'category20',
    col_color_legend_title_line: str = None, #'target',

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

    # ヒストグラムチャート
    chart_hist = _plot_histogram_over_bin(
        *dfs,
        col_bin=col_bin,
        df_bin_detail_info=df_bin_detail_info,
        chart_title=chart_title,

        col_color_scale_domain=col_color_scale_domain,
        col_color=col_color,
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

    # 折れ線チャート
    chart_line = _plot_line_over_bin(
        *dfs,
        col_x_bin=col_bin,
        col_y=col_target,
        df_bin_detail_info=df_bin_detail_info,
        chart_title=chart_title,
        col_color_scale_domain=col_color_scale_domain,

        col_color=col_color, # 優先
        col_color_scale_mode_line=col_color_scale_mode_line,
        col_color_scale_range_line=col_color_scale_range_line, # map先(色)
        col_color_scale_scheme_line=col_color_scale_scheme_line,
        col_color_legend_title_line=col_color_legend_title_line,

        agg_func_line=agg_func_line,
        num_y_scale_zero_line=num_y_scale_zero_line,
        line_size=line_size,
        line_opacity=line_opacity,
        point_size=point_size,

        verbose=verbose,
    )

    # 総合チャート（ヒストグラム＋折れ線群）
    chart = alt.layer(chart_hist, chart_line).resolve_scale(
        y='independent', color='independent'
    )
    # chart = chart_line
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
    col_color: str = 'data_name', # 優先

    # histogram
    col_color_scale_mode_hist: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_hist: list[str] = ['royalblue', 'indianred'], # map先(色)
    col_color_scale_scheme_hist: str = 'category10',
    col_color_legend_title_hist: str = None, #'data',
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

    if col_color_legend_title_hist is None:
        col_color_legend_title_hist = chart_title
    
    # 色のスケール(変換ルール)を作成する
    if col_color_scale_mode_hist == 'domain_range':
        col_color_scale_hist=alt.Scale(domain=col_color_scale_domain, range=col_color_scale_range_hist)
    else:
        col_color_scale_hist=alt.Scale(scheme=col_color_scale_scheme_hist) 

    # グラフ描画で必要な列
    cols_needed = [col_bin]
    if col_color and col_color not in cols_needed:
        cols_needed.append(col_color)

    # グラフ描画で必要な列だけをselect
    # col_color_histを持っているDataFrameが1つもない場合、col_color_histを外付けする(複数dfの場合などでdf自体への色付け指定と見做す)
    any_df_has_col_color_hist = any(col_color in df.columns for df in dfs)
    dfs_selected = [
        df.select([col for col in cols_needed if col in df.columns]).with_columns(pl.lit(col_color_scale_domain[i]).alias(col_color)) if not any_df_has_col_color_hist else
        df.select([col for col in cols_needed if col in df.columns])
        for i, df in enumerate(dfs)
    ]

    # binとcolorごとに集約する
    # ついでに正規化(normalize)も行う
    col_y = 'count'
    def _agg(df:pl.DataFrame) -> pl.DataFrame:
        group_keys = [col_bin]
        if col_color and (col_bin != col_color):
            group_keys.append(col_color)
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
        chart = chart.encode(color=alt.Color(f'{col_color}:N', legend=alt.Legend(title=col_color_legend_title_hist), scale=col_color_scale_hist))

    return chart


"""
★plot_line_point_over_bin関数
"""
def _plot_line_over_bin(
    *dfs: pl.DataFrame,
    col_x_bin: str,
    col_y: str,
    df_bin_detail_info: pl.DataFrame = None,
    agg_func_line = pl.mean,
    col_color_scale_domain: list[str] = ['train', 'test'], # map元(値)
    chart_title: str = None,
    
    col_color: str = None, # 優先
    col_color_scale_mode_line: Literal["scheme", "domain_range"] = "domain_range",
    col_color_scale_range_line: list[str] = ['gold', 'orange'], # map先(色)
    col_color_scale_scheme_line: str = 'category20',
    col_color_legend_title_line: str = 'target',

    line_size: int = 1,
    line_opacity: float = 0.7,
    point_size: int = 5,
    num_y_scale_zero_line: bool = False,
    verbose: int = 0,
) -> alt.Chart:

    if chart_title is None:
        if col_x_bin.endswith("_bin"):
            chart_title = col_x_bin.removesuffix("_bin")
        else:
            chart_title = col_x_bin

    if col_color_legend_title_line is None:
        col_color_legend_title_line = col_y
    
    # 色のスケール(変換ルール)を作成する
    if col_color_scale_mode_line == 'domain_range':
        col_color_scale_line=alt.Scale(domain=col_color_scale_domain, range=col_color_scale_range_line)
    else:
        col_color_scale_line=alt.Scale(scheme=col_color_scale_scheme_line) 

    # グラフ描画で必要な列
    cols_needed = [col_x_bin]
    if col_y and col_y not in cols_needed:
        cols_needed.append(col_y)
    if col_color and col_color not in cols_needed:
        cols_needed.append(col_color)

    # グラフ描画で必要な列だけをselect
    # col_color_histを持っているDataFrameが1つもない場合、col_color_histを外付けする(複数dfの場合などでdf自体への色付け指定と見做す)
    any_df_has_col_color_line = any(col_color in df.columns for df in dfs)

    def _select_columns(
        df: pl.DataFrame,
        index: int,
    ) -> pl.DataFrame:
        # 必要な列のみselect
        df_selected = df.select([col for col in cols_needed if col in df.columns])

        # col_color 列が無いなら補う
        if not any_df_has_col_color_line:
            df_selected = df_selected.with_columns(
                pl.lit(col_color_scale_domain[index]).alias(col_color)
            )

        # col_y 列が無いなら None で追加
        if col_y not in df_selected.columns:
            df_selected = df_selected.with_columns(
                pl.lit(None).alias(col_y)
            )

        return df_selected
    dfs_selected = [
        _select_columns(df, index)
        for index, df in enumerate(dfs)
    ]

    # binとcolorごとに集約する
    col_y_agg = f'{col_y} ({agg_func_line.__name__})'
    group_keys = [col_x_bin] + ([col_color] if col_color else [])
    dfs_agg = [
        df.group_by(group_keys).agg(agg_func_line(col_y).alias(col_y_agg))
        for df in dfs_selected
    ]

    # 結合
    df_agg_concat = pl.concat(dfs_agg)

    if verbose:
        print('df_agg_concat')
        display(df_agg_concat)

    # bin情報からxとして採用する情報を決定
    if df_bin_detail_info is not None:
        df_agg_concat = df_agg_concat.join(df_bin_detail_info, on=col_x_bin, how="left")
    col_bin_median = f"{col_x_bin}_median"
    col_x = col_bin_median if col_bin_median in df_agg_concat.columns else col_x_bin

    # グラフ(色以外)
    chart_base = alt.Chart(df_agg_concat)
    enc_x = alt.X(col_x)
    enc_y = alt.Y(f"{col_y_agg}:Q", axis=alt.Axis(orient="right"), scale=alt.Scale(zero=num_y_scale_zero_line))
    chart_line = chart_base.mark_line(size=line_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y
    )
    chart_point = chart_base.mark_point(size=point_size, opacity=line_opacity).encode(
        x=enc_x, y=enc_y
    )

    # 色塗り
    if col_color:
        enc_color_line = alt.Color(f"{col_color}:N", legend=alt.Legend(title=col_color_legend_title_line), scale=col_color_scale_line)
        chart_line = chart_line.encode(color=enc_color_line)
        enc_color_point = alt.Color(f"{col_color}:N", legend=None, scale=col_color_scale_line)
        chart_point = chart_point.encode(color=enc_color_point)

    # lineとpointを重ねて返す
    chart =  alt.layer(chart_line, chart_point).resolve_scale(y='shared', color='independent', shape='independent')
    return chart

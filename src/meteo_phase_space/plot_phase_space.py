"""Phase-space and Köppen-frequency figures (matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .koppen import KOPPEN_COLORS

SpreadType = Literal["quantile", "standard deviation"]

ModeLiteral = Literal["spaghetti", "normals"]


def save_figure(
    fig: Figure,
    path: Path | str,
    *,
    dpi: float = 200,
    bbox_inches: str | None = "tight",
) -> None:
    """Save ``fig`` then close it to avoid leaks in batch workflows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {"dpi": dpi}
    if bbox_inches is not None:
        kw["bbox_inches"] = bbox_inches
    fig.savefig(p, **kw)
    plt.close(fig)


def _subtitle_station(results: dict[str, Any]) -> str:
    return f"{results['display_name']} (station {results['station_id']})"


def plot_phase_space(
    results: dict[str, Any],
    mode: ModeLiteral = "spaghetti",
    spread_type: SpreadType | str = "standard deviation",
) -> Figure:
    """Build temperature–precipitation phase space; does not ``show()`` or save."""
    df = results["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    loc = _subtitle_station(results)

    if mode == "spaghetti":
        history: dict[int, Any] = results["history"]
        used_classes = sorted(set(history.values()))
        for year, k_class in history.items():
            year_df = df[df.index.year == year]
            color = KOPPEN_COLORS.get(str(k_class), "#000000")
            ax.plot(year_df["temp"], year_df["prcp"], color=color, alpha=0.3, linewidth=1.5, zorder=1)
            ax.scatter(
                year_df["temp"],
                year_df["prcp"],
                color=color,
                marker="o",
                s=15,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )

        legend_elements = [
            Line2D([0], [0], color=KOPPEN_COLORS.get(cast(str, c), "#000000"), lw=2, label=str(c))
            for c in used_classes
        ]
        ax.legend(handles=legend_elements, title="Climate Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

        title = (
            f"Yearly spaghetti: {loc}\nNormals class: {results['normal_class']}"
        )
        if results["official_normal_class"] is not None:
            title += f" | Official normals class: {results['official_normal_class']}"
        ax.set_title(title)

    elif mode == "normals":
        grouped = df.groupby(df.index.month)
        if spread_type == "quantile":
            q = grouped[["temp", "prcp"]].quantile([0.25, 0.5, 0.75]).unstack()
            q = cast(Any, q)
            x_mid = q["temp"][0.5]
            y_mid = q["prcp"][0.5]
            x_err = [x_mid - q["temp"][0.25], q["temp"][0.75] - x_mid]
            y_err = [y_mid - q["prcp"][0.25], q["prcp"][0.75] - y_mid]
            label_suffix = "(median & IQR)"
        elif spread_type == "standard deviation":
            stats = grouped.agg({"temp": ["mean", "std"], "prcp": ["mean", "std"]})
            x_mid = stats["temp"]["mean"]
            y_mid = stats["prcp"]["mean"]
            x_err = stats["temp"]["std"]
            y_err = stats["prcp"]["std"]
            label_suffix = "(mean & std dev)"
        else:
            raise NotImplementedError(f"Spread type {spread_type} not implemented")

        ax.plot(x_mid, y_mid, "ko-", linewidth=2, label="Central path", zorder=3)
        ax.errorbar(
            x_mid,
            y_mid,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            alpha=0.6,
            zorder=2,
        )
        for i, m in enumerate(x_mid.index):
            ax.annotate(
                str(m),
                (float(x_mid.iloc[i]), float(y_mid.iloc[i])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color="red",
                fontweight="bold",
            )

        ax.set_title(
            f"Monthly normals phase space: {loc} {label_suffix}\nNormals class: {results['normal_class']}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Precipitation (mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_koppen_frequency(results: dict[str, Any]) -> Figure:
    """Bar chart of yearly Köppen class counts."""
    history: dict[int, Any] = results["history"]
    fig, ax = plt.subplots(figsize=(10, 6))
    if not history:
        ax.set_title(f"Köppen frequency: {_subtitle_station(results)}\n(No full calendar years)")
        ax.set_xlabel("Köppen classification")
        ax.set_ylabel("Number of years")
        fig.tight_layout()
        return fig

    history_values = [str(v) for v in history.values()]
    unique_classes, counts = np.unique(history_values, return_counts=True)
    colors = [KOPPEN_COLORS.get(cast(str, c), "#000000") for c in unique_classes]

    bars = ax.bar(unique_classes, counts, color=colors, edgecolor="black", alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.1,
            str(int(yval)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    y0 = min(history.keys())
    y1 = max(history.keys())
    ax.set_title(
        f"Köppen class frequency: {_subtitle_station(results)}\n{y0}–{y1}",
    )
    ax.set_ylabel("Number of years")
    ax.set_xlabel("Köppen classification")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig

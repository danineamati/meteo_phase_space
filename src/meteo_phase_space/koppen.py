"""Köppen-Geiger color map and vectorized classification (Beck et al. 2018 logic)."""

from __future__ import annotations

import numpy as np

KOPPEN_COLORS = {
    "Af": "#0000ff",
    "Am": "#0077ff",
    "Aw": "#41a0fc",
    "BWh": "#ff0000",
    "BWk": "#ff9696",
    "BSh": "#f5a500",
    "BSk": "#ffdb5c",
    "Csa": "#ffff00",
    "Csb": "#c8c800",
    "Csc": "#8c8c00",
    "Cwa": "#96ff96",
    "Cwb": "#64ff64",
    "Cwc": "#32cd32",
    "Cfa": "#c8ff50",
    "Cfb": "#64ff50",
    "Cfc": "#32cd00",
    "Dsa": "#ff00ff",
    "Dsb": "#c800c8",
    "Dsc": "#960096",
    "Dsd": "#9600ff",
    "Dwa": "#b464ff",
    "Dwb": "#a000ff",
    "Dwc": "#640096",
    "Dwd": "#c800ff",
    "Dfa": "#00ffff",
    "Dfb": "#32c8ff",
    "Dfc": "#007d7d",
    "Dfd": "#004b4b",
    "ET": "#b2b2b2",
    "EF": "#666666",
    "None": "#000000",
}


def classify_koppen(temp: np.ndarray, precip: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Classifies climate zones using vectorized Köppen-Geiger logic (Beck et al. 2018).

    Args:
        temp: (N, 12) Celsius monthly means.
        precip: (N, 12) mm monthly totals.
        lats: (N,) decimal degrees (positive north).

    Returns:
        (N,) string array of Köppen codes.
    """
    assert isinstance(temp, np.ndarray)
    assert isinstance(precip, np.ndarray)
    assert isinstance(lats, np.ndarray)
    assert temp.shape == precip.shape, f"T shape {temp.shape} and P shape {precip.shape}"
    assert temp.shape[0] == lats.shape[0], f"T shape {temp.shape} and lat shape {lats.shape}"
    assert len(temp.shape) == 2, f"T shape {temp.shape}"

    mat = temp.mean(axis=1)
    map_total = precip.sum(axis=1)

    t_hot = temp.max(axis=1)
    t_cold = temp.min(axis=1)
    t_mon10 = (temp > 10).sum(axis=1)

    p_apr_sep = precip[:, 3:9].sum(axis=1)
    p_oct_mar = map_total - p_apr_sep
    is_northern = lats >= 0

    p_summer = np.where(is_northern, p_apr_sep, p_oct_mar)
    p_winter = np.where(is_northern, p_oct_mar, p_apr_sep)

    pth = np.where(
        p_summer >= 0.7 * map_total,
        2 * mat + 28,
        np.where(p_winter >= 0.7 * map_total, 2 * mat, 2 * mat + 14),
    )

    p_s_dry = np.where(
        is_northern,
        precip[:, 3:9].min(axis=1),
        np.minimum(precip[:, :3].min(axis=1), precip[:, 9:].min(axis=1)),
    )
    p_s_wet = np.where(
        is_northern,
        precip[:, 3:9].max(axis=1),
        np.maximum(precip[:, :3].max(axis=1), precip[:, 9:].max(axis=1)),
    )
    p_w_dry = np.where(
        is_northern,
        np.minimum(precip[:, :3].min(axis=1), precip[:, 9:].min(axis=1)),
        precip[:, 3:9].min(axis=1),
    )
    p_w_wet = np.where(
        is_northern,
        np.maximum(precip[:, :3].max(axis=1), precip[:, 9:].max(axis=1)),
        precip[:, 3:9].max(axis=1),
    )
    p_dry = precip.min(axis=1)

    classes = np.full(temp.shape[0], "None", dtype="U4")

    is_b = map_total < 10 * pth
    is_bw = is_b & (map_total < 5 * pth)
    is_bs = is_b & ~is_bw
    is_bh = is_b & (mat >= 18)
    is_bk = is_b & ~is_bh
    classes = np.where(
        is_bw & is_bh,
        "BWh",
        np.where(
            is_bw & is_bk,
            "BWk",
            np.where(
                is_bs & is_bh,
                "BSh",
                np.where(is_bs & is_bk, "BSk", classes),
            ),
        ),
    )

    is_a = (t_cold >= 18) & ~is_b
    is_af = is_a & (p_dry >= 60)
    is_am = is_a & ~is_af & (p_dry >= 100 - map_total / 25)
    is_aw = is_a & ~is_af & ~is_am
    classes = np.where(is_af, "Af", np.where(is_am, "Am", np.where(is_aw, "Aw", classes)))

    is_e = (t_hot <= 10) & ~is_b
    classes = np.where(
        is_e & (t_hot > 0), "ET", np.where(is_e & (t_hot <= 0), "EF", classes)
    )

    is_c = ~is_b & ~is_a & ~is_e & (t_cold > 0) & (t_cold < 18)
    is_d = ~is_b & ~is_a & ~is_e & (t_cold <= 0)

    for zone_mask, zone_letter in [(is_c, "C"), (is_d, "D")]:
        is_s = zone_mask & (p_s_dry < 40) & (p_s_dry < p_w_wet / 3)
        is_w = zone_mask & ~is_s & (p_w_dry < p_s_wet / 10)
        is_f = zone_mask & ~is_s & ~is_w

        is_ta = zone_mask & (t_hot >= 22)
        is_tb = zone_mask & ~is_ta & (t_mon10 >= 4)
        is_tc = zone_mask & ~is_ta & ~is_tb & (t_cold >= -38)
        is_td = zone_mask & (zone_letter == "D") & ~is_ta & ~is_tb & (t_cold < -38)

        for p_sub, p_let in [(is_s, "s"), (is_w, "w"), (is_f, "f")]:
            for t_sub, t_let in [(is_ta, "a"), (is_tb, "b"), (is_tc, "c"), (is_td, "d")]:
                classes = np.where(p_sub & t_sub, f"{zone_letter}{p_let}{t_let}", classes)

    return classes

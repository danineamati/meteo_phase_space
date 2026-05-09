"""Temperature–precipitation phase-space plots and Köppen classification."""

from __future__ import annotations

from .koppen import KOPPEN_COLORS, classify_koppen
from .meteo_query import fetch_and_classify_station, nearest_station_id, station_inventory_summary
from .plot_paths import location_slug, plots_subdir
from .plot_phase_space import plot_koppen_frequency, plot_phase_space, save_figure

__all__ = [
    "KOPPEN_COLORS",
    "classify_koppen",
    "fetch_and_classify_station",
    "nearest_station_id",
    "station_inventory_summary",
    "location_slug",
    "plots_subdir",
    "plot_phase_space",
    "plot_koppen_frequency",
    "save_figure",
]

__version__ = "0.1.0"

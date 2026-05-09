"""CLI: ``meteo-phase-space``."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Annotated

import matplotlib.pyplot as plt
import tyro
from matplotlib.figure import Figure
from tyro.conf import arg, subcommand

from .meteo_query import fetch_and_classify_station, nearest_station_id
from .plot_paths import plots_subdir
from .plot_phase_space import plot_koppen_frequency, plot_phase_space, save_figure

_LOG = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = "generated_plots"


@dataclass
class StationPlottingDataClass:
    """Fetch and plot using a Meteostat station identifier (CLI subcommand ``station``)."""

    station_id: str
    """Station id (numeric WMO style or ICAO-style string as supported by Meteostat)."""

    name: str | None = None
    """Human-readable label for titles and filenames (default: ``Station {id}``)."""

    start_year: int = 1990
    end_year: int = 2025
    verbose: bool = False
    """Enable debug logging."""

    save_plots: bool = False
    """Write PNG figures when set; output root defaults to ``generated_plots`` (override with ``--plots-dir``)."""

    plots_dir: str = DEFAULT_PLOTS_DIR
    """Output root; figures go under ``<plots-dir>/<location_slug>/``."""

    plot_dpi: float = 150
    bbox_tight: bool = True
    """When false, omit ``bbox_inches`` from ``savefig``."""

    show: bool = False
    """Show interactive matplotlib windows after optional save."""

    should_plot_spaghetti: Annotated[bool, arg(aliases=("--plot-spaghetti",))] = True
    should_plot_normals_sd: Annotated[bool, arg(aliases=("--plot-normals-sd",))] = True
    should_plot_normals_iqr: Annotated[bool, arg(aliases=("--plot-normals-iqr",))] = True
    should_plot_koppen_frequency: Annotated[bool, arg(aliases=("--plot-koppen-frequency",))] = True


@dataclass
class Coords:
    """Resolve the nearest station to latitude/longitude, then fetch and plot."""

    lat: float
    lon: float

    name: str | None = None
    """Human-readable label (default: ``lat_lon`` truncated)."""

    start_year: int = 1990
    end_year: int = 2025
    verbose: bool = False
    save_plots: bool = False
    plots_dir: str = DEFAULT_PLOTS_DIR
    plot_dpi: float = 150
    bbox_tight: bool = True
    show: bool = False
    should_plot_spaghetti: Annotated[bool, arg(aliases=("--plot-spaghetti",))] = True
    should_plot_normals_sd: Annotated[bool, arg(aliases=("--plot-normals-sd",))] = True
    should_plot_normals_iqr: Annotated[bool, arg(aliases=("--plot-normals-iqr",))] = True
    should_plot_koppen_frequency: Annotated[bool, arg(aliases=("--plot-koppen-frequency",))] = True


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _display_name_station(cli: StationPlottingDataClass) -> str:
    return cli.name if cli.name is not None else f"Station {cli.station_id}"


def _display_name_coords(cli: Coords) -> str:
    if cli.name is not None:
        return cli.name
    return f"{cli.lat:.2f},{cli.lon:.2f}"


def _run_results(
    results: dict,
    *,
    save_plots: bool,
    plots_dir: str,
    plot_dpi: float,
    bbox_inches: str | None,
    show: bool,
    should_plot_spaghetti: bool,
    should_plot_normals_sd: bool,
    should_plot_normals_iqr: bool,
    should_plot_koppen_frequency: bool,
) -> None:
    want_figures = any(
        [
            should_plot_spaghetti,
            should_plot_normals_sd,
            should_plot_normals_iqr,
            should_plot_koppen_frequency,
        ]
    )
    if not want_figures:
        if show:
            _LOG.warning(
                "Nothing to display: enable at least one plotting flag "
                "(e.g. `--should-plot-spaghetti` or `--plot-spaghetti`)."
            )
        return

    if save_plots:
        out_root = plots_subdir(plots_dir, results["display_name"])
        dpi = plot_dpi
        figures = _build_requested_figures(
            results,
            should_plot_spaghetti,
            should_plot_normals_sd,
            should_plot_normals_iqr,
            should_plot_koppen_frequency,
        )

        for fname, fig in figures:
            save_figure(fig, out_root / fname, dpi=dpi, bbox_inches=bbox_inches)

        _LOG.info("Saved %s plot(s) under %s", len(figures), out_root)

    elif show:
        figures = _build_requested_figures(
            results,
            should_plot_spaghetti,
            should_plot_normals_sd,
            should_plot_normals_iqr,
            should_plot_koppen_frequency,
        )
        plt.show()
        plt.close("all")

    else:
        _LOG.debug("Skipping plots (use --save-plots or --show to produce output).")


def _build_requested_figures(
    results: dict,
    should_plot_spaghetti: bool,
    should_plot_normals_sd: bool,
    should_plot_normals_iqr: bool,
    should_plot_koppen_frequency: bool,
) -> list[tuple[str, Figure]]:
    pairs: list[tuple[str, Figure]] = []
    if should_plot_spaghetti:
        pairs.append(("spaghetti.png", plot_phase_space(results, mode="spaghetti")))
    if should_plot_normals_sd:
        pairs.append(
            (
                "normals_standard_deviation.png",
                plot_phase_space(results, mode="normals", spread_type="standard deviation"),
            )
        )
    if should_plot_normals_iqr:
        pairs.append(
            ("normals_quantile.png", plot_phase_space(results, mode="normals", spread_type="quantile"))
        )
    if should_plot_koppen_frequency:
        pairs.append(("koppen_frequency.png", plot_koppen_frequency(results)))
    return pairs


def _dispatch_station(cli: StationPlottingDataClass) -> None:
    display = _display_name_station(cli)

    _configure_logging(cli.verbose)

    payload = fetch_and_classify_station(
        cli.station_id,
        cli.start_year,
        cli.end_year,
        display_name=display,
    )
    if payload is None:
        _LOG.error("No data retrieved for station %s", cli.station_id)
        sys.exit(1)

    _LOG.info(
        "Normals Köppen: %s (official normals: %s)",
        payload["normal_class"],
        payload["official_normal_class"],
    )

    bbox = "tight" if cli.bbox_tight else None
    _run_results(
        payload,
        save_plots=cli.save_plots,
        plots_dir=cli.plots_dir,
        plot_dpi=cli.plot_dpi,
        bbox_inches=bbox,
        show=cli.show,
        should_plot_spaghetti=cli.should_plot_spaghetti,
        should_plot_normals_sd=cli.should_plot_normals_sd,
        should_plot_normals_iqr=cli.should_plot_normals_iqr,
        should_plot_koppen_frequency=cli.should_plot_koppen_frequency,
    )


def _dispatch_coords(cli: Coords) -> None:
    display = _display_name_coords(cli)

    _configure_logging(cli.verbose)

    sid = nearest_station_id(cli.lat, cli.lon)
    _LOG.info("Nearest station to (%.4f, %.4f): %s", cli.lat, cli.lon, sid)

    payload = fetch_and_classify_station(
        sid,
        cli.start_year,
        cli.end_year,
        display_name=display,
    )
    if payload is None:
        _LOG.error("No data retrieved for resolved station %s", sid)
        sys.exit(1)

    _LOG.info(
        "Normals Köppen: %s (official normals: %s)",
        payload["normal_class"],
        payload["official_normal_class"],
    )

    bbox = "tight" if cli.bbox_tight else None
    _run_results(
        payload,
        save_plots=cli.save_plots,
        plots_dir=cli.plots_dir,
        plot_dpi=cli.plot_dpi,
        bbox_inches=bbox,
        show=cli.show,
        should_plot_spaghetti=cli.should_plot_spaghetti,
        should_plot_normals_sd=cli.should_plot_normals_sd,
        should_plot_normals_iqr=cli.should_plot_normals_iqr,
        should_plot_koppen_frequency=cli.should_plot_koppen_frequency,
    )


def _entry(cli: StationPlottingDataClass | Coords) -> None:
    if isinstance(cli, StationPlottingDataClass):
        _dispatch_station(cli)
    else:
        _dispatch_coords(cli)


def main() -> None:
    CliSpec = Annotated[StationPlottingDataClass, subcommand("station")] | Annotated[
        Coords, subcommand("coords")
    ]
    cmd = tyro.cli(CliSpec)
    _entry(cmd)


if __name__ == "__main__":
    main()

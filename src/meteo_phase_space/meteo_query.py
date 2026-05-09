"""Fetch monthly Meteostat data and Köppen classification for a station."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import meteostat as ms
import numpy as np
import pandas as pd

from .koppen import classify_koppen

logger = logging.getLogger(__name__)


def nearest_station_id(latitude: float, longitude: float) -> str:
    """
    Resolve the geographically nearest station using Meteostat ``stations.nearby``.

    Rows are assumed to be ordered by proximity; the first row is chosen.
    """
    point = ms.Point(latitude=latitude, longitude=longitude)
    nearby = ms.stations.nearby(point)
    if nearby is None or nearby.empty:
        raise ValueError(f"No Meteostat station found near ({latitude}, {longitude})")
    station_id = str(nearby.index[0])
    logger.debug("Nearest station id=%s to (%.4f, %.4f):\n%s", station_id, latitude, longitude, nearby.head())
    return station_id


def fetch_and_classify_station(
    station_id: str,
    start_yr: int,
    end_yr: int,
    *,
    display_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Fetch monthly normals and yearly history for ``station_id`` and classify Köppen zones.

    Returns ``None`` if no monthly rows are returned. Otherwise a dict:

    ``station_id``, ``latitude``, ``display_name``, ``df``, ``normal_class``,
    ``official_normal_class``, ``official_normal_df``, ``history`` (year -> class).
    """
    label_default = display_name if display_name is not None else f"Station {station_id}"

    data = ms.monthly(
        station_id,
        datetime(start_yr, 1, 1),
        datetime(end_yr, 12, 31),
        parameters=["temp", "prcp"],
    ).fetch()
    if data.empty:
        logger.warning("No monthly data for station %s in %s–%s", station_id, start_yr, end_yr)
        return None

    station = ms.stations.meta(station_id)
    logger.debug("Station meta: %s", station)
    lat_val = getattr(station, "latitude", None)
    lat = float(lat_val) if lat_val is not None else float("nan")
    logger.info("Fetched station %s (lat=%.4f) %s–%s", station_id, lat, start_yr, end_yr)

    logger.debug("Monthly rows:\n%s", data)
    logger.debug("Missing temp: %s, prcp: %s", data["temp"].isna().sum(), data["prcp"].isna().sum())

    data["temp"] = data["temp"].ffill().bfill()
    data["prcp"] = data["prcp"].fillna(0)

    normals = data.groupby(data.index.month).agg({"temp": "mean", "prcp": "mean"})
    normal_class = classify_koppen(
        np.array(normals["temp"].values.reshape(1, 12)),
        np.array(normals["prcp"].values.reshape(1, 12)),
        np.array([lat], dtype=float),
    )[0]
    logger.debug("Derived normals Köppen class=%s:\n%s", normal_class, normals)

    official_normals_ts = ms.normals(ms.Station(id=station_id), start_yr, end_yr, parameters=["temp", "prcp"])
    official_normals_df = official_normals_ts.fetch()
    logger.debug("Official normals:\n%s", official_normals_df)

    official_normals_class: str | None = None
    if "temp" in official_normals_df.columns and "prcp" in official_normals_df.columns:
        if len(official_normals_df.index) >= 12:
            official_normals_class = classify_koppen(
                np.array(official_normals_df["temp"].values.reshape(1, 12)),
                np.array(official_normals_df["prcp"].values.reshape(1, 12)),
                np.array([lat], dtype=float),
            )[0]

    grouped = sorted(
        [(int(y), g) for y, g in data.groupby(data.index.year) if len(g) == 12],
        key=lambda yg: yg[0],
    )
    years_sorted = [y for y, _ in grouped]
    logger.debug("Full years (%s): %s", len(years_sorted), years_sorted)

    if not grouped:
        yearly_classes = np.array([], dtype="U4")
    else:
        t_years = np.stack([g["temp"].values for _, g in grouped])
        p_years = np.stack([g["prcp"].values for _, g in grouped])
        yearly_classes = classify_koppen(np.asarray(t_years), np.asarray(p_years), np.full(len(years_sorted), lat))

    history = dict(zip(years_sorted, list(yearly_classes)))

    return {
        "station_id": station_id,
        "latitude": lat,
        "display_name": label_default,
        "df": data,
        "normal_class": normal_class,
        "official_normal_class": official_normals_class,
        "official_normal_df": official_normals_df,
        "history": history,
    }


def station_inventory_summary(station_id: str) -> str:
    """Return a short string describing inventory for a station (for CLI/logging)."""
    inv = ms.stations.inventory(station_id)
    lines = [
        f"Station {station_id}: data available from {inv.start} to {inv.end}.",
        "Available parameters:",
        *[f"  - {param}" for param in inv.parameters],
    ]
    return "\n".join(lines)

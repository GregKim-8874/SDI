import json
import re
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box, mapping, shape
from shapely.ops import transform

WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:3857"


def _build_roi_polygon(
    lat: Optional[float],
    lon: Optional[float],
    geom_json: Optional[Dict[str, Any]],
    buffer_m: float,
) -> Polygon:
    if geom_json is not None:
        roi_geom = shape(geom_json)
        if not isinstance(roi_geom, Polygon):
            raise ValueError("geom_json must be a GeoJSON Polygon")
        return roi_geom

    if lat is None or lon is None:
        raise ValueError("Either (lat, lon) or geom_json must be provided")

    forward = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
    inverse = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)
    pt_m = transform(forward.transform, Point(lon, lat))
    roi_m = pt_m.buffer(buffer_m)
    return transform(inverse.transform, roi_m)


def _as_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if pd.isna(val):
        return []
    return [val]


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).lower().strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = float(match.group(1))
    if "mph" in text:
        number *= 1.60934
    return number


def _extract_lane_values(val: Any) -> List[float]:
    values: List[float] = []
    for item in _as_list(val):
        if item is None or pd.isna(item):
            continue
        for num in re.findall(r"\d+(?:\.\d+)?", str(item)):
            values.append(float(num))
    return values


def _axis_positions(start: float, step: float, grid_size: float, min_v: float, max_v: float) -> List[float]:
    span = max_v - min_v
    n = int(span / step) + 4
    vals: List[float] = []
    for k in range(-n, n + 1):
        pos = start + (k * step)
        if pos < max_v and (pos + grid_size) > min_v:
            vals.append(pos)
    return sorted(set(vals))


def _build_slices_grid(
    roi_m: Polygon,
    grid_size_m: float,
    overlap_m: float,
    anchor: str,
) -> List[Tuple[str, Polygon]]:
    if overlap_m >= grid_size_m:
        raise ValueError("overlap_m must be smaller than grid_size_m")
    if anchor not in {"left", "right", "mid"}:
        raise ValueError("anchor must be one of: left, right, mid")

    minx, miny, maxx, maxy = roi_m.bounds
    step = grid_size_m - overlap_m
    width = maxx - minx
    height = maxy - miny

    if anchor == "left":
        x0 = minx
        y0 = miny
    elif anchor == "right":
        x0 = maxx - grid_size_m
        y0 = maxy - grid_size_m
    else:
        x0 = minx + ((width % step) / 2.0)
        y0 = miny + ((height % step) / 2.0)

    xs = _axis_positions(x0, step, grid_size_m, minx, maxx)
    ys = _axis_positions(y0, step, grid_size_m, miny, maxy)

    slices: List[Tuple[str, Polygon]] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            if cell.intersects(roi_m):
                slices.append((f"grid_{row}_{col}", cell))
    return slices


def _subset_by_sindex(gdf: gpd.GeoDataFrame, geom) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    idx = list(gdf.sindex.query(geom, predicate="intersects"))
    if not idx:
        return gdf.iloc[0:0]
    return gdf.iloc[idx]


def _sum_line_length_within(lines: gpd.GeoDataFrame, geom) -> float:
    if lines.empty:
        return 0.0
    clipped = lines.geometry.intersection(geom)
    return float(clipped.length.sum())


def _sum_area_within(polys: gpd.GeoDataFrame, geom) -> float:
    if polys.empty:
        return 0.0
    clipped = polys.geometry.intersection(geom)
    return float(clipped.area.sum())


def _yes_like(v: Any) -> bool:
    if pd.isna(v) or v is None:
        return False
    return str(v).lower() in {"yes", "true", "1"}


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, pd.Series):
        return _serialize(value.to_dict())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def extract_sdi_raw_slices(
    lat=None,
    lon=None,
    geom_json=None,
    buffer_m=500.0,
    slicing_mode="grid",
    grid_size_m=100.0,
    overlap_m=50.0,
    anchor="mid",
    r2_df=None,
    network_type="drive",
    output_path=None,
):
    """Return slice-level raw SDI features + raw R2."""
    if slicing_mode != "grid":
        raise ValueError("Only slicing_mode='grid' is currently supported")

    roi_4326 = _build_roi_polygon(lat, lon, geom_json, buffer_m)
    to_m = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
    to_wgs = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)
    roi_m = transform(to_m.transform, roi_4326)

    slices = _build_slices_grid(roi_m, grid_size_m, overlap_m, anchor)

    G = ox.graph_from_polygon(roi_4326, network_type=network_type)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    features_gdf = ox.features_from_polygon(
        roi_4326,
        tags={
            "highway": [
                "traffic_signals",
                "crossing",
                "bus_stop",
                "footway",
                "path",
                "pedestrian",
            ],
            "junction": "roundabout",
            "amenity": True,
            "bridge": "yes",
            "tunnel": "yes",
            "building": True,
            "landuse": True,
            "shop": True,
        },
    )

    nodes_m = nodes_gdf.to_crs(METRIC_CRS)
    edges_m = edges_gdf.to_crs(METRIC_CRS)
    features_m = features_gdf.to_crs(METRIC_CRS)

    degree_map = dict(nx.degree(G))
    nodes_m["degree"] = nodes_m.index.map(lambda i: degree_map.get(i, 0))

    empty_features = features_m.iloc[0:0]
    traffic_signals = (
        features_m[features_m["highway"].astype(str) == "traffic_signals"]
        if "highway" in features_m.columns
        else empty_features
    )
    crossings = (
        features_m[features_m["highway"].astype(str) == "crossing"]
        if "highway" in features_m.columns
        else empty_features
    )
    bus_stops = (
        features_m[features_m["highway"].astype(str) == "bus_stop"]
        if "highway" in features_m.columns
        else empty_features
    )
    roundabouts = (
        features_m[features_m["junction"].astype(str) == "roundabout"]
        if "junction" in features_m.columns
        else empty_features
    )
    parkings = (
        features_m[features_m["amenity"].astype(str) == "parking"]
        if "amenity" in features_m.columns
        else empty_features
    )
    buildings = features_m[features_m["building"].notna()] if "building" in features_m.columns else empty_features
    landuse = features_m[features_m["landuse"].notna()] if "landuse" in features_m.columns else empty_features
    footways = (
        features_m[features_m["highway"].isin(["footway", "path", "pedestrian"])].copy()
        if "highway" in features_m.columns
        else empty_features
    )
    commercial_poi = features_m[features_m["shop"].notna()] if "shop" in features_m.columns else empty_features

    green_landuse_classes = {
        "grass",
        "forest",
        "meadow",
        "park",
        "recreation_ground",
        "village_green",
        "greenfield",
    }
    green_areas = landuse[landuse["landuse"].astype(str).isin(green_landuse_classes)] if not landuse.empty else landuse

    r2_points_m = None
    if isinstance(r2_df, pd.DataFrame) and {"lat", "lon"}.issubset(set(r2_df.columns)):
        r2_points_m = gpd.GeoDataFrame(
            r2_df.copy(),
            geometry=gpd.points_from_xy(r2_df["lon"], r2_df["lat"]),
            crs=WGS84,
        ).to_crs(METRIC_CRS)

    slices_out = []
    for slice_id, slice_geom in slices:
        slice_roi_geom = slice_geom.intersection(roi_m)

        edges_slice = _subset_by_sindex(edges_m, slice_geom)
        nodes_slice = _subset_by_sindex(nodes_m, slice_geom)
        ts_slice = _subset_by_sindex(traffic_signals, slice_geom)
        crossings_slice = _subset_by_sindex(crossings, slice_geom)
        roundabout_slice = _subset_by_sindex(roundabouts, slice_geom)
        bus_slice = _subset_by_sindex(bus_stops, slice_geom)
        parking_slice = _subset_by_sindex(parkings, slice_geom)
        building_slice = _subset_by_sindex(buildings, slice_geom)
        landuse_slice = _subset_by_sindex(landuse, slice_geom)
        green_slice = _subset_by_sindex(green_areas, slice_geom)
        footway_slice = _subset_by_sindex(footways, slice_geom)
        commercial_slice = _subset_by_sindex(commercial_poi, slice_geom)

        road_total_length_m = _sum_line_length_within(edges_slice, slice_geom)

        road_class_counts: Dict[str, int] = {}
        if not edges_slice.empty and "highway" in edges_slice.columns:
            for hv in edges_slice["highway"]:
                for cls in _as_list(hv):
                    key = str(cls)
                    road_class_counts[key] = road_class_counts.get(key, 0) + 1

        lane_values: List[float] = []
        if "lanes" in edges_slice.columns:
            for item in edges_slice["lanes"]:
                lane_values.extend(_extract_lane_values(item))

        speed_values: List[float] = []
        if "maxspeed" in edges_slice.columns:
            for item in edges_slice["maxspeed"]:
                for v in _as_list(item):
                    parsed = _parse_numeric(v)
                    if parsed is not None:
                        speed_values.append(parsed)

        turn_lanes_ratio = None
        if "turn:lanes" in edges_slice.columns and len(edges_slice) > 0:
            present = edges_slice["turn:lanes"].notna().sum()
            turn_lanes_ratio = float(present / len(edges_slice))

        node_intersections = nodes_slice[nodes_slice["degree"] >= 3]

        bridge_count = int(edges_slice["bridge"].apply(_yes_like).sum()) if "bridge" in edges_slice.columns else 0
        tunnel_count = int(edges_slice["tunnel"].apply(_yes_like).sum()) if "tunnel" in edges_slice.columns else 0

        landuse_area_by_class: Dict[str, float] = {}
        if not landuse_slice.empty:
            grouped = landuse_slice.groupby(landuse_slice["landuse"].astype(str))
            for lu_class, subset in grouped:
                landuse_area_by_class[lu_class] = _sum_area_within(subset, slice_geom)

        raw_r2_values: List[Dict[str, Any]] = []
        if r2_points_m is not None:
            subset = _subset_by_sindex(r2_points_m, slice_geom)
            subset = subset[subset.geometry.within(slice_geom)]
            raw_r2_values = subset.drop(columns="geometry").to_dict(orient="records")

        slice_4326 = transform(to_wgs.transform, slice_roi_geom)
        bbox = [float(v) for v in slice_4326.bounds]

        slices_out.append(
            {
                "slice_id": slice_id,
                "geometry": mapping(slice_4326),
                "area_m2": float(slice_roi_geom.area),
                "bbox_4326": bbox,
                "RGF": {
                    "road_total_length_m": road_total_length_m,
                    "road_class_counts": road_class_counts,
                    "lane_count_mean": float(sum(lane_values) / len(lane_values)) if lane_values else None,
                    "lane_count_max": max(lane_values) if lane_values else None,
                    "speed_limit_mean": float(sum(speed_values) / len(speed_values)) if speed_values else None,
                    "speed_limit_max": max(speed_values) if speed_values else None,
                },
                "SLF": {
                    "traffic_signals_count": int(len(ts_slice)),
                    "crossing_count": int(len(crossings_slice)),
                    "turn_lanes_present_ratio": turn_lanes_ratio,
                },
                "ICF": {
                    "intersection_count": int(len(node_intersections)),
                    "roundabout_count": int(len(roundabout_slice)),
                    "bridge_edge_count": bridge_count,
                    "tunnel_edge_count": tunnel_count,
                    "bus_stop_count": int(len(bus_slice)),
                    "parking_count": int(len(parking_slice)),
                },
                "SEF": {
                    "building_area_m2": _sum_area_within(building_slice, slice_geom),
                    "landuse_area_by_class_m2": landuse_area_by_class,
                    "green_area_m2": _sum_area_within(green_slice, slice_geom),
                    "footway_total_length_m": _sum_line_length_within(footway_slice, slice_geom),
                    "commercial_poi_count": int(len(commercial_slice)),
                },
                "raw_r2": raw_r2_values,
            }
        )

    result = {
        "meta": {
            "slicing_mode": slicing_mode,
            "grid_size_m": grid_size_m,
            "overlap_m": overlap_m,
            "anchor": anchor,
            "buffer_m": buffer_m,
            "network_type": network_type,
            "roi_geometry": mapping(roi_4326),
            "slice_count": len(slices_out),
        },
        "slices": slices_out,
    }

    result = _serialize(result)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result

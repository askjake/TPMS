"""Utility functions for TPMS Tracker"""

import numpy as np
from typing import List, Tuple
import folium
from streamlit_folium import folium_static


def create_encounter_map(encounters: List[dict]) -> folium.Map:
    """Create a map showing encounter locations"""
    if not encounters:
        return None

    # Filter encounters with location data
    located_encounters = [e for e in encounters if e.get('latitude') and e.get('longitude')]

    if not located_encounters:
        return None

    # Calculate center
    avg_lat = np.mean([e['latitude'] for e in located_encounters])
    avg_lon = np.mean([e['longitude'] for e in located_encounters])

    # Create map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

    # Add markers
    for encounter in located_encounters:
        folium.CircleMarker(
            location=[encounter['latitude'], encounter['longitude']],
            radius=5,
            popup=f"Encounter at {encounter['timestamp']}",
            color='blue',
            fill=True
        ).add_to(m)

    return m


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

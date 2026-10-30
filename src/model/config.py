"""Canonical configuration contract for the pure-Python flood model."""

from __future__ import annotations

from typing import Any, Dict, List


TABULAR_FEATURES: List[str] = [
    "precipitation_1h",
    "precipitation_3h",
    "temperature",
    "relative_humidity",
    "station_pressure",
    "wind_speed",
]
TERRAIN_FEATURES: List[str] = [
    "relative_height_norm",
    "slope_degrees",
    "aspect_degrees",
    "flow_accumulation_log",
    "elevated_terrain_indicator",
]


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate the checked-in model schema before pipeline startup."""
    if config.get("model_name") != "MultimodalFlashFloodNet":
        raise ValueError("model config model_name must be MultimodalFlashFloodNet")
    tabular = config.get("tabular_branch")
    terrain = config.get("terrain_branch")
    satellite = config.get("satellite_branch")
    if not isinstance(tabular, dict) or not isinstance(terrain, dict) or not isinstance(satellite, dict):
        raise ValueError("model config must define satellite_branch, tabular_branch, and terrain_branch")

    configured_tabular = tabular.get("features")
    if configured_tabular != TABULAR_FEATURES:
        raise ValueError(
            "model config tabular_branch.features does not match the dataset/model schema: "
            f"expected {TABULAR_FEATURES}, got {configured_tabular}"
        )
    if tabular.get("input_dim") != len(TABULAR_FEATURES):
        raise ValueError("model config tabular_branch.input_dim must match the six canonical weather features")

    if terrain.get("features") != TERRAIN_FEATURES:
        raise ValueError(
            "model config terrain_branch.features does not match the dataset/model schema: "
            f"expected {TERRAIN_FEATURES}, got {terrain.get('features')}"
        )
    if terrain.get("input_dim") != len(TERRAIN_FEATURES):
        raise ValueError("model config terrain_branch.input_dim must match the five canonical terrain features")

    if satellite.get("frame_encoder_type") != "pure_python_spatial_pooling":
        raise ValueError("model config satellite_branch.frame_encoder_type must be pure_python_spatial_pooling")
    if satellite.get("temporal_encoder_type") != "masked_attention":
        raise ValueError("model config satellite_branch.temporal_encoder_type must be masked_attention")
    if satellite.get("frame_embedding_dim") != 32 or satellite.get("temporal_hidden_dim") != 32:
        raise ValueError("model config satellite embedding dimensions must match the pure-Python model (32)")
    if tabular.get("sequence_length") != 6 or tabular.get("rnn_num_layers") != 1:
        raise ValueError("model config tabular sequence settings must match the pure-Python model")
    if tabular.get("rnn_hidden_dim") != 24:
        raise ValueError("model config tabular rnn_hidden_dim must match the pure-Python model (24)")
    if terrain.get("mlp_hidden_dims") != [16, 16]:
        raise ValueError("model config terrain mlp_hidden_dims must match the pure-Python model ([16, 16])")
    if config.get("fusion_head", {}).get("fusion_hidden_dims") != [32]:
        raise ValueError("model config fusion_head.fusion_hidden_dims must match the pure-Python model ([32])")

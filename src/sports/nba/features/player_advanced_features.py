"""Stub: player advanced features (pendiente de implementación)."""

import pandas as pd


def build_player_advanced_history(*args, **kwargs):
    return {}


def get_game_player_advanced(*args, **kwargs):
    return {}


def add_player_advanced_to_frame(frame, features_list):
    if not features_list or not any(features_list):
        return frame
    df = pd.DataFrame(features_list, index=frame.index)
    return pd.concat([frame, df], axis=1)

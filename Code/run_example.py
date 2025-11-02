# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 16:13:51 2025

@author: james
"""

import numpy as np
import pandas as pd

import feature_analysis

def run_example():
    """
    Run a toy example with a good predictor.

    Outputs a score with statistics and graphs to show performance.

    Returns
    -------
    None.

    """
    TIME_PERIOD = 5 #how far ahead to predict

    # Create a constant upward trend
    n = 500
    trend = np.linspace(100, 200, n)  # linear trend from 0 to 100

    # Add a wave pattern (sine wave)
    wave = 5 * np.sin(np.linspace(0, 10 * np.pi, n))  # 5 is amplitude

    # Combine trend and wave
    data = trend + wave
    raw_target = pd.DataFrame({'value': data})
    target = raw_target.pct_change(TIME_PERIOD).shift(-TIME_PERIOD).fillna(0)

    good_prediction = target.map(lambda x: 2 if x > 0 else (-2 if x < 0 else 0))

    # Bad prediction: random values in same range as target
    min_val, max_val = target.min(), target.max()
    bad_prediction = pd.DataFrame(np.random.uniform(low=min_val, high=max_val, size=n))

    feature_analysis.score_feature_against_target(good_prediction, target, raw_target)

if __name__ == "__main__":
    run_example()

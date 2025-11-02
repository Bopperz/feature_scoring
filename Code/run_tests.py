"""Test scripts for whole repository"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import unittest
from unittest.mock import patch

import feature_analysis
from data_analysis import spearman_rank

patch('feature_analysis.plt.show')  # disable plt.show()

class Testscore_feature_against_target(unittest.TestCase):
    def setUp(self):
        TIME_PERIOD = 5 #how far ahead to predict

        # Suppress graphs
        self.original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

        # Create a constant upward trend
        n = 500
        trend = np.linspace(100, 200, n)  # linear trend from 0 to 100

        # Add a wave pattern (sine wave)
        wave = 5 * np.sin(np.linspace(0, 10 * np.pi, n))  # 5 is amplitude

        # Combine trend and wave
        data = trend + wave
        self.raw_target = pd.DataFrame({'value': data})
        self.target = self.raw_target.pct_change(TIME_PERIOD).shift(-TIME_PERIOD).fillna(0)

        self.good_prediction = self.target.map(lambda x: 2 if x > 0 else (-2 if x < 0 else 0))

        # Bad prediction: random values in same range as target
        min_val, max_val = self.target.min(), self.target.max()
        self.bad_prediction = pd.DataFrame(np.random.uniform(low=min_val, high=max_val, size=n))

    def tearDown(self):
        # Restore the original backend
        matplotlib.use(self.original_backend)

    def test_with_all_variables(self):
        feature = self.good_prediction
        target = self.target
        raw_target = self.raw_target
        feature_analysis.score_feature_against_target(feature, target, raw_target)

    def test_with_mandatory_variables(self):
        feature = self.good_prediction
        target = self.target
        feature_analysis.score_feature_against_target(feature, target)

    def test_bad_feature_with_mandatory_variables(self):
        feature = self.bad_prediction
        target = self.target
        feature_analysis.score_feature_against_target(feature, target)

    def test_send_target_as_pandas_series(self):
        feature = self.bad_prediction
        target = self.target.squeeze()
        with self.assertRaises(TypeError):
            feature_analysis.score_feature_against_target(feature, target)

    def test_send_negative_asset(self):
        feature = self.bad_prediction
        target = self.target  * -1
        raw_target = self.raw_target * -1
        feature_analysis.score_feature_against_target(feature, target, raw_target)

class Testplot_prediction_at_time_interval(unittest.TestCase):
    def setUp(self):

        TIME_PERIOD = 5 #how far ahead to predict

        # Suppress graphs
        self.original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

        # Create a constant upward trend
        n = 500
        trend = np.linspace(100, 200, n)  # linear trend from 0 to 100

        # Add a wave pattern (sine wave)
        wave = 5 * np.sin(np.linspace(0, 10 * np.pi, n))  # 5 is amplitude

        # Combine trend and wave
        data = trend + wave
        self.raw_target = pd.DataFrame({'value': data})
        self.target = self.raw_target.pct_change(TIME_PERIOD).shift(-TIME_PERIOD).fillna(0)

        self.good_prediction = self.target.map(lambda x: 2 if x > 0 else (-2 if x < 0 else 0))

        # Bad prediction: random values in same range as target
        min_val, max_val = self.target.min(), self.target.max()
        self.bad_prediction = pd.DataFrame(np.random.uniform(low=min_val, high=max_val, size=n))

    def tearDown(self):
        # Restore the original backend
        matplotlib.use(self.original_backend)

    def test_plot_prediction_at_time_interval_working(self):

        feature = self.good_prediction
        raw_target_data = self.raw_target
        end_period = 5
        feature_analysis._plot_prediction_at_time_interval(feature, raw_target_data, end_period)

    def test_plot_prediction_at_time_interval_end_period_before_start(self):

        feature = self.good_prediction
        raw_target_data = self.raw_target
        end_period = 5
        start_period = 10
        with self.assertRaises(AssertionError):
            feature_analysis._plot_prediction_at_time_interval(feature, raw_target_data, end_period, start_period)


class TestSpearmanRank(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 6, 7, 8, 7]
        })
        self.y = pd.Series([1, 2, 3, 4, 5])

    def test_spearman_rank_basic(self):
        result = spearman_rank(self.X, self.y)
        expected_rho, _ = spearmanr(self.X, self.y)
        expected_rho = expected_rho[:-1, -1]
        expected = pd.DataFrame(expected_rho, index=self.X.columns, columns=['spearman_rank_score'])
        pd.testing.assert_frame_equal(result, expected)

    def test_spearman_rank_abs(self):
        result = spearman_rank(self.X, self.y, rtn_abs=True)
        expected_rho, _ = spearmanr(self.X, self.y)
        expected_rho = np.abs(expected_rho[:-1, -1])
        expected = pd.DataFrame(expected_rho, index=self.X.columns, columns=['spearman_rank_score'])
        pd.testing.assert_frame_equal(result, expected)

    def test_spearman_rank_single_feature(self):
        X_single = self.X[['feature1']]
        result = spearman_rank(X_single, self.y)
        expected_rho, _ = spearmanr(X_single, self.y)
        expected = pd.DataFrame([expected_rho], index=X_single.columns, columns=['spearman_rank_score'])
        pd.testing.assert_frame_equal(result, expected)

    def test_assertion_error(self):
        y_mismatch = pd.Series([1, 2, 3])
        with self.assertRaises(AssertionError):
            spearman_rank(self.X, y_mismatch)

if __name__ == "__main__":
    unittest.main()

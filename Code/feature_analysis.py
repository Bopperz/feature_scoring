"""
Score a feature against the target and check if the feature has predictive
power at set interval returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
from skrebate import MultiSURF

import data_analysis

def score_feature_against_target(feature: pd.DataFrame,
                                 target: pd.DataFrame,
                                 raw_target_data: pd.DataFrame = pd.DataFrame):
    """
    For a given feature, create a variety of metrics to score against the target.

    Parameters
    ----------
    feature : pd.DataFrame
        Table with feature to score.
    target : pd.DataFrame
        Table with target to score against.
    raw_target_data : pd.DataFrame, optional
        Table of raw target data to score against different time horizons.
        The default is an empty dataframe.

    Returns
    -------
    None.

    """
    feature.plot()
    plt.title('Feature over time')
    plt.show()

    stationary_result = data_analysis.check_dataframe_stationary(feature)
    if stationary_result.iloc[0,0] is False:
        print('Warning: Factor is not stationary!')

    _plot_returns_from_factor(feature, target)
    _plot_scatter_feature_against_target(feature, target)

    if raw_target_data.empty is False:
        for end_period in [1,5,10,20]:
            _plot_prediction_at_time_interval(feature, raw_target_data, end_period)

def _plot_returns_from_factor(feature, target):
    """
    Create a graph of the performance of the feature.

    Parameters
    ----------
    feature : pd.DataFrame
        Table with feature to score.
    target : pd.DataFrame
        Table with target to score against.

    Returns
    -------
    None.

    """
    if not isinstance(target, pd.DataFrame):
        raise TypeError('Expected the target to be a pandas dataframe.')
    pnl = feature.values * target
    pnl.cumsum().plot(figsize=(10, 6))
    plt.title('Factor Cumulative Returns')
    plt.ylabel('Returns')
    plt.show()

def _plot_scatter_feature_against_target(feature, target):
    """
    Create graph showing the feature against the target.

    Parameters
    ----------
    feature : pd.DataFrame
        Table with feature to score.
    target : pd.DataFrame
        Table with target to score against.

    Returns
    -------
    None.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(feature, target, s=10, marker='.')
    plt.colorbar()
    plt.title('Scatter graph with target')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.show()

def _plot_prediction_at_time_interval(feature, raw_target_data, end_period, start_period=0):
    #Check for alpha against various time offsets

    def _calc_target_at_different_time_scales(asset_data, feature, end_period, start_period=0):
        """
        Calculate the returns for a given return period.

        Parameters
        ----------
        asset_data : pd.DataFrame
            Raw data to calculate returns from.
        feature : pd.DataFrame
            The new feature.
        end_period : int
            Number of timeperiods ahead to end the return.
        start_period : int, optional
            Number of timeperiods ahead to start the return. The default is 0.

        Returns
        -------
        asset_rtns : pd.DataFrame
            Table of asset returns.

        """
        num_periods = end_period - start_period
        assert num_periods > 0, ('You cannot calculate returns into the past, the end_period '
                                 'must less than the start period.')
        asset_rtns = asset_data.pct_change(num_periods).shift(-end_period).replace([np.inf, -np.inf], 0)
        asset_rtns = asset_rtns.replace([np.inf, -np.inf], np.nan).fillna(0)
        asset_rtns.columns = ['asset_rtns']
        asset_rtns = asset_rtns.loc[feature.index] #only include returns where we have a signal.
        return asset_rtns


    def _create_kde_charts_of_prediction(feature, asset_rtns):
        """
        Create a kde plot showing how the predictions are graphically seperate.

        Parameters
        ----------
        feature : pd.DataFrame
            Table with the new feature.
        asset_rtns : pd.DataFrame
            Table of asset returns to score against.

        Returns
        -------
        None.

        """
        factor_binary = feature.squeeze().apply(lambda x: 1 if x>0 else 0)
        factor_binary.columns = ['feature_binary']
        factor_binary = factor_binary.to_frame().join(asset_rtns, how='left')
        factor_binary.columns = ['feature_binary', 'asset_returns']
        cht = sns.FacetGrid(data=factor_binary, hue='feature_binary', height=4)
        cht = cht.map(sns.kdeplot, 'asset_returns')
        cht.add_legend(title='Class Prediction')
        plt.show()


    def _create_statistics_table(feature, asset_rtns):
        """
        Score the feature against the asset returns.

        Returns a table with mutual information, chi_squared and f_score.

        Parameters
        ----------
        feature : pd.DataFrame
            Table with the new feature.
        asset_rtns : pd.DataFrame
            Table of asset returns to score against.

        Returns
        -------
        df : pd.DataFrame
            Table with mutual information, chi_squared and f_score..

        """
        def run_multisurf(x, y):

            fs = MultiSURF(n_jobs=-1, n_features_to_select=1)
            fs.fit(x.values, y.values)
            df = pd.DataFrame(fs.feature_importances_, index=x.columns, columns=['Multi_Surf'])
            return df

        y = asset_rtns
        y_class = y.squeeze().apply(lambda x: 1 if x>0 else 0)
        x = feature
        x_class = x.iloc[:,0].apply(lambda x: 1 if x>0 else 0)

        mi_reg = data_analysis.mutual_info_regression(x, y)
        mi_class = data_analysis.mutual_info_class(x, y_class)
        chi_sq = data_analysis.chi2(x_class.to_frame(), y_class.to_frame())
        sr = data_analysis.spearman_rank(x, y)
        f_score = pd.DataFrame(f_regression(x.values, np.ravel(y.values))[0],
                               index=x.columns,
                               columns=['F_Score'])
        out = pd.concat([mi_reg.T, mi_class.T, chi_sq.T, sr.T, f_score.T])
        print(out.round(3))


    print('\n')
    print(f'Period: {start_period}:{end_period}')
    asset_rtns = _calc_target_at_different_time_scales(raw_target_data,
                                                       feature,
                                                       end_period,
                                                       start_period)
    _create_kde_charts_of_prediction(feature, asset_rtns)
    _create_statistics_table(feature, asset_rtns)

"""Functions to score against indervidual metrics."""

import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression as mi_reg
from sklearn.feature_selection import chi2 as sk_chi2
from statsmodels.tsa.stattools import adfuller

def check_dataframe_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if each column in a dataframe is stationary using adfuller test.

    Scored against 1% confidence level.

    Parameters
    ----------
    df : pd.DataFrame
        Table of data, where each column represents a feature and each row is a sample.

    Returns
    -------
    out : pd.DataFrame
        Table of data with adfuller result at 1% confidence.

    """
    out = pd.DataFrame(None, index=df.columns, columns=['Stationary', 'pvalue'])

    for col in df.columns:
        result = adfuller(df[col].values)
        out.loc[col, 'pvalue'] = result[1]
        if result[0] < result[4]['1%']:
            out.loc[col, 'Stationary'] = True
        else:
            out.loc[col, 'Stationary'] = False

    return out

def mutual_info_regression(df_var, df_target):
    """
    Create a dataframe of Mutual Info scores.

    Parameters
    ----------
    df_var : DataFrame
        Features to compare to target.
    df_target : DataFrame
        Target.

    Returns
    -------
    out : DataFrame
        Sorted list of Mutual Info scores for each feature.

    """
    x = df_var.copy()  # independent columns
    y = df_target.to_numpy().ravel()  # target column i.e price range

    out = mi_reg(x, y, n_neighbors=3, random_state=42)

    out = pd.DataFrame(out, index=df_var.columns, columns=['mutual_info_reg_score'])
    out = out.sort_values(by=['mutual_info_reg_score'], ascending=False)

    return out

def mutual_info_class(x, y):
    """
    Create a dataframe of Mutual Info scores.

    Parameters
    ----------
    df_var : DataFrame
        Features to compare to target.
    df_target : DataFrame
        Target.

    Returns
    -------
    out : DataFrame
        Sorted list of Mutual Info scores for each feature.
        Index as feature name, column as score.

    """
    y = y.to_numpy().ravel()
    x = x.select_dtypes(include='number')
    out = mutual_info_classif(x, y, random_state=42)
    out = pd.DataFrame(out, index=x.columns, columns=['mutual_info_class_score'])
    out = out.sort_values(by=['mutual_info_class_score'], ascending=False)

    return out

def chi2(df_var, df_target):
    """
    Run chi2 test on entire dataframe.

    When two features are independent, the observed count is close to the expected count,
    thus we will have smaller Chi-Square value. So high Chi-Square value indicates that
    the hypothesis of independence is incorrect. In simple words, higher the Chi-Square
    value the feature is more dependent on the response and it can be selected
    for model training.
    aim for p_val <0.05 and a high chi_stat.

    Parameters
    ----------
    df_var : DataFrame
        Features to compare to target.
    df_target : DataFrame
        Target.

    Returns
    -------
    out : pd.DataFrame
        Sorted table of chi2 scores.
        Index as feature name, column as score.

    """
    x = df_var.copy()  # independent columns
    y = df_target.copy()  # target column i.e price range

    chi_stat, p_val = sk_chi2(x, y)

    out = pd.DataFrame(chi_stat, index=df_var.columns, columns=['chi2'])
    out = out.sort_values(by=['chi2'], ascending=False)
    return out

def spearman_rank(x, y, rtn_abs=False):
    """
    Spearman rank coeff, allows for non-linear relationships to target.

    Parameters
    ----------
    df_var : DataFrame
        Features to compare to target.
    df_target : DataFrame
        Target.

    Returns
    -------
    out : DataFrame
        Sorted list of Spearman Rank scores for each feature.

    """
    assert x.shape[0] == y.shape[0]

    rho, p_value = spearmanr(a=x, b=y)
    if x.shape[1] > 1:
        rho = rho[:-1, -1]

    out = pd.DataFrame(rho, index=x.columns, columns=['spearman_rank_score'])
    if rtn_abs is True:
        out = out.abs()
    return out

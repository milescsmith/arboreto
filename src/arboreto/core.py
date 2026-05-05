"""
Core functional building blocks, composed in a Dask graph for distributed computation.
"""

import logging
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from dask import delayed
from dask.dataframe import from_delayed
from dask.dataframe.utils import make_meta
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from ._docs import (
    doc_algo_args,
    doc_regressor_arg,
    doc_regressor_kwargs,
    doc_tf_gene_args,
    doc_tf_matrix_args,
)
from .utils import _doc_params

logger = logging.getLogger(__package__)

DEMON_SEED = 666
ANGEL_SEED = 777
EARLY_STOP_WINDOW_LENGTH = 25

# scikit-learn random forest regressor
RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}

# scikit-learn extra-trees regressor
ET_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}

# scikit-learn gradient boosting regressor
GBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 500, "max_features": 0.1}

# scikit-learn stochastic gradient boosting regressor
SGBM_KWARGS = {
    "learning_rate": 0.01,
    "n_estimators": 5000,  # can be arbitrarily large
    "max_features": 0.1,
    "subsample": 0.9,
}

class SklearnRegressorFactory(StrEnum):
    RF = "RandomForestRegressor"
    ET = "ExtraTreesRegressor"
    GBM = "GradientBoostingRegressor"
    XGB = "XGBRegressor"

type Regressor = RandomForestRegressor | ExtraTreesRegressor | GradientBoostingRegressor | XGBRegressor
# @_doc_params(regressor_arg=doc_regressor_arg)
# def is_sklearn_regressor(regressor_type) -> bool:
#     """
#     Parameters
#     ----------
#     {regressor_arg}

#     Returns
#     -------
#     bool :
#         value indicating whether the regressor type is a scikit-learn regressor, following the scikit-learn API.
#     """
#     return regressor_type.upper() in SklearnRegressorFactory


# @_doc_params(regressor_arg=doc_regressor_arg)
# def is_xgboost_regressor(regressor_type) -> bool:
#     """
#     Parameters
#     ----------
#     {regressor_arg}

#     Returns
#     -------
#     bool :
#         value indicating whether the regressor type is the xgboost regressor.
#     """
#     return regressor_type.upper() == "XGB"


@_doc_params(
    regressor_arg=doc_regressor_arg,
    regressor_kwargs=doc_regressor_kwargs
    )
def is_oob_heuristic_supported(
    regressor_type: SklearnRegressorFactory,
    regressor_kwargs: dict[str, Any]
    ):
    """
    Parameters
    ----------
    {regressor_arg}
    {regressor_kwargs}

    Returns
    -------
    bool :
        whether early stopping heuristic based on out-of-bag improvement is supported.

    """
    return regressor_type == SklearnRegressorFactory.GBM and "subsample" in regressor_kwargs and regressor_kwargs["subsample"] < 1.0


@_doc_params(
    common_algo_params=doc_algo_args
)
def to_tf_matrix(
    expression_matrix: npt.ArrayLike,
    gene_names: list[str],
    tf_names: str | list[str] | Literal["all"] = "all",
):
    """
    Parameters
    ----------
    {common_algo_params}

    Returns
    -------
    :class:`np.ndarray` :
        matrix representing the predictor matrix for the regressions.
    list[str] :
        The gene names corresponding to the columns in the predictor matrix.
    """

    tuples = [(index, gene) for index, gene in enumerate(gene_names) if gene in tf_names]

    tf_indices = [t[0] for t in tuples]
    tf_matrix_names = [t[1] for t in tuples]

    return expression_matrix[:, tf_indices], tf_matrix_names

@_doc_params(
    regressor_arg=doc_regressor_arg,
    regressor_kwargs=doc_regressor_kwargs
)
def fit_model(
    regressor_type: SklearnRegressorFactory,
    regressor_kwargs: dict[str, Any],
    tf_matrix: npt.ArrayLike,
    target_gene_expression: npt.ArrayLike,
    early_stop_window_length: int=EARLY_STOP_WINDOW_LENGTH,
    seed: int=DEMON_SEED,
) -> Regressor:
    """
    Parameters
    ----------
    {regressor_arg}
    {regressor_kwargs}
    tf_matrix : :class:`npt.ArrayLike`
        the predictor matrix (transcription factor matrix) as a numpy array.
    target_gene_expression : :class:`npt.ArrayLike`
        the target (y) gene expression to predict in function of the tf_matrix (X).
    early_stop_window_length : int, default = 25
        window length of the early stopping monitor.
    seed : int, default = 666
        random seed for the regressors.

    Returns
    -------
    :class:`Regressor` :
        A trained regression model.
    """

    if isinstance(target_gene_expression, sp.sparse.spmatrix):
        target_gene_expression = target_gene_expression.A.flatten()

    if tf_matrix.shape[0] != target_gene_expression.shape[0]:
        msg = "`tf_matrix` and `target_gene_expression` do not have the same shape."
        raise ValueError(msg)

    match regressor_type:
        case SklearnRegressorFactory.RF:
            regressor = RandomForestRegressor(random_state=seed, **regressor_kwargs)
        case SklearnRegressorFactory.ET:
            regressor = ExtraTreesRegressor(random_state=seed, **regressor_kwargs)
        case SklearnRegressorFactory.GBM:
            regressor = GradientBoostingRegressor(random_state=seed, **regressor_kwargs)
        case SklearnRegressorFactory.XGB:
            regressor = XGBRegressor(random_state=seed, **regressor_kwargs)
        case _:
            msg = f"{regressor_type!s} is unknown and unsupported."
            raise ValueError(msg)

    with_early_stopping = is_oob_heuristic_supported(regressor_type, regressor_kwargs)

    if with_early_stopping:
        regressor.fit(tf_matrix, target_gene_expression, monitor=EarlyStopMonitor(early_stop_window_length))
    else:
        regressor.fit(tf_matrix, target_gene_expression)

    return regressor

@_doc_params(regressor_arg=doc_regressor_arg, regressor_kwargs=doc_regressor_kwargs)
def to_feature_importances(regressor_type: SklearnRegressorFactory, regressor_kwargs: dict[str, Any], trained_regressor: Regressor) -> np.ndarray:
    """Motivation: when the out-of-bag improvement heuristic is used, we cancel the effect of normalization by dividing
    by the number of trees in the regression ensemble by multiplying again by the number of trees used.

    This enables prioritizing links that were inferred in a regression where lots of

    Parameters
    ----------
    {regressor_arg}
    {regressor_kwargs}
    trained_regressor : :class:`Regressor`
        the trained model from which to extract the feature importances.

    Returns
    -------
    :class:`np.ndarray` :
        the feature importances inferred from the trained model.
    """

    if is_oob_heuristic_supported(regressor_type, regressor_kwargs):
        n_estimators = len(trained_regressor.estimators_)

        denormalized_importances = trained_regressor.feature_importances_ * n_estimators

        return denormalized_importances
    else:
        return trained_regressor.feature_importances_


def to_meta_df(trained_regressor: Regressor, target_gene_name: str):
    """
    Parameters
    ----------
    trained_regressor : :class:`Regressor`
       the trained model from which to extract the meta information.
    target_gene_name : str
        the name of the target gene.

    Returns
    -------
    :class:`pd.DataFrame` :
       dataframe containing side information about the regression.
    """
    n_estimators = len(trained_regressor.estimators_)

    return pd.DataFrame({"target": [target_gene_name], "n_estimators": [n_estimators]})


@_doc_params(regressor_arg=doc_regressor_arg, regressor_kwargs=doc_regressor_kwargs, tf_gene_args=doc_tf_gene_args)
def to_links_df(
    regressor_type: SklearnRegressorFactory,
    regressor_kwargs: dict[str, Any],
    trained_regressor: Regressor,
    tf_matrix_gene_names: list[str],
    target_gene_name: str
    ) -> pd.DataFrame:
    """
    Parameters
    ----------
    {regressor_arg}
    {regressor_kwargs}
    trained_regressor : :class:`Regressor`
        the trained model from which to extract the feature importances.
    {tf_gene_args}

    Returns
    -------
    :class:`pd.DataFrame` :
        dataframe with columns `['TF', 'target', 'importance']` representing inferred regulatory links and their
        connection strength.
    """
    # feature_importances = trained_regressor.feature_importances_
    feature_importances = to_feature_importances(regressor_type, regressor_kwargs, trained_regressor)

    links_df = pd.DataFrame({"TF": tf_matrix_gene_names, "importance": feature_importances})
    links_df["target"] = target_gene_name

    clean_links_df = links_df[links_df.importance > 0].sort_values(by="importance", ascending=False)

    return clean_links_df[["TF", "target", "importance"]]


@_doc_params(tf_args=doc_tf_matrix_args)
def clean(
    tf_matrix: npt.ArrayLike | sp.sparse.spmatrix,
    tf_matrix_gene_names: list[str],
    target_gene_name: str
    ) -> tuple[npt.ArrayLike | sp.sparse.spmatrix, list[str]]:
    """
    Parameters
    ----------
    {tf_args}

    Returns
    -------
    matrix : :class:`npt.ArrayLike` | :class:`sp.sparse.spmatrix`
        the cleaned transcription factor matrix, equal to the specified one but with the target gene column removed
    names : list[str]
        the cleaned list of transcription factor names, equal to the specified one but with the target gene name removed
    """

    if target_gene_name not in tf_matrix_gene_names:
        clean_tf_matrix = tf_matrix
    else:
        ix = tf_matrix_gene_names.index(target_gene_name)
        if isinstance(tf_matrix, sp.sparse.spmatrix):
            clean_tf_matrix = sp.sparse.hstack([tf_matrix[:, :ix], tf_matrix[:, ix + 1 :]])
        else:
            clean_tf_matrix = np.delete(tf_matrix, ix, 1)

    clean_tf_names = [tf for tf in tf_matrix_gene_names if tf != target_gene_name]

    assert clean_tf_matrix.shape[1] == len(clean_tf_names)  # sanity check

    return clean_tf_matrix, clean_tf_names


# I do not think we need to work around a 10 year old bug that was fixed in scikit-learn 0.2.0, 8 fucking years ago.
# and if we do revisit this, just use the `tenacity`/`tenacity-rs`, `pyresilience`, or `retrying` libaries.
# def retry(fn, max_retries=10, warning_msg=None, fallback_result=None):
#     """Minimalistic retry strategy to compensate for failures probably caused by a thread-safety bug in scikit-learn:
#     * https://github.com/scikit-learn/scikit-learn/issues/2755
#     * https://github.com/scikit-learn/scikit-learn/issues/7346

#     Parameters
#     ----------
#     :param fn: the function to retry.
#     :param max_retries: the maximum number of retries to attempt.
#     :param warning_msg: a warning message to display when an attempt fails.
#     :param fallback_result: result to return when all attempts fail.

#     Returns
#     -------
#     :return: Returns the result of fn if one attempt succeeds, else return fallback_result.
#     """
#     nr_retries = 0

#     result = fallback_result

#     for attempt in range(max_retries):
#         try:
#             result = fn()
#         except Exception as cause:
#             nr_retries += 1

#             msg_head = "" if warning_msg is None else repr(warning_msg) + " "
#             msg_tail = "Retry ({1}/{2}). Failure caused by {0}.".format(repr(cause), nr_retries, max_retries)

#             logger.warning(msg_head + msg_tail)
#         else:
#             break

#     return result


@_doc_params(regressor_arg=doc_regressor_arg, regressor_kwargs=doc_regressor_kwargs, tf_args=doc_tf_matrix_args)
def infer_partial_network(
    regressor_type: SklearnRegressorFactory,
    regressor_kwargs: dict[str, Any],
    tf_matrix: npt.ArrayLike | sp.sparse.spmatrix,
    tf_matrix_gene_names: list[str],
    target_gene_name: str,
    target_gene_expression: npt.ArrayLike,
    include_meta: bool=False,
    early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
    seed=DEMON_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """Ties together regressor model training with regulatory links and meta data extraction.

    Parameters
    ----------
    {regressor_arg}
    {regressor_kwargs}
    {tf_args}
    target_gene_expression : npt.ArrayLike
        Expression profile of the target gene.
    include_meta : bool, default=False
        Whether to also return meta information/
    early_stop_window_length : int, default = 25
        Window length of the early stopping monitor.
    seed : int, default = 666
        Random seed for the regressors.

    Returns
    -------
    :class:`pd.DataFrame` :
        DataFrame containing inferred regulatory links and their connection strength.
    :class:`pd.DataFrame` :
        if include_meta is True, a dataframe containing meta information regarding the trained regression model.
    """
    clean_tf_matrix, clean_tf_matrix_gene_names = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

    # special case in which only a single TF is passed and the target gene
    # here is the same as the TF (clean_tf_matrix is empty after cleaning):
    if clean_tf_matrix.size == 0:
        msg = f"Cleaned TF matrix is empty, skipping inference of target {target_gene_name}."
        raise ValueError(msg)

    try:
        trained_regressor = fit_model(
            regressor_type,
            regressor_kwargs,
            clean_tf_matrix,
            target_gene_expression,
            early_stop_window_length,
            seed,
        )
    except ValueError as e:
        msg = f"Regression for target gene {target_gene_name} failed. Cause {e!s}."
        raise ValueError(msg) from e

    links_df = to_links_df(
        regressor_type, regressor_kwargs, trained_regressor, clean_tf_matrix_gene_names, target_gene_name
    )

    if include_meta:
        meta_df = to_meta_df(trained_regressor, target_gene_name)
        return links_df, meta_df
    else:
        return links_df


# TODO: can this be replaced by functionality already present in pandas?
def target_gene_indices(gene_names: list[str], target_genes: int | list[str] | Literal["all"]) -> list[int]:
    """
    Parameters
    ----------
    gene_names : list[str]
        List of gene names.
    target_genes : int | list[str] | Literal["all"]
        Either an integer (the top n), 'all', or a collection (subset of gene_names).

    Returns
    -------
    list[int] :
        The column indices of the target genes in the expression_matrix.
    """

    if isinstance(target_genes, list) and len(target_genes) == 0:
        return []

    if isinstance(target_genes, str) and target_genes.upper() == "ALL":
        return list(range(len(gene_names)))

    elif isinstance(target_genes, int):
        top_n = target_genes
        assert top_n > 0

        return list(range(min(top_n, len(gene_names))))

    elif isinstance(target_genes, list):
        if not target_genes:  # target_genes is empty
            return target_genes
        elif all(isinstance(target_gene, str) for target_gene in target_genes):
            return [index for index, gene in enumerate(gene_names) if gene in target_genes]
        elif all(isinstance(target_gene, int) for target_gene in target_genes):
            return target_genes
        else:
            msg = "Mixed types in target genes."
            raise ValueError(msg)

    else:
        msg = "Unable to interpret target_genes."
        raise ValueError(msg)


_GRN_SCHEMA = make_meta({"TF": str, "target": str, "importance": float})
_META_SCHEMA = make_meta({"target": str, "n_estimators": int})

@_doc_params(regressor_arg=doc_regressor_arg, regressor_kwargs=doc_regressor_kwargs)
def create_graph(
    expression_matrix,
    gene_names,
    tf_names,
    regressor_type,
    regressor_kwargs,
    client,
    target_genes="all",
    limit=None,
    include_meta=False,
    early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
    repartition_multiplier=1,
    seed=DEMON_SEED,
):
    """Main API function. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    Parameters
    ----------
    expression_matrix : :class:`npt:.ArrayLike`
        Expression matrix to analyze, with observations as rows and genes as columns.
    gene_names : list[str]
        Entry corresponds to the expression_matrix column with same index.
    tf_names : list[str]
        list of transcription factor names. Should have a non-empty intersection with gene_names.
    {regressor_arg}
    {regressor_kwargs}
    client : :class:`dask.distributed.Client`
        Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    target_genes : int | list[str] | Literal["all"], default="all"
        gene_names.
    limit : int, optional
        optional number of top regulatory links to return. Default None.
    include_meta : bool, default=False
        Also return the meta information from the regressor.
    early_stop_window_length : int, default = 25
        Eindow length of the early stopping monitor.
    repartition_multiplier : int, default=1
        Multiplier for the number of partitions to repartition the resulting DataFrames to, relative to the number of workers.
    seed : int, default=666
        Random seed for the regressors.

    Returns
    -------
    :class:`
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """

    if not expression_matrix.shape[1] != len(gene_names):
        msg = "Number of columns in expression_matrix does not match the number of gene names."
        raise ValueError(msg)
    if client is None:
        msg = "client is required"
        raise ValueError(msg)

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_names)

    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    delayed_link_dfs = []  # collection of delayed link DataFrames
    delayed_meta_dfs = []  # collection of delayed meta DataFrame

    for target_gene_index in target_gene_indices(gene_names, target_genes):
        target_gene_name = delayed(gene_names[target_gene_index], pure=True)
        target_gene_expression = delayed(expression_matrix[:, target_gene_index], pure=True)

        if include_meta:
            delayed_link_df, delayed_meta_df = delayed(infer_partial_network, pure=True, nout=2)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                future_tf_matrix_gene_names,
                target_gene_name,
                target_gene_expression,
                include_meta,
                early_stop_window_length,
                seed,
            )

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)
                delayed_meta_dfs.append(delayed_meta_df)
        else:
            delayed_link_df = delayed(infer_partial_network, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                future_tf_matrix_gene_names,
                target_gene_name,
                target_gene_expression,
                include_meta,
                early_stop_window_length,
                seed,
            )

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # gather the DataFrames into one distributed DataFrame
    all_links_df = from_delayed(delayed_link_dfs, meta=_GRN_SCHEMA)

    # optionally limit the number of resulting regulatory links, descending by top importance
    if limit:
        maybe_limited_links_df = all_links_df.nlargest(limit, columns=["importance"])
    else:
        maybe_limited_links_df = all_links_df

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    if include_meta:
        all_meta_df = from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)
        return maybe_limited_links_df.repartition(npartitions=n_parts), all_meta_df.repartition(npartitions=n_parts)
    else:
        return maybe_limited_links_df.repartition(npartitions=n_parts)


class EarlyStopMonitor:
    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        """
        Parameters
        ----------
        window_length : int
            Length of the window over the out-of-bag errors.

        Returns
        -------
        """

        self.window_length = window_length

    def window_boundaries(self, current_round):
        """
        Parameters
        ----------
        current_round : int
            The current boosting round.

        Returns
        -------
        int :
            The low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(self, current_round, regressor, _):
        """Implementation of the GradientBoostingRegressor monitor function API.

        Parameters
        ----------
        current_round : int
            The current boosting round.
        regressor : :class:`Regressor`
         the regressor.
        _ : ignored.

        Returns
        -------
        bool :
            Whether the regressor should stop early
        """

        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo:hi]) < 0
        else:
            return False

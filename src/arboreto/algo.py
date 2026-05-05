"""
Top-level functions.
"""
# from enum import StrEnum
from typing import Literal

import distributed
import numpy.typing as npt
import pandas as pd
import scipy as sp
from loguru import logger

from ._docs import doc_algo_args, doc_client_args, doc_misc_args
from .core import EARLY_STOP_WINDOW_LENGTH, RF_KWARGS, SGBM_KWARGS, RegressorType, create_graph
from .logging import init_logger
from .utils import _doc_params


@_doc_params(
    common_algo_params=doc_algo_args,
    common_dask_args=doc_client_args,
    misc_args=doc_misc_args
    )
def grnboost2(
    expression_data: pd.DataFrame | npt.ArrayLike | sp.sparse.sparray,
    gene_names: list[str] | None = None,
    tf_names: str | Literal["all"] = "all",
    client_or_address: str | distributed.Client | Literal["local"] = "local",
    early_stop_window_length: int = EARLY_STOP_WINDOW_LENGTH,
    limit: int | None = None,
    seed: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Launch arboreto with [GRNBoost2] profile.

    Parameters
    ----------
    {common_algo_params}
    {common_dask_args}
    early_stop_window_length : int, default = 25
        Early stop window length.
    {misc_args}

    Returns
    -------
    :class:`pd.DataFrame` :
        DataFrame with columns 'TF', 'target', and 'importance' representing the inferred gene regulatory links.
    """

    return diy(
        expression_data=expression_data,
        regressor_type=RegressorType.GBM,
        regressor_kwargs=SGBM_KWARGS,
        gene_names=gene_names,
        tf_names=tf_names,
        client_or_address=client_or_address,
        early_stop_window_length=early_stop_window_length,
        limit=limit,
        seed=seed,
        verbose=verbose,
    )

@_doc_params(
    common_algo_params=doc_algo_args,
    common_dask_args=doc_client_args,
    misc_args=doc_misc_args
    )
def genie3(
    expression_data: pd.DataFrame | npt.ArrayLike | sp.sparse.sparray,
    gene_names: list[str] | None = None,
    tf_names: str | Literal["all"] = "all",
    client_or_address: str | distributed.Client | Literal["local"] = "local",
    limit: int | None = None,
    seed: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Launch arboreto with [GENIE3] profile.

    Parameters
    ----------
    {common_algo_params}
    {common_dask_args}
    {misc_args}

    Returns
    -------
    :class:`pd.DataFrame` :
        DataFrame with columns 'TF', 'target', and 'importance' representing the inferred gene regulatory links.
    """

    return diy(
        expression_data=expression_data,
        regressor_type=RegressorType.RF,
        regressor_kwargs=RF_KWARGS,
        gene_names=gene_names,
        tf_names=tf_names,
        client_or_address=client_or_address,
        limit=limit,
        seed=seed,
        verbose=verbose,
    )

@_doc_params(
    common_algo_params=doc_algo_args,
    common_dask_args=doc_client_args,
    misc_args=doc_misc_args
    )
def diy(
    expression_data: pd.DataFrame | npt.ArrayLike | sp.sparse.sparray,
    regressor_type : RegressorType,
    regressor_kwargs,
    gene_names: list[str] | None = None,
    tf_names: str | list[str] | Literal["all"] = "all",
    client_or_address: str | distributed.Client | Literal["local"] = "local",
    early_stop_window_length: int = EARLY_STOP_WINDOW_LENGTH,
    limit: int | None = None,
    seed: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    {common_algo_params}
    {common_dask_args}
    regressor_type : :class:`SklearnRegressorFactory`
        Case insensitive.
    regressor_kwargs : dict[str, Any]
        a dictionary of key-value pairs that configures the regressor.
    early_stop_window_length : int, default = 25
        Early stop window length.
    {misc_args}

    Returns
    -------
    :class:`pd.DataFrame` :
        DataFrame with columns 'TF', 'target', and 'importance' representing the inferred gene regulatory links.
    """
    if verbose > 0:
        init_logger(verbose=verbose)

    logger.debug("preparing dask client")
    client, shutdown_callback = _prepare_client(client_or_address)

    try:
        logger.debug("parsing input")
        expression_matrix, gene_names, tf_names = _prepare_input(expression_data, gene_names, tf_names)

        logger.debug("creating dask graph")
        graph = create_graph(
            expression_matrix,
            gene_names,
            tf_names,
            client=client,
            regressor_type=regressor_type,
            regressor_kwargs=regressor_kwargs,
            early_stop_window_length=early_stop_window_length,
            limit=limit,
            seed=seed,
        )

        if verbose:
            logger.debug(f"{graph.npartitions!s} partitions")
            logger.debug("computing dask graph")

        return client.compute(graph, sync=True).sort_values(by="importance", ascending=False)

    finally:
        shutdown_callback(verbose)

        logger.debug("finished")

@logger.catch(level="DEBUG")
@_doc_params(
    common_dask_args=doc_client_args
    )
def _prepare_client(client_or_address):
    """
    Parameters
    ----------
    {common_dask_args}

    Returns
    -------
    :class:`distributed.Client`
    :class:`Callable` :
        shutdown callback function.

    Raises
    ------
    ValueError :
        if no valid client input was provided.
    """

    if isinstance(client_or_address, str):
        client_or_address = client_or_address.lower()
    match client_or_address:
        case None if client_or_address.lower == "local":
            local_cluster = distributed.LocalCluster(diagnostics_port=None)
            cl = distributed.Client(local_cluster)

            def close_client_and_local_cluster(verbose=False):
                logger.debug("shutting down client and local cluster")
                cl.close()
                local_cluster.close()

            ret_val = (cl, close_client_and_local_cluster)
        case str():
            if client_or_address.lower() == "local":
                cl = distributed.Client()
            else:
                cl = distributed.Client(client_or_address)

            def close_client(verbose=False):
                logger.debug("shutting down client")
                cl.close()

            ret_val = (cl, close_client)
        case distributed.Client(_):
            def close_dummy(verbose=False):
                logger.debug("not shutting down client, client was created externally")

            ret_val = (client_or_address, close_dummy)
        case _:
            msg = f"Invalid client specified {client_or_address!s}"
            raise ValueError(msg)

    return ret_val


@_doc_params(
    misc_args=doc_misc_args
    )
def _prepare_input(
    expression_data: pd.DataFrame | npt.ArrayLike | sp.sparse.sparray,
    gene_names: list[str] | None = None,
    tf_names: list[str] | Literal["all"] = "all",
    ):
    """Wrangle the inputs into the correct formats.

    Parameters
    ----------
    {misc_args}

    Returns
    -------
    :class:`np.ndarray` | :class:`sp.sparse.sparray`
    list[str]
        List of gene names
    list[str]
        List of transcription factors
    """

    if isinstance(expression_data, pd.DataFrame):
        expression_matrix = expression_data.to_numpy()
        gene_names = list(expression_data.columns)
    else:
        expression_matrix = expression_data
        assert expression_matrix.shape[1] == len(gene_names)

    if tf_names == "all":
        tf_names = gene_names
    else:
        if len(tf_names) == 0:
            msg = "Specified tf_names is empty"
            raise ValueError(msg)

        if not set(gene_names).intersection(set(tf_names)):
            msg = "Intersection of gene_names and tf_names is empty."
            raise ValueError(msg)

    return expression_matrix, gene_names, tf_names

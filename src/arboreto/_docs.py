"""Common function parameter documentation

Idea stolen from scanpy. Place parameter docs here to save from
retyping them out and minimize potential for typos, mistakes, etc...
"""

doc_client_args = """\
client_or_address : str | Literal["local"]
        If `None` or 'local', a new Client(LocalCluster()) will be used to perform the computation.
        If an IP address, a new Client(address) will be used to perform the computation.
        If a Client instance, the specified Client instance will be used to perform the computation.
"""

doc_algo_args = """\
expression_data : :class:`pd.DataFrame` | :class:`npt.ArrayLike` | :class:`sp.sparse.sparray`
gene_names : list[str], optional
    List of gene names. Required when a (dense or sparse) matrix is passed as 'expression_data' instead of a DataFrame.
tf_names : list[str] | Literal["all"], optional
    List of transcription factors. If None or 'all', the list of gene_names will be used.\
"""

doc_misc_args = """\
limit : int, optional
    Number of top regulatory links to return.
seed : int, optional
    Random seed for the regressors.
verbose : int, default = 0
    Logging level, from 0 to 3 meaning "little logging" to "log everything".\
"""

doc_regressor_arg = """\
regressor_type : Literal["RF", "ET", "GBM", "XGB"]
    Type of regressor to use. Only :class:`sklearn.ensemble.RandomForestRegressor`,
    :class:`sklearn.ensemble.ExtraTreesRegressor`,
    :class:`sklearn.ensemble.GradientBoostingRegressor`
    are supported. While it may look like xgboost is supported, you'd be wrong about that.
"""

doc_regressor_kwargs = """\
regressor_kwargs : dict[str, Any]
    A dictionary of key-value pairs that configures the regressor.\
"""


doc_tf_gene_args = """\
tf_matrix_gene_names : list[str]
    Full list of transcription factor names, corresponding to the tf_matrix columns used to train the regression model.
target_gene_name : str
    Name of the target gene to infer the regulatory links for. \
"""

doc_tf_matrix_args = """\
tf_matrix : :class:`npt.ArrayLike` | :class:`sp.sparse.spmatrix`
    The full transcription factor matrix. Used for regression.
{doc_tf_gene_args}\
"""
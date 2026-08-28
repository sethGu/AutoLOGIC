import copy
import numpy as np
from .preprocessing import convert_categorical_to_integer_f
from typing import Any, Dict, Optional
import pandas as pd


def run_llm_code(code: str, df: pd.DataFrame, target_name: Optional[str] = None, convert_categorical_to_integer: Optional[bool] = True, fill_na: Optional[bool] = True) -> pd.DataFrame:

    try:
        loc = {}
        df = copy.deepcopy(df)
        if fill_na and False:
            df.loc[:, (df.dtypes == object)] = df.loc[:, (df.dtypes == object)].fillna(
                ""
            )
        if convert_categorical_to_integer and False:
            df = df.apply(convert_categorical_to_integer_f)
        access_scope = {"df": df, "pd": pd, "np": np}
        reject_quadratic_generated_code(code)
        parsed = ast.parse(code)
        check_ast(parsed)
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)
        df = copy.deepcopy(df)
        if target_name==None:
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                for c in non_numeric_cols:
                    col = df[c]
                    numeric = pd.to_numeric(col, errors="coerce")
                    if numeric.notna().any():
                        df[c] = numeric
                        continue
                    col_str = col.astype("string").fillna("__missing__").str.strip()
                    df[c] = pd.Categorical(col_str).codes.astype("float32")

            if df.select_dtypes(exclude=[np.number]).shape[1] != 0:
                raise ValueError(
                    "DataFrame contains columns with non-numeric types. Please ensure all columns are of numeric type.")
            if df.isna().any().any():
                df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            values = df.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                df = df.replace([np.inf, -np.inf], 0)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return df


def run_llm_code_new(code: str, df: pd.DataFrame, target_name: Optional[str] = None,
                 convert_categorical_to_integer: Optional[bool] = True,
                 fill_na: Optional[bool] = True) -> pd.DataFrame:

    try:
        loc = {}
        df = copy.deepcopy(df)

        if fill_na:
            object_cols = df.select_dtypes(include=['object']).columns
            df[object_cols] = df[object_cols].fillna("")

        if convert_categorical_to_integer:
            df = df.apply(convert_categorical_to_integer_f)

        access_scope = {"df": df, "pd": pd, "np": np}
        reject_quadratic_generated_code(code)
        parsed = ast.parse(code)

        check_ast(parsed)

        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)

        df = copy.deepcopy(df)

        if target_name is None:
            if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
                raise ValueError("DataFrame contains non-numeric columns. Please ensure all columns are numeric.")
            if df.isna().any().any():
                raise ValueError("DataFrame contains NaN values. Please handle missing values.")
            if np.isinf(df.values).any():
                raise ValueError("DataFrame contains infinite values. Please handle them.")
    except Exception as e:
        print("Code could not be executed:", e)
        raise e

    return df



import ast
import pandas as pd


def reject_quadratic_generated_code(code: str) -> None:
    text = code or ""
    lowered = text.lower()
    banned_patterns = [
        "pairwise_distances",
        "pairwise_kernels",
        "euclidean_distances",
        "cosine_similarity",
        "sklearn.metrics.pairwise",
        "scipy.spatial.distance.pdist",
        "scipy.spatial.distance.cdist",
        "squareform",
        "agglomerativeclustering",
        "spectralclustering",
        "meanshift",
        "dbscan",
        "affinitypropagation",
        "optics",
        "affinity='precomputed'",
        'affinity="precomputed"',
        "metric='precomputed'",
        'metric="precomputed"',
        "precomputed",
        "distance_matrix",
        "coassociation",
        "co-association",
    ]
    for pattern in banned_patterns:
        if pattern in lowered:
            raise ValueError(f"FORBIDDEN_O_N2_CODE: generated feature code contains `{pattern}`")


def check_ast(node: ast.AST) -> None:

    allowed_nodes = {
        ast.Module,
        ast.Expr,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Num,
        ast.Str,
        ast.Bytes,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Name,
        ast.Call,
        ast.Attribute,
        ast.keyword,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.ExtSlice,
        ast.Assign,
        ast.AugAssign,
        ast.NameConstant,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.Constant,
        ast.Store,
        ast.If,
        ast.IfExp,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Assert,
        ast.Return,
        ast.FunctionDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Lambda,
        ast.BoolOp,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Set,
        ast.Ellipsis,
        ast.expr,
        ast.stmt,
        ast.expr_context,
        ast.boolop,
        ast.operator,
        ast.unaryop,
        ast.cmpop,
        ast.comprehension,
        ast.arguments,
        ast.arg,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
    }

    allowed_packages = {"numpy", "pandas", "sklearn"}

    allowed_funcs = {
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "len": len,
        "isinstance": isinstance,
        "hash": hash,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "zip": zip,
        "range": range,
        "sorted": sorted,
        "reversed": reversed,
    }

    allowed_attrs = {
        # NP
        "array",
        "arange",
        "values",
        "linspace",
        # PD
        "mean",
        "sum",
        "contains",
        "where",
        "min",
        "max",
        "median",
        "std",
        "sqrt",
        "pow",
        "iloc",
        "cut",
        "qcut",
        "inf",
        "nan",
        "isna",
        "map",
        "reshape",
        "shape",
        "split",
        "var",
        "codes",
        "abs",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "diff",
        "repeat",
        "index",
        "log",
        "log10",
        "log1p",
        "slice",
        "exp",
        "expm1",
        "pow",
        "pct_change",
        "corr",
        "cov",
        "round",
        "clip",
        "dot",
        "transpose",
        "T",
        "astype",
        "copy",
        "drop",
        "dropna",
        "fillna",
        "replace",
        "merge",
        "append",
        "join",
        "groupby",
        "resample",
        "rolling",
        "expanding",
        "ewm",
        "agg",
        "aggregate",
        "filter",
        "transform",
        "apply",
        "pivot",
        "melt",
        "sort_values",
        "sort_index",
        "reset_index",
        "set_index",
        "reindex",
        "shift",
        "extract",
        "get",
        "rename",
        "tail",
        "head",
        "describe",
        "count",
        "value_counts",
        "unique",
        "nunique",
        "idxmin",
        "idxmax",
        "isin",
        "between",
        "duplicated",
        "rank",
        "to_numpy",
        "to_numeric",
        "to_dict",
        "to_list",
        "to_frame",
        "squeeze",
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "columns",
        "loc",
        "lt",
        "le",
        "eq",
        "ne",
        "ge",
        "gt",
        "all",
        "any",
        "clip",
        "conj",
        "conjugate",
        "round",
        "trace",
        "cumprod",
        "cumsum",
        "prod",
        "dot",
        "flatten",
        "ravel",
        "T",
        "transpose",
        "swapaxes",
        "clip",
        "item",
        "tolist",
        "argmax",
        "argmin",
        "argsort",
        "max",
        "mean",
        "min",
        "nonzero",
        "ptp",
        "sort",
        "std",
        "var",
        "str",
        "dt",
        "cat",
        "sparse",
        "keys",
        "plot"
        # Add other DataFrame methods you want to allow here.
    }

    if type(node) not in allowed_nodes:
        raise ValueError(f"Disallowed code: {ast.unparse(node)} is {type(node)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in allowed_funcs:
            raise ValueError(f"Disallowed function: {node.func.id}")

    if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
        raise ValueError(f"Disallowed attribute: {node.attr}")

    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name not in allowed_packages:
                raise ValueError(f"Disallowed package import: {alias.name}")

    for child in ast.iter_child_nodes(node):
        check_ast(child)

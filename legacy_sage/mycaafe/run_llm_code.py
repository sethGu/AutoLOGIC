import copy
import numpy as np
from .preprocessing import convert_categorical_to_integer_f
from typing import Any, Dict, Optional
import pandas as pd

# 安全地执行一段字符串形式的代码（通常由 LLM 生成），对传入的 DataFrame df 进行修改并返回修改后的结果
# 返回特征工程后的 DataFrame 新数据
def run_llm_code(code: str, df: pd.DataFrame, target_name: Optional[str] = None, convert_categorical_to_integer: Optional[bool] = True, fill_na: Optional[bool] = True) -> pd.DataFrame:
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:
        loc = {}
        # 对输入的 df 做深拷贝，避免原始数据被修改
        df = copy.deepcopy(df)
        ##################### 这段不会执行 #########################
        if fill_na and False:
            df.loc[:, (df.dtypes == object)] = df.loc[:, (df.dtypes == object)].fillna(
                ""
            )
        if convert_categorical_to_integer and False:
            df = df.apply(convert_categorical_to_integer_f)
        ##########################################################
        access_scope = {"df": df, "pd": pd, "np": np}
        blocked = _reject_quadratic_generated_code(code)
        if blocked is not None:
            raise ValueError(blocked)
        # 用 ast.parse 将字符串代码解析为 AST（抽象语法树），为安全起见避免直接用 exec(code)
        parsed = ast.parse(code)
        # 调用自定义的 check_ast 函数，对 AST 结构做安全检查（例如防止导入 os, sys, 删除文件等操作）
        check_ast(parsed)
        # 将 AST 编译为可执行代码，access_scope 是全局作用域，loc 是局部作用域（可存储函数内部定义的变量等
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)
        # 再深拷贝一次 df，可能是为了防止引用异常，确保返回的 df 是安全的副本
        df = copy.deepcopy(df)
        # 检查 df 是否包含非数值类型的列，如果有则抛出异常
        if target_name==None:
            if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
                raise ValueError(
                    "DataFrame contains columns with non-numeric types. Please ensure all columns are of numeric type.")
            # 检查 df 中是否有 NaN 或无穷值，如果有则抛出异常
            if df.isna().any().any():
                raise ValueError("DataFrame contains NaN values. Please ensure all missing values are handled.")
            if np.isinf(df.values).any():
                raise ValueError("DataFrame contains infinite values. Please ensure all infinite values are handled.")

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return df


# def convert_categorical_to_integer_f(col: pd.Series) -> pd.Series:
#     if col.dtype == "object" or col.dtype.name == "category":
#         return col.astype("category").cat.codes
#     return col

def run_llm_code_new(code: str, df: pd.DataFrame, target_name: Optional[str] = None,
                 convert_categorical_to_integer: Optional[bool] = True,
                 fill_na: Optional[bool] = True) -> pd.DataFrame:
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:
        loc = {}
        df = copy.deepcopy(df)

        # ✅ 处理缺失值：填充 object 类型列的 NaN 为 ""（空字符串）
        if fill_na:
            object_cols = df.select_dtypes(include=['object']).columns
            df[object_cols] = df[object_cols].fillna("")

        # ✅ 转换类别变量为整数编码
        if convert_categorical_to_integer:
            df = df.apply(convert_categorical_to_integer_f)

        access_scope = {"df": df, "pd": pd, "np": np}
        blocked = _reject_quadratic_generated_code(code)
        if blocked is not None:
            raise ValueError(blocked)
        parsed = ast.parse(code)

        # 需要你提供 check_ast 函数，否则这里会报错
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
import re


def _reject_quadratic_generated_code(code: str):
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
        "all_pairs",
        "precomputed",
        "distance_matrix",
        "connectivity_matrix",
        "coassociation",
        "co-association",
    ]
    for pattern in banned_patterns:
        if pattern in lowered:
            return f"FORBIDDEN_O_N2_CODE: generated feature code contains `{pattern}`"

    matrix_patterns = [
        r"np\.(zeros|ones|empty|full)\s*\(\s*\(\s*(?:n_samples|len\(\s*df\s*\)|df\.shape\s*\[\s*0\s*\])\s*,\s*(?:n_samples|len\(\s*df\s*\)|df\.shape\s*\[\s*0\s*\])",
        r"np\.(zeros|ones|empty|full)\s*\(\s*\[\s*(?:n_samples|len\(\s*df\s*\)|df\.shape\s*\[\s*0\s*\])\s*,\s*(?:n_samples|len\(\s*df\s*\)|df\.shape\s*\[\s*0\s*\])",
        r"@\s*(?:df|[A-Za-z0-9_]+)\.T",
        r"np\.matmul\s*\(.*,\s*.*\.T\s*\)",
        r"np\.dot\s*\(.*,\s*.*\.T\s*\)",
    ]
    for pattern in matrix_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return "FORBIDDEN_O_N2_CODE: generated feature code constructs an n_samples by n_samples object"
    return None


def check_ast(node: ast.AST) -> None:
    """
    Checks if the given AST node is allowed.

    Parameters:
    node (ast.AST): The AST node to check.

    Raises:
    ValueError: If the AST node is not allowed.
    """
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
        # These nodes represent loop structures. If you allow arbitrary loops, a user could potentially create an infinite loop that consumes system resources and slows down or crashes your system.
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
        # These nodes represent the yield keyword, which is used in generator functions. If you allow arbitrary generator functions, a user might be able to create a generator that produces an infinite sequence, potentially consuming system resources and slowing down or crashing your system.
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
        # Add other functions you want to allow here.
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

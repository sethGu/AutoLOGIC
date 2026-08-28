import json
import os
import time

from openai import OpenAI


def record_token_usage(stage, model, prompt_tokens, completion_tokens, total_tokens):
    ledger = os.environ.get("FIG5_TOKEN_LEDGER")
    if not ledger:
        return
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = int(total_tokens or prompt_tokens + completion_tokens)
    prompt_price = float(os.environ.get("FIG5_PROMPT_PRICE_PER_M", "1.5"))
    completion_price = float(os.environ.get("FIG5_COMPLETION_PRICE_PER_M", "1.95"))
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "stage": stage,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": prompt_tokens * prompt_price / 1_000_000 + completion_tokens * completion_price / 1_000_000,
    }
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    budget = os.environ.get("FIG5_MAX_LEDGER_COST_USD")
    if budget:
        total_cost = 0.0
        try:
            with open(ledger, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total_cost += float(json.loads(line).get("cost_usd") or 0)
        except OSError:
            total_cost = row["cost_usd"]
        if total_cost >= float(budget):
            raise SystemExit(f"FIG5 token budget exceeded: ${total_cost:.2f} >= ${float(budget):.2f}")


def compact_messages(messages):
    max_messages = int(os.environ.get("FIG5_MAX_MESSAGES", "6"))
    max_chars = int(os.environ.get("FIG5_MAX_MESSAGE_CHARS", "6000"))
    if len(messages) > max_messages:
        messages = messages[:1] + messages[-(max_messages - 1):]
    compacted = []
    for msg in messages:
        item = dict(msg)
        content = str(item.get("content", ""))
        if len(content) > max_chars:
            item["content"] = content[-max_chars:]
        compacted.append(item)
    return compacted


# Generate downstream model code
def generate_model(model, messages,base_url = None,api_key = None):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided")
    # Create an OpenAI API client instance
    client = OpenAI(
        # api_key=xxx,
        # base_url=xxx  # Modify if using proxy or local service
        base_url=base_url,
        api_key=api_key,
        timeout=float(os.environ.get("FIG5_OPENAI_TIMEOUT_S", "120")),
    )
    # This section calls the OpenAI Chat Completion API to generate responses based on the messages conversation context
    completion = client.chat.completions.create(
        model=model,
        messages=compact_messages(messages),
        stop=["```end"],
        temperature=0.7,
        max_completion_tokens=int(os.environ.get("FIG5_MAX_COMPLETION_TOKENS", "700"))
    )
    usage = completion.usage
    record_token_usage(
        "model_generation",
        model,
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
        getattr(usage, "total_tokens", 0),
    )
    # Extract the first generated message content from the model's response
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code


def generate_model_claude(model: str, messages: list[dict], base_url: str = None, api_key: str = None):
    """
    Call Anthropic Claude API to generate model code and return Token usage statistics (separate return value)
    """
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, base_url=base_url)

    # Claude API messages format is "messages=[{'role':'user','content':'...'}]"
    resp = client.messages.create(
        model=model,
        max_tokens=10000,
        # temperature=0.7,
        # stop_sequences=["```end"],
        messages=messages
    )

    # Parse token usage (Claude's field)
    prompt_tokens = resp.usage.input_tokens
    completion_tokens = resp.usage.output_tokens
    total_tokens = prompt_tokens + completion_tokens

    # Claude 输出结构: resp.content is list，需要取第一 text block
    code = resp.content[0].text
    code = code.replace("```python", "").replace("```", "").replace("<end>", "").strip()

    return {
        "code": code,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

def generate_model_gemini(model: str, messages: list[dict], base_url: str = None, api_key: str = None):
    """
    Call Google Gemini API to generate model code and return Token usage statistics（独立返回值）
    """
    if api_key is None:
        raise ValueError("api_key must be provided")

    import google.generativeai as genai

    # Google Gemini API not使use base_url，因this base_url 无效但保留接口格式
    genai.configure(api_key=api_key)

    # will OpenAI 格式 messages 转as Gemini  of 字符串 prompt
    # （Gemini: 输入as纯 text prompt）
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += f"User: {m['content']}\n"
        else:
            prompt += f"Assistant: {m['content']}\n"

    model = genai.GenerativeModel(model)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 700,
            "stop_sequences": ["```end"]
        }
    )

    # Token 消耗
    usage = response.usage_metadata
    prompt_tokens = usage.input_token_count
    completion_tokens = usage.output_token_count
    total_tokens = prompt_tokens + completion_tokens

    # 清洗生成代码
    code = response.text
    code = code.replace("```python", "").replace("```", "").replace("<end>", "").strip()

    return {
        "code": code,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


def generate_model_2(model: str, messages: list[dict], base_url: str = None, api_key: str = None) -> tuple[str, int, int, int]:
    """
    调用 OpenAI API 生成模型代码，并返回 Token 消耗统计（独立返回值）
    
    Args:
        model: 大模型名称（如 gpt-3.5-turbo、gpt-4o）
        messages: 对话提示词列表
        base_url: API 基础地址
        api_key: API 密钥
    
    Returns:
        code: 清洗后的生成代码
        prompt_tokens: 提示词 Token 消耗（API 官方统计）
        completion_tokens: 生成内容 Token 消耗（API 官方统计）
        total_tokens: 本次调用总 Token 消耗（API 官方统计）
    """
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided")
    
    # Initialize OpenAI 客户端
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=float(os.environ.get("FIG5_OPENAI_TIMEOUT_S", "120")))
    
    # Call API 生成代码
    completion = client.chat.completions.create(
        model=model,
        messages=compact_messages(messages),
        stop=["```end"],
        temperature=0.7,
        max_completion_tokens=int(os.environ.get("FIG5_MAX_COMPLETION_TOKENS", "700"))
    )
    
    # 提取 API 官方 Token 统计（核心：分开获取 prompt and completion 消耗）
    usage = completion.usage
    prompt_tokens = usage.prompt_tokens  # 提示词 Token 数
    completion_tokens = usage.completion_tokens  # Generate内容 Token 数
    total_tokens = usage.total_tokens  # 总 Token 数
    record_token_usage("model_generation", model, prompt_tokens, completion_tokens, total_tokens)
    
    # 清洗生成 of 代码（保持原have逻辑）
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "").strip()
    
    # Return：代码 + 三独立 of  Token 统计值
    # Return JSON 格式字典
    return {
        "code": code,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


def build_prompt_samples(df):
    samples = ""
    df_ = df.head(10)
    for i in list(df_):
        # show the list of values
        s = df_[i].tolist()
        # Ifthe列 of 数据类型is float64（浮点数），就把前10样本值四舍五入保留两位小数
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        # 构建一行字符串，表示当前列 of 列名、数据类型、缺失值频率，以及样本值列表，and把它追加to samples this总字符串in
        samples += (
            f"{df_[i].name} ({df[i].dtype}): Samples {s}\n"
        )
    return samples



def get_model_prompt_skold(
        target_column_name=None, samples=None
):
    prompt = f'''
    Here is what I provide:
    
    All column names and a few sample rows of data (in tabular form):
    {samples}
    
    The name of the target column for classification:
    "{target_column_name}"
    
    Based on this information, you must:
    1.Generate a new Python classifier class named myclassifier each round.
    2.The model should differ from previous versions (in model type or structure).
    3.The goal is to maximize the metrics on the given test data.
    
    The class must support the following methods:
    model = myclassifier()                      # Initialize the classifier
    model.fit(train_aug_x, train_aug_y)                # Train the model (both are pandas DataFrames)
    pred = model.predict(test_aug_x)                # Predict class labels (returns 1D array or list)
    proba = model.predict_proba(test_aug_x)            # Predict class probabilities (for computing AUC)
    
    Important instructions (must follow strictly):
    - Only output the complete Python class named myclassifier (no explanation, no comments, no extra code).
    - The class must be ready to use in Python (no missing imports or undefined variables).
    - train_aug_x, train_aug_y, and test_aug_x are all pandas DataFrames.
    - The predict method must return a 1D array or list of labels (same length as test_x).
    - The predict_proba method must return the probability of the positive class using the format: return self.model.predict_proba(test_x)[:, 1]
    - Each new version must differ from the previous one (by model type).
    - Always use a different model type or structure from all previous versions. Model diversity is critical (e.g., LogisticRegression, RandomForest, Gradient Boosting, SVC with probability, neural nets, pipelines, stacked models, etc.).
    - You may use scikit-learn, XGBoost, LightGBM, CatBoost, or any compatible library.
    - You may use libraries such as scikit-learn (>=1.4), XGBoost, CatBoost, etc.
    - The classifier class must be fully self-contained and ready to run in Python.
    - Strictly output only the complete Python class named myclassifier. No extra explanation or code.
    '''
    return prompt


def get_model_prompt(target_column_name=None, samples=None):
    prompt = f"""
        Here is what I provide:

        All column names and a few sample rows of data (in tabular form):
        {samples}

        The name of the target column for classification:
        "{target_column_name}"

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to **maximize AUC (Area Under the ROC Curve)** on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities of the positive class

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list (same length as `test_aug_x`).
        - `predict_proba` must return the **positive class** probability:
              return self.model.predict_proba(test_x)[:, 1]
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, LightGBM, CatBoost, etc.
        - **You must NOT use `BaggingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
        - Always prefer models that support probabilistic outputs suitable for AUC optimization.

        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt


def get_model_prompt_multi(target_column_name=None, samples=None):
    prompt = f"""
        Here is what I provide:

        All column names and a few sample rows of data (in tabular form):
        {samples}

        The name of the target column for multi-class classification:
        "{target_column_name}"

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to maximize classification performance on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities for each class

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list of predicted labels (same length as `test_aug_x`).
        - `predict_proba` must return a 2D NumPy array of shape (n_samples, n_classes), representing probabilities for **each class**:
              return self.model.predict_proba(test_aug_x)
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, LightGBM, CatBoost, etc.
        - The model must be designed for multi-class classification tasks (i.e., handle `n_classes > 2`).
        - **You must NOT use `BaggingClassifier` under any circumstances. It is forbidden due to compatibility issues.**

        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt


"""
回归任务模型生成的提示词
"""
def get_regression_model_prompt(
        target_column_name=None, samples=None, n_clusters=3
):
    prompt = f'''
    Here is what I provide:
    
    All column names and a few sample rows of data (in tabular form):
    {samples}
    
    The name of the target column for regression:
    "{target_column_name}"
    
    After each iteration, I will return the current RMSE value, the lowest RMSE value in all previous iterations, and the regressor model code when the lowest RMSE value was achieved in all previous iterations.
    
    Based on this information, you must:
    1. Generate a new Python regressor class named `myregressor` each round.
    2. The model should differ from previous versions (in model type or structure).
    3. The goal is to minimize RMSE on the given test data.
    
    The class must support the following methods:
    ```python
    model = myregressor()                          # Initialize the regressor
    model.fit(train_aug_x, train_aug_y)            # Train the model (both are pandas DataFrames)
    pred = model.predict(test_aug_x)               # Predict regression values (returns 1D array or list)
    ```
    
    Important instructions (must follow strictly):
    - Only output the complete Python class named myregressor (no explanation, no comments, no extra code).
    - The class must be ready to use in Python (no missing imports or undefined variables).
    - You should lean more towards exploring powerful regressors, such as LightGBM, CatBoost, XGBoost, or ensemble/regression classification_stacking methods.
    - train_aug_x, train_aug_y and test_aug_x are all pandas DataFrames.
    - The predict method must return a 1D array or list of predicted values (using the format: return self.model.predict(test_aug_x), do not convert it to a python list using .tolist()).
    - Each new version must differ from the previous one (by model type or structure).
    - After I provide the current RMSE and the best (lowest) RMSE in previous iterations, improve the model in the next round accordingly.
    '''

    return prompt



"""
聚类任务的模型生成提示词
"""
def get_clustering_model_prompt(target_column_name=None, samples=None, n_clusters=3):
    prompt = f'''
        You are to act as an expert in machine learning, especially unsupervised learning and clustering.
        
        Dataset:
        - Samples (tabular form):
        {samples}
        - Number of clusters to discover: {n_clusters}
        
        Task:
        - Generate a Python class named `mycluster` for clustering.
        - Each new version must use a different type of clustering algorithm or pipeline.
        - The objective is to maximize Adjusted Rand Index (ARI).
        
        Requirements for the class:
        - Class name: `mycluster`
        - Constructor must accept `n_clusters` as a parameter (even if not explicitly used).
        - Must implement `fit_predict(X)` method where X is a pandas DataFrame, returning predicted labels.
        - Class must include all required imports.
        - Be ready-to-execute Python code only.
        - Avoid repeating previously used algorithms (e.g. KMeans, Spectral, DBSCAN...); innovate with hybrid, ensemble, or dimensionality-reduction-based models.
        - The generated model should be structurally different from all previous ones.
        - If you use sklearn.pipeline.Pipeline: every intermediate step MUST be a transformer with fit+transform; do NOT place estimators (e.g., IsolationForest/KMeans) as intermediate steps.
        - Do NOT use IsolationForest inside Pipeline steps.
        - Avoid text, explanations, or comments. Output only the class definition code.
        - **You must NOT use `VotingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
        '''
    return prompt



"""
LLM 自动生成簇数
"""
def get_clustering_model_prompt_v2(samples=None, dataset_summary=None):
    prompt = f'''
You are to act as an expert in machine learning, especially unsupervised learning and clustering.

You will be provided with a dataset preview and statistical summary. Based on this information, estimate a reasonable number of clusters and generate Python code for clustering accordingly.

Dataset Preview (first 10 rows):
{samples}

Dataset Summary:
- Number of samples: {dataset_summary.get("n_samples")}
- Number of features: {dataset_summary.get("n_features")}
- Feature types: {dataset_summary.get("feature_types")}
- Feature ranges: {dataset_summary.get("feature_ranges")}
- Mean and standard deviation (numerical features only): {dataset_summary.get("mean_std")}

Task:
- Analyze the dataset summary to determine the optimal number of clusters automatically (e.g., via internal metrics or distribution analysis).
- Generate a Python class named `mycluster` for clustering.
- Each new version must use a different type of clustering algorithm or pipeline.
- The objective is to maximize Adjusted Rand Index (ARI).

Requirements for the class:
- Class name: `mycluster`
- Do not accept `n_clusters` as a constructor parameter.
- The optimal number of clusters should be hard-coded within the class based on your estimation.
- Must implement `fit_predict(X)` method where X is a pandas DataFrame, returning predicted labels.
- Class must include all required imports.
- Be ready-to-execute Python code only.
- Avoid repeating previously used algorithms (e.g., KMeans, Spectral, DBSCAN...); innovate with hybrid, ensemble, or dimensionality-reduction-based models.
- The generated model should be structurally different from all previous ones.
- If you use sklearn.pipeline.Pipeline: every intermediate step MUST be a transformer with fit+transform; do NOT place estimators (e.g., IsolationForest/KMeans) as intermediate steps.
- Do NOT use IsolationForest inside Pipeline steps.
- Avoid text, explanations, or comments. Output only the class definition code.
- **You must NOT use `VotingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
'''
    return prompt

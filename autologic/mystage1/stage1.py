from __future__ import annotations

import copy
import numpy as np
import torch
from openai import OpenAI
from sklearn.model_selection import RepeatedKFold
from .stage1_evaluate import (
    evaluate_dataset,
    filter_generated_features
)
from .run_llm_code import run_llm_code
import re
import json
import os
import time


def record_feature_token_usage(model, usage):
    ledger = os.environ.get("FIG5_TOKEN_LEDGER")
    if not ledger or usage is None:
        return
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens)
    prompt_price = float(os.environ.get("FIG5_PROMPT_PRICE_PER_M", "1.5"))
    completion_price = float(os.environ.get("FIG5_COMPLETION_PRICE_PER_M", "1.95"))
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "stage": "feature_generation",
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
        with open(ledger, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_cost += float(json.loads(line).get("cost_usd") or 0)
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


def get_prompt(
        df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"
Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

Important: the target column "{ds[4][-1]}" is NOT included in `df` during feature generation. Do not reference it.

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.

Formatting rules:

You must only output codeblocks following this format:

For generating new features:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using '{df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Strict rules:
- Only output one codeblock at a time.
- Do not output natural language outside codeblocks.
- Do not output markdown, comments, explanations, or blank lines outside the codeblock.
- Each codeblock must be self-contained and follow the format exactly.
- Each block must include the feature name, usefulness, input samples, and code.
- The codeblock for a new feature must be different from those of previously generated features.
- Only generate numerical features (features must be of numerical type).

Violation of these rules will result in rejection of your output.
"""


def build_prompt_from_df(ds, df, iterative=1):
    data_description_unparsed = ds[-1]
    feature_importance = {}

    samples = ""
    target_col = ds[4][-1]
    df_features = df.drop(columns=[target_col], errors="ignore")
    df_ = df_features.head(10)
    for i in list(df_):
        nan_freq = "%s" % float("%.2g" % (df_features[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df_features[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += (
            f"{df_[i].name} ({df_features[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df_features,
        ds,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


def load_model(model_loc):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, tokenizer = (
        AutoModelForCausalLM.from_pretrained(model_loc, device_map='auto', torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, trust_remote_code=True),
        AutoTokenizer.from_pretrained(model_loc),
    )
    return model, tokenizer


def generate_completion_transformers(
        input: list,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_token=512,
        device=torch.device('cuda')
):
    from transformers import GenerationConfig

    model.to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False)

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    with torch.no_grad():
        generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_texts


def generate_features(
        ds,
        df,
        model="gpt-3.5-turbo",
        iterative=1,
        metric_used=None,
        iterative_method="logistic",
        n_splits=10,
        n_repeats=2,
        task_type="classification",
        base_url:str=None,
        api_key:str=None,
        logger=None
):
    assert (
            iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    prompt = build_prompt_from_df(ds, df, iterative=iterative)
    if model == 'llama2':
        local_model_path = os.environ.get("AUTOLOGIC_LOCAL_LLM_PATH")
        if not local_model_path:
            raise RuntimeError("AUTOLOGIC_LOCAL_LLM_PATH must point to the local model directory.")
        basemodel, tokenizer = load_model(local_model_path)

    def generate_code(messages):
        if model == "skip":
            return ""
        elif model == 'llama2':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            code = generate_completion_transformers(
                input=messages,
                model=basemodel,
                tokenizer=tokenizer,
                device=device
            )
        elif model == 'deepseek':
            client = OpenAI(
                base_url='http://localhost:11434/v1/',
                api_key='ollama'
            )
            chat_completion = client.chat.completions.create(
                model='deepseek-r1:8b',
                messages=messages,
                stop=["```end"],
                temperature=0.5,
                max_completion_tokens=200
            )
            code_block = chat_completion.choices[0].message.content
            code_block = re.sub(r'<think>.*?</think>', '', code_block, flags=re.DOTALL)
            match = re.search(r"```python\s*(.*?)```", code_block, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                code = code_block

        else:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=float(os.environ.get("FIG5_OPENAI_TIMEOUT_S", "120"))
            )
            completion = client.chat.completions.create(
                model=model,
                messages=compact_messages(messages),
                stop=["```end"],
                temperature=0.5,
                max_completion_tokens=int(os.environ.get("FIG5_FEATURE_MAX_COMPLETION_TOKENS", "500"))
            )
            record_feature_token_usage(model, completion.usage)
            code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_code_block(code):
        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for (train_idx, valid_idx) in ss.split(df):
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                run_llm_code(
                    code,
                    df_train_extended,
                )
                run_llm_code(
                    code,
                    df_valid_extended,
                )

            except Exception as e:
                print(f"Error in code execution. {type(e)} {e}")
                print(f"```python\n{code}\n```\n")
                return e

        return None

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    n_iter = iterative
    full_code = ""
    code_blocks = []
    i = 0
    retry_count = 0
    max_retries = int(os.environ.get("AUTOLOGIC_MAX_FEATURE_RETRIES", "2"))
    while i < n_iter:
        if logger is not None:
            try:
                logger.start_round(
                    stage="feature_iter",
                    r=i + 1,
                    task_type=task_type,
                    llm=model,
                    n_rows=int(len(df)),
                    n_cols=int(df.shape[1]),
                )
            except Exception:
                pass
        try:
            code = generate_code(messages)
        except Exception as e:
            print("Error in LLM API." + str(e))
            retry_count += 1
            if logger is not None:
                try:
                    logger.event("llm_api_error", 1)
                    logger.end_round(status="llm_api_error", error_type=type(e).__name__, error=str(e))
                except Exception:
                    pass
            if retry_count >= max_retries:
                print(f"Maximum feature-generation retries exceeded, skipping feature {i + 1}")
                retry_count = 0
                i += 1
            continue

        e = execute_code_block(code)

        if e is not None:
            retry_count += 1
            extra_hint = ""
            if isinstance(e, KeyError) and str(e).strip("'\"") == ds[4][-1]:
                extra_hint = f"\nImportant: do not access df['{ds[4][-1]}'] because the target column is not included in df during feature generation."
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""
                    Code execution failed with error: {type(e)} {e}.{extra_hint}\n Code: ```python{code}```\n Generate next feature (fixing error?):
                    ```python
                    """,
                },
            ]
            if logger is not None:
                try:
                    logger.event("code_exec_error", 1)
                    logger.end_round(status="code_exec_error", error_type=type(e).__name__, error=str(e), code_len=len(code))
                except Exception:
                    pass
            if retry_count >= max_retries:
                print(f"Maximum feature-execution retries exceeded, skipping feature {i + 1}")
                retry_count = 0
                i += 1
            continue

        i = i + 1
        retry_count = 0

        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""
                    The feature code execution successed.
                    Next codeblock:
                    """,
                },
            ]

        full_code += code
        code_blocks.append(code)
        if logger is not None:
            try:
                logger.end_round(status="success", code_len=len(code), full_code_len=len(full_code))
            except Exception:
                pass

    return full_code, prompt, messages, code_blocks

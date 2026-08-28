from __future__ import annotations

import copy
import numpy as np
import torch
from openai import OpenAI
from sklearn.model_selection import RepeatedKFold
from .caafe_evaluate import (
    evaluate_dataset,
    filter_generated_features
)
from .run_llm_code import run_llm_code
import re


# 获取提示词
def get_prompt(
        df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"
Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

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
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(10)
    # df_ = df.drop(columns=ds[4][-1]).head(10)
    for i in list(df_):
        # show the list of values
        # 计算当前列 i 中缺失值的比例（用 df[i].isna().mean() 计算缺失率，再乘以 100 转换成百分比）
        # 用 "%.2g" 保留两位有效数字，再格式化为字符串 nan_freq，表示缺失频率
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        # 如果该列的数据类型是 float64（浮点数），就把前10个样本值四舍五入保留两位小数
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        # 构建一行字符串，表示当前列的列名、数据类型、缺失值频率，以及样本值列表，并把它追加到 samples 这个总字符串中
        samples += (
            f"{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df,
        ds,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


# 加载本地大模型
def load_model(model_loc):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, tokenizer = (
        AutoModelForCausalLM.from_pretrained(model_loc, device_map='auto', torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, trust_remote_code=True),
        AutoTokenizer.from_pretrained(model_loc),
    )
    return model, tokenizer


# 生成回答
"""
函数的主要作用是使用大语言模型（LLM），根据输入的对话消息列表生成文本回复
只有用 llama2 模型的时候，才使用到了该函数
先不用管
"""


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

    # logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
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
        api_key:str=None
):
    # 如果这两个条件都不满足，就会抛出错误
    assert (
            iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    # 获取生成特征代码提示词
    prompt = build_prompt_from_df(ds, df, iterative=iterative)
    # 是否使用本地模型
    if model == 'llama2':
        local_model_path = os.environ.get("AUTOLOGIC_LOCAL_LLM_PATH")
        if not local_model_path:
            raise RuntimeError("AUTOLOGIC_LOCAL_LLM_PATH must point to the local model directory.")
        basemodel, tokenizer = load_model(local_model_path)

    # 生成代码
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
                api_key='ollama'  # 必填，但可以是任意字符串
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
            # 匹配 ```python 开头 和 ```end 结尾之间的内容（非贪婪匹配）
            match = re.search(r"```python\s*(.*?)```", code_block, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                code = code_block


        ####    实验中只用到 openAI 的模型===============================================================================
        else:
            # 创建一个 OpenAI 的 API 客户端实例
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            # 这一段是调用 OpenAI 的 Chat Completion 接口，让模型根据 messages 对话上下文生成回复
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["```end"],
                temperature=0.5,
                max_completion_tokens=500
            )
            # 从模型的返回结果中提取第一条生成的消息内容
            code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    # 在交叉验证中执行某段 LLM 生成的代码
    def execute_code_block(code):
        # 创建一个交叉验证器，使用重复 K 折交叉验证。控制随机种子为 0 以保证可重复
        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for (train_idx, valid_idx) in ss.split(df):
            # 将原始数据按索引划分为训练集和验证集
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            # 从特征中移除这目标列
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            # 对训练和验证集做深拷贝，用于之后加上新代码后的版本（extended）
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
                # 如果执行中报错（如代码不合法），就输出错误信息 + 展示这段代码，然后返回空值，跳过评估
                print(f"Error in code execution. {type(e)} {e}")
                print(f"```python\n{code}\n```\n")
                return e

        return None

    # 输入给大模型的完整提示词 generate_code 用到
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


    # # 将 messages 写入文件，便于调试和查看
    # file_path = 'prompt_debug.jsonl'
    # import json

    # with open(file_path, 'a') as f:
    #     # 将消息转为 JSON 字符串，确保可读性且适合存储
    #     f.write(json.dumps(messages, ensure_ascii=False) + '\n')



    n_iter = iterative
    full_code = ""
    i = 0
    attempts = 0
    api_errors = 0
    max_attempts = max(5, n_iter * 5)
    while i < n_iter and attempts < max_attempts:
        attempts += 1
        try:
            code = generate_code(messages)
            api_errors = 0
        except Exception as e:
            print("Error in LLM API." + str(e))
            api_errors += 1
            if api_errors >= 3:
                raise RuntimeError(f"LLM API failed {api_errors} consecutive times during feature generation.") from e
            continue

        e = execute_code_block(code)

        # 如果生成的代码执行出错，把错误信息加到 messages 中，反馈给 LLM
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""
                    Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                    ```python
                    """,
                },
            ]
            continue

        i = i + 1

        # print(
        #     "\n"
        #     + f"*Iteration {i}*\n"
        #     + f"```python\n{code}\n```\n"
        # )

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

    if i < n_iter:
        print(f"Feature generation stopped after {attempts} attempts; completed {i}/{n_iter} iterations.")
    if i == 0:
        raise RuntimeError("Feature generation produced zero valid iterations.")

    return full_code, prompt, messages

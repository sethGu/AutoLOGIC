import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)


from mycaafe import CAAFEClassifier  # Automated Feature Engineering for tabular datasets
from mycaafe.run_llm_code import run_llm_code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils.model_generate import build_prompt_samples, generate_model,get_regression_model_prompt
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
import pickle
import numpy as np
import random
import argparse
import warnings
import copy

from utils.ensembleUtils import get_regression_metaModel
from ensemble.regression_ensemble_utils import stacking_regression_util,voting_regression_util,bagging_regression_util
from utils.utils import format_mean_std,format_mean_std_four

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


# 加载本地数据集
def load_origin_data(loc, seed):
    # 读取数据集
    with open(loc, 'rb') as f:
        ds = pickle.load(f)

    # 目标列名
    target_column_name = ds[4][-1]

    df = ds[1]

    # 数据集描述，包含列描述
    dataset_description = ds[-1]
    # print(dataset_description)

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=seed)

    return df_train, df_test, target_column_name, dataset_description

def base_model(seed):
    rforest = RandomForestRegressor(n_estimators=100, random_state=seed)

    return rforest

def to_pd(df, target_name):
    y = df[target_name]
    x = df.drop(target_name, axis=1)

    return x, y

def code_exec(code):
    try:
        # 尝试编译检查（compile 成 AST 再执行）
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals())
        return None
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e)

# 生成特征
def generate_feat(
        base_model,
        df_train,
        df_test,
        dataset_name,
        round_num,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',
        dataset_description=None,
        task_type="regression",
        base_url:str=None,
        api_key:str=None
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    caafe_clf = CAAFEClassifier(base_classifier=base_model,
                                llm_model=llm_model,
                                iterations=iterations)

    caafe_clf.fit_pandas(df_train,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description,
                         dataset_name=dataset_name,
                         round_num=round_num,
                         task_type=task_type,
                         base_url=base_url,
                         api_key=api_key
                         )

    df_train_aug = run_llm_code(caafe_clf.code, df_train, target_column_name)
    df_test_aug = run_llm_code(caafe_clf.code, df_test, target_column_name)

    df_train_aug = df_train_aug[caafe_clf.final_columns]
    df_test_aug = df_test_aug[caafe_clf.final_columns]

    return df_train_aug, df_test_aug

def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

def print_rmsle(name, values):
    print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='大模型')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-e', '--exam_iterations', default=2, type=int, help='实验次数')
    parser.add_argument('-f', '--feat_iterations', default=5, type=int, help='特征迭代次数')
    parser.add_argument('-m', '--model_iterations', default=5, type=int, help='模型迭代次数')
    parser.add_argument('-d', '--dataset',default="concrete",help="数据集")   # 'boston','concrete','california','insurance','winequality'
    parser.add_argument('--enable_feedback', action='store_true',default=True,
                    help='是否开启模型生成反馈机制')
    args = parser.parse_args()

    # 模型标签，全局区分 LLM 生成的模型
    model_tab = 1

    """
    openAI API 设置 
    """
    # TODO     openAI API 设置 
    # --- 获取环境变量 ---
    env_url = os.getenv("OPENAI_BASE_URL")
    env_key = os.getenv("OPENAI_API_KEY")
    # --- 手动设置 ---
    manual_url = ""
    manual_key = ""

    # --- 选择优先级：环境变量 > 手动设置 ---
    if env_url and env_key:
        base_url = env_url
        api_key = env_key
    elif manual_url and manual_key:
        base_url = manual_url
        api_key = manual_key
    else:
        raise ValueError(
            "No valid OpenAI API configuration found. "
            "Please set environment variables or manual config."
        )

    ds_name = args.dataset 
    print(f"=========== Dataset {ds_name} ===========")
    # 新增：存储每次实验结果的列表
    mae_list, rmse_list, rmsle_list = [], [], []

    loc = f"{project_root}/data/" + ds_name + ".pkl"

    #存储集成学习的评估指标的结果
    test_MSE_list_ensemble = []
    test_RMSE_list_ensemble = []
    test_RMSLE_list_ensemble = []

    test_MSE_list_ensemble_voting = []
    test_RMSE_list_ensemble_voting  = []
    test_RMSLE_list_ensemble_voting  = []

    test_MSE_list_ensemble_bagging = []
    test_RMSE_list_ensemble_bagging  = []
    test_RMSLE_list_ensemble_bagging  = []

    exam_iter = args.exam_iterations

    # 实验次数
    for exp in range(exam_iter):
        print(f"=========== Experiment {exp + 1}/{exam_iter} ===========")

        # 存储每次实验结果的列表
        test_mse_list = []
        test_rmse_list = []
        test_rmsle_list = []

        seed = args.default_seed + exp

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        df_train, df_test, target_column_name, dataset_description = load_origin_data(loc, seed)
        baseline_model = base_model(seed)
        feat_iter = args.feat_iterations
        df_train_aug, df_test_aug = generate_feat(
            base_model=baseline_model,
            df_train=df_train,
            df_test=df_test,
            dataset_name=ds_name,
            round_num=exp + 1,
            llm_model=args.llm,
            iterations=feat_iter,
            target_column_name=target_column_name,
            dataset_description=dataset_description,
            task_type="regression",
            base_url=base_url,
            api_key=api_key
        )
        df_train_aug, df_valid_aug = train_test_split(df_train_aug, test_size=0.25, random_state=seed)
        train_aug_x, train_aug_y = to_pd(df_train_aug, target_column_name)
        test_aug_x, test_aug_y = to_pd(df_test_aug, target_column_name)
        val_aug_x, val_aug_y = to_pd(df_valid_aug, target_column_name)

        s = build_prompt_samples(df_train_aug)

        model_prompt = get_regression_model_prompt(
            target_column_name=target_column_name,
            samples=s
        )

        model_messages = [
            {
                "role": "system",
                "content": "You are a top-level regression algorithm expert. Your task is to help me iteratively search for the most suitable regression model that performs best on a regression task based on the RMSE (Root Mean Squared Error) metric. Your answer should only generate code.",
            },
            {
                "role": "user",
                "content": model_prompt,
            },
        ]

        model_iter = args.model_iterations
        best_mse = float("inf")
        best_rmse = float("inf")
        best_rmsle = float("inf")
        best_code = None
        i = 0

        # 参与集成学习基础模型列表
        base_models = []
        base_models_voting_bagging = []
        # 模型生成迭代
        while i < model_iter:
            try:
                code = generate_model(args.llm, model_messages,base_url,api_key)
                # with open(f"code/ours/regression/{ds_name}_{args.llm}_regressor{exp + 1}.py", "r",
                #           encoding="utf-8") as f:
                #     code = f.read()
                code = code.replace("```python", "").replace("```", "").replace("<end>", "")
                # print(code)
            except Exception as e:
                print("Error in LLM API." + str(e))
                continue

            e = code_exec(code)

            if e is not None:
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            The last generated regressor model code failed with error: {type(e)} {e}.\n Here is the code that failed: ```python{code}```
                            New strict instructions:
                            - Only output a corrected Python code block, immediately starting with triple backticks (```python).
                            - Do not apologize, explain, or add any text outside the code block.
                            - Any non-code text will invalidate the output.
                            
                            Generate next corrected code block:
                            """,
                    },
                ]
                continue

            try:
                model = myregressor()
                # 用于集成的模型
                model_copy = copy.deepcopy(model)
                model_tab = model_tab + 1
                model.fit(train_aug_x, train_aug_y)
                base_models_voting_bagging.append(model)
                pred = model.predict(val_aug_x)
            except Exception as e:
                print("Model code execution failed with error:" + str(e))
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            Code execution failed with error: {type(e)} {e}.\n Here is the code that failed: ```python{code}```\n Generate next code block(fixing error?):
                            """,
                    },
                ]
                continue
            base_models.append(model_copy)

            i = i + 1
            print(f"........... Model_Iterations {i}/{model_iter} ...........")

            # 计算评估指标
            test_mse = mean_absolute_error(val_aug_y, pred)
            # test_rmse = mean_squared_error(val_aug_y, pred, squared=False)
            test_rmse = np.sqrt(np.mean((val_aug_y.values - pred) ** 2))
            test_mse_list.append(test_mse)
            test_rmse_list.append(test_rmse)
            if (pred < 0).any() or (val_aug_y < 0).any():
                test_rmsle = np.nan
                print("RMSLE cannot be computed due to negative values.")
            else:
                test_rmsle = np.sqrt(mean_squared_log_error(val_aug_y, pred))
                test_rmsle_list.append(test_rmsle)

            if test_rmse < best_rmse:
                best_mse = test_mse
                best_rmse = test_rmse
                best_rmsle = test_rmsle
                best_code = code
                # with open(f"{project_root}/code/regression/{ds_name}_{args.llm}_regressor{exp + 1}.py", "w",
                #           encoding="utf-8") as f:
                #     f.write(code)

            if args.enable_feedback and len(code) > 10:
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            The regressor model code executed successfully. The RMSE value after the current code is executed: {test_rmse}. The lowest RMSE value in all previous iterations: {best_rmse}.
                            The regressor model code with the highest ARI value in all previous iterations:\n```python{best_code}```
                            Please continue to find a regressor model that is more likely to reduce the RMSE value.
                            Moreover, the new regressor model must differ from all the previous ones (by model type or structure).
                            Remember, your answer should only generate code.
                            Next code block:
                            """,
                    },
                ]
            else:
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            Please continue to find a regressor model that is more likely to reduce the RMSE value.
                            Moreover, the new regressor model must differ from all the previous ones (by model type or structure).
                            Remember, your answer should only generate code.
                            Next code block:
                            """,
                    },
                ]

            # print(f"best_code:{best_code}")
            # print(f"Model_Iterations ended. best_rmse:{best_rmse}")

            # 存储结果
            mae_list.append(test_mse)
            rmse_list.append(test_rmse)
            rmsle_list.append(test_rmsle)
            # 打印当前实验详细结果
            print(f"当前实验结果第 {i}/{args.model_iterations}")
            print(f"\nTest  MAE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, RMSLE: {test_rmsle:.4f}")

        print(f"\n=========== 第 {exp+1} 次实验实验结果指标列表 ===========")
        print("MAE", mae_list)
        print("RMSE", rmse_list)
        print("RMSLE", rmsle_list)

        print(f"\n=========== 第 {exp+1} 次实验实验结果指标列表 ===========")
        print_stats("MAE", mae_list)
        print_stats("RMSE", rmse_list)
        print_rmsle("RMSLE", rmsle_list)
        print("\n")

        # model_iter 结束后，筛选 top 5 模型：筛选条件是 MAE
        top_k = 5
        if len(base_models) > top_k:
            # 根据 test_mse_list 排序，取前 top_k 的索引
            top_k_indices = np.argsort(test_mse_list)[:top_k]
            selected_base_models = [base_models[i] for i in top_k_indices]
        else:
            selected_base_models = base_models



        """
        集成学习  调用 classification_stacking 集成学习方法
        """
        # 想要测试的元模型名字列表
        metaModelName_list = [
            # 'RandomForestRegressor',
            # 'XGBRegressor',
            # 'LGBMRegressor',-----
            # 'CatBoostRegressor',------
            # 'SVR',
            # 'DecisionTreeRegressor',-----
            'LinearRegression',
            # 'Ridge',-------
            # 'Lasso',-------
            # 'ElasticNet',
            # 'MLPRegressor',
            # 'KNeighborsRegressor',
            # 'BaggingRegressor',
        ]

        # 想要测试的元模型列表
        metaModelName_list = get_regression_metaModel(metaModelName_list)

        for meta_model in metaModelName_list:
            # 调用集成学习方法
            result = stacking_regression_util(selected_base_models, meta_model, train_aug_x, train_aug_y, test_aug_x, test_aug_y)
            stacking_metrics = result['stacking_metrics']

            print(f"\n=========== 第{exp + 1}次Stacking集成结果--{type(meta_model).__name__} ===========")
            print(f"MAE : {stacking_metrics['mae']:.4f}")
            print(f"RMSE : {stacking_metrics['rmse']:.4f}")
            print(f"RMLSE: {stacking_metrics['rmsle']:.4f}")

            # 保存集成结果
            test_MSE_list_ensemble.append(round(stacking_metrics['mae'], 4))
            test_RMSE_list_ensemble.append(round(stacking_metrics['rmse'], 4))
            test_RMSLE_list_ensemble.append(round(stacking_metrics['rmsle'], 4))



        # 接下来是 voting 集成方法
        # 调用集成学习方法
        result = voting_regression_util(base_models_voting_bagging, test_aug_x, test_aug_y)
        stacking_metrics = result['voting_metrics']

        print(f"\n=========== 第{exp + 1}次 voting 集成结果--{type(meta_model).__name__} ===========")
        print(f"MAE : {stacking_metrics['mae']:.4f}")
        print(f"RMSE : {stacking_metrics['rmse']:.4f}")
        print(f"RMLSE: {stacking_metrics['rmsle']:.4f}")

        # 保存集成结果
        test_MSE_list_ensemble_voting.append(round(stacking_metrics['mae'], 4))
        test_RMSE_list_ensemble_voting.append(round(stacking_metrics['rmse'], 4))
        test_RMSLE_list_ensemble_voting.append(round(stacking_metrics['rmsle'], 4))


        # 接下来是 bagging 集成方法
        # 调用集成学习方法
        result = bagging_regression_util(base_models_voting_bagging, test_aug_x, test_aug_y)
        stacking_metrics = result['bagging_metrics']
        print(f"\n=========== 第{exp + 1}次 bagging 集成结果--{type(meta_model).__name__} ===========")
        print(f"MAE : {stacking_metrics['mae']:.4f}")
        print(f"RMSE : {stacking_metrics['rmse']:.4f}")
        print(f"RMLSE: {stacking_metrics['rmsle']:.4f}")

        # 保存集成结果
        test_MSE_list_ensemble_bagging.append(round(stacking_metrics['mae'], 4))
        test_RMSE_list_ensemble_bagging.append(round(stacking_metrics['rmse'], 4))
        test_RMSLE_list_ensemble_bagging.append(round(stacking_metrics['rmsle'], 4))


    # 构建输出内容
    summary_lines = [
        f"\n=========== 实验结束：数据集 {ds_name}，共 {args.exam_iterations} 次 regression_stacking 集成学习统计结果 ===========",
        f"MAE: {format_mean_std(test_MSE_list_ensemble)}",
        f"RMSE: {format_mean_std(test_RMSE_list_ensemble)}",
        f"RMSLE: {format_mean_std_four(test_RMSLE_list_ensemble)}",
        "\n"
        f"\n=========== 实验结束：数据集 {ds_name}，共 {args.exam_iterations} 次 regression_voting 集成学习统计结果 ===========",
        f"MAE: {format_mean_std(test_MSE_list_ensemble_voting)}",
        f"RMSE: {format_mean_std(test_RMSE_list_ensemble_voting)}",
        f"RMSLE: {format_mean_std_four(test_RMSLE_list_ensemble_voting)}",
        "\n"
        f"\n=========== 实验结束：数据集 {ds_name}，共 {args.exam_iterations} 次 regression_bagging 集成学习统计结果 ===========",
        f"MAE: {format_mean_std(test_MSE_list_ensemble_bagging)}",
        f"RMSE: {format_mean_std(test_RMSE_list_ensemble_bagging)}",
        f"RMSLE: {format_mean_std_four(test_RMSLE_list_ensemble_bagging)}",
        "\n"
    ]

    # 打印到控制台
    for line in summary_lines:
        print(line)
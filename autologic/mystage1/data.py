import pandas as pd
import torch
import numpy as np
import openml
import re
import os
from sklearn.model_selection import train_test_split
import copy



def get_openml_classification(did, max_samples, multiclass=True, shuffled=True):
    
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    description = refactor_openml_description(dataset.description)

    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if multiclass and not shuffled:
        raise NotImplementedError(
            "This combination of multiclass and shuffling isn't implemented"
        )

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print("Not a NP Array, skipping")
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = (
            torch.tensor(X)
            .reshape(2, -1, X.shape[1])
            .transpose(0, 1)
            .reshape(-1, X.shape[1])
            .flip([0])
            .float()
        )
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return (
        X,
        y,
        list(np.where(categorical_indicator)[0]),
        attribute_names + [list(dataset.features.values())[-1].name],
        description,
    )


def load_openml_list(
        dids,
        filter_for_nan=False,
        num_feats=100,
        min_samples=100,
        max_samples=400,
        multiclass=True,
        max_num_classes=10,
        shuffled=True,
        return_capped=False,
):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f"Number of datasets: {len(openml_list)}")

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist["NumberOfInstancesWithMissingValues"] == 0]
        print(
            f"Number of datasets after Nan and feature number filtering: {len(datalist)}"
        )

    for ds in datalist.index:
        modifications = {
            "samples_capped": False,
            "classes_capped": False,
            "feats_capped": False,
        }
        entry = datalist.loc[ds]

        print("Loading", entry["name"], entry.did, "..")

        if entry["NumberOfClasses"] == 0.0:
            raise Exception("Regression not supported")
        else:
            (
                X,
                y,
                categorical_feats,
                attribute_names,
                description,
            ) = get_openml_classification(
                int(entry.did), max_samples, multiclass=multiclass, shuffled=shuffled
            )
        if X is None:
            continue

        if X.shape[1] > num_feats:
            if return_capped:
                X = X[:, 0:num_feats]
                categorical_feats = [c for c in categorical_feats if c < num_feats]
                modifications["feats_capped"] = True
            else:
                print("Too many features")
                continue
        if X.shape[0] == max_samples:
            modifications["samples_capped"] = True

        if X.shape[0] < min_samples:
            print(f"Too few samples left")
            continue

        if len(np.unique(y)) > max_num_classes:
            if return_capped:
                X = X[y < np.unique(y)[10]]
                y = y[y < np.unique(y)[10]]
                modifications["classes_capped"] = True
            else:
                print(f"Too many classes")
                continue

        datasets += [
            [
                entry["name"],
                X,
                y,
                categorical_feats,
                attribute_names,
                modifications,
                description,
            ]
        ]

    return datasets, datalist


def refactor_openml_description(description):
    
    splits = re.split("\n", description)
    blacklist = [
        "Please cite",
        "Author",
        "Source",
        "Author:",
        "Source:",
        "Please cite:",
    ]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n", np.array(splits)[sel].tolist())

    splits = re.split("###", description)
    blacklist = ["Relevant Papers"]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n\n", np.array(splits)[sel].tolist())
    return description


def get_X_y(df_train, target_name):
    y = torch.tensor(df_train[target_name].astype(int).to_numpy())
    x = torch.tensor(df_train.drop(target_name, axis=1).to_numpy())

    return x, y




def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)

    return x, y



def get_data_split(ds, seed):
    def get_df(X, y):
        df = pd.DataFrame(
            data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4]
        )
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    ds = copy.deepcopy(ds)

    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    return ds, df_train, df_test

def load_kaggle():
    cc_test_datasets_multiclass = []
    for name in kaggle_dataset_ids:
        try:
            df_all = pd.read_csv(f"datasets_kaggle/{name[0]}/{name[1]}.csv")
            df_train, df_test = train_test_split(df_all, test_size=0.25, random_state=0)
            ds = [
                "kaggle_" + name[0],
                df_all.copy().drop(columns=[name[2]], inplace=False).values,
                df_all[name[2]].values,
                [],
                df_train.copy().drop(columns=[name[2]], inplace=False).columns.tolist()
                + [name[2]],
                "",
            ]
            data_dir = os.environ.get("DATA_DIR", "data/")
            path = f"{data_dir}/dataset_descriptions/kaggle_{name[0]}.txt"
            try:
                with open(path) as f:
                    ds[-1] = f.read()
            except:
                print("Using initial description")

            cc_test_datasets_multiclass += [ds]
        except:
            print(
                f"{name[0]} at datasets_kaggle/{name[0]}/{name[1]}.csv not found, skipping..."
            )

    for name in kaggle_competition_ids:
        try:
            df_all = pd.read_csv(f"datasets_kaggle/{name}/train.csv")
            df_train, df_test = train_test_split(df_all, test_size=0.25, random_state=0)
            ds = [
                "kaggle_" + name,
                df_all[df_all.columns[:-1]].values,
                df_all[df_all.columns[-1]].values,
                [],
                df_train.columns.tolist(),
                "",
            ]
            path = f"dataset_descriptions/kaggle_{name}.txt"
            try:
                with open(path) as f:
                    ds[-1] = f.read()
            except:
                print("Using initial description")

            cc_test_datasets_multiclass += [ds]
        except:
            print(f"{name} at datasets_kaggle/{name}/train.csv not found, skipping...")

    return cc_test_datasets_multiclass



def load_all_data():
    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(
        benchmark_ids,
        multiclass=True,
        shuffled=True,
        filter_for_nan=False,
        max_samples=10000,
        num_feats=25,
        return_capped=False,
    )

    cc_test_datasets_multiclass += load_kaggle()

    return postprocess_datasets(cc_test_datasets_multiclass)



def postprocess_datasets(cc_test_datasets_multiclass):
    for ds in cc_test_datasets_multiclass:
        dataset_down_size = {
            "balance-scale": 0.2,
            "breast-w": 0.1,
            "tic-tac-toe": 0.1,
        }
        p = dataset_down_size.get(ds[0], 1.0)

        if p < 1.0:
            print(f"Downsampling {ds[0]} to {p * 100}% of samples")

        df = pd.DataFrame(
            np.concatenate([ds[1], ds[2][:, np.newaxis]], 1)
        ).infer_objects()
        if ds[0].startswith("kaggle"):
            # sel = [pd.api.types.is_numeric_dtype(t) for t in df.dtypes]
            # df.loc[:, sel] = df.loc[:, sel].fillna(0)
            # sel = [pd.api.types.is_bool_dtype(t) for t in df.dtypes]
            # df.loc[:, (df.dtypes == object)] = df.loc[:, (df.dtypes != object)].fillna(False)
            df = df.dropna()

        df.loc[:, (df.dtypes == object)] = df.loc[:, (df.dtypes == object)].fillna("")

        l = len(df)
        l = min(l, 2000)
        df = df.sample(frac=1)

        ds[1] = df.values[0: int(p * l), :-1]
        ds[2] = df.values[0: int(p * l), -1]

    return cc_test_datasets_multiclass




kaggle_competition_ids = ["spaceship-titanic", "playground-series-s3e12"]
kaggle_dataset_ids = [ 
    (
        "health-insurance-lead-prediction-raw-data",
        "Health Insurance Lead Prediction Raw Data",
        "Response",
        "owaiskhan9654",
    ),
    ("pharyngitis", "pharyngitis", "radt", "yoshifumimiya"),
]

benchmark_ids = [
    31,
    50
]



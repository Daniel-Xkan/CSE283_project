# task 1: cancer vs normal

import pandas as pd
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
dir = os.path.dirname(cur_dir)
fig_save_path = os.path.join(dir, "cls1_output", "figs")
table_save_path = os.path.join(dir, "cls1_output", "table")


def read_cancer_info():
    """read tpm matrix and labels"""
    current_dir = os.getcwd()
    dir = os.path.dirname(current_dir)
    dst = os.path.join(dir, "data", "postfilter_data", "cancer", "tpm_filtered_ranked_normal_vs_96.txt")
    df = pd.read_csv(dst, sep='\t')
    return df


def read_label():
    current_dir = os.getcwd()
    dir = os.path.dirname(current_dir)
    dst = os.path.join(dir, "data", "postfilter_data", "cancer", "sample_status.txt")
    with open(dst, "r") as f:
        reads = f.readlines()
    reads.pop(0)
    res = []
    for r in reads:

        rs = r.split('\n')[0].split('\t')
        if len(rs) == 2:
            st = 0 if rs[1] == 'normal' else 1
            res.append((rs[0], st))
        else:
            print('sth wrong.')
    return res


def preprocessing(data, k):
    """customize our preprocessing, here, is only subsampling"""
    if isinstance(data, list):
        # top-k significant genes (tpm)
        data = np.array(data)[:k]
    return data


def make_whole_dataset(k):
    X_df = read_cancer_info()
    y_li = read_label()
    X_df = X_df.set_index('gene_id').T
    X_dict = X_df.apply(lambda row: row.tolist(), axis=1).to_dict()

    X = np.zeros(shape=(len(y_li), k))
    y = np.zeros(shape=(len(y_li),))
    record = []

    for id in range(len(y_li)):
        pa = y_li[id][0]
        st = y_li[id][1]
        exps = X_dict[pa]
        exps = preprocessing(exps, k)
        X[id] = exps
        y[id] = st
        record.append(pa)

    return X, y, record


def sample_dataset(X, y, record):
    """random sample //ratio// dataset as training set, keeping the ratio of pos/neg sample the same"""
    ratio = 0.8

    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    r_series = pd.Series(record)

    X_df["label"] = y_series
    X_df["id"] = r_series

    X_1 = X_df[X_df["label"] == 1]
    X_0 = X_df[X_df["label"] == 0]

    n_sample_1 = round(ratio * len(X_1))
    n_sample_0 = round(ratio * len(X_0))
    X_1_train = X_1.sample(n=n_sample_1)
    X_0_train = X_0.sample(n=n_sample_0)

    X_1_test = X_1.drop(X_1_train.index)
    X_0_test = X_0.drop(X_0_train.index)

    X_train = pd.concat([X_1_train, X_0_train])
    y_train = X_train["label"].values
    r_train = X_train["id"].values
    X_train = X_train.drop(columns=["label", "id"])
    X_train = np.array(X_train)

    X_test = pd.concat([X_1_test, X_0_test])
    y_test = X_test["label"].values
    r_test = X_test["id"].values
    X_test = X_test.drop(columns=["label", "id"])
    X_test = np.array(X_test)

    return (X_train, y_train, r_train), (X_test, y_test, r_test)


def evaluate(prob, pred, gt):
    cm = confusion_matrix(gt, pred)
    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(gt, pred)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(gt, pred)
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(gt, pred)
    print(f"F1 Score: {f1:.4f}")

    accuracy = accuracy_score(gt, pred)
    print(f"Accuracy: {accuracy:.4f}")

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "prob": prob,
        "label": gt
    }


class ModelWrapper:
    def __init__(self, name, k):
        self.name = name
        self.model = self.get_model(name)
        self.dim = k

    def get_model(self, name):
        """get the model"""
        distributor = {
            "svm": lambda: SVC(probability=True),
            "random_forest": lambda: RandomForestClassifier(),
            "logic_regression": lambda: LogisticRegression(max_iter=1000),
            "knn": lambda: KNeighborsClassifier(n_neighbors=5),
            "mlp": lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        }
        return distributor.get(name, lambda: None)()


def draw_ROC(wrapper, all_prob, all_gt, n):
    clrs = ['red', 'green', 'blue', 'purple', 'orange', 'gray', 'brown', 'black', 'yellow', 'pink']  # n less than this list's length
    used_clr = clrs[:n]

    auc = 0.0

    for i in range(n):
        fpr, tpr, _ = roc_curve(all_gt[i], all_prob[i])
        roc_auc = roc_auc_score(all_gt[i], all_prob[i])
        auc += roc_auc
        plt.plot(fpr, tpr, color=used_clr[i], label=f'Run {i + 1} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {wrapper.name}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()

    filename = f'ROC_dim_{wrapper.dim}_model_{wrapper.name}.png'
    path = os.path.join(fig_save_path, filename)
    plt.savefig(path)
    plt.close()

    return auc / n


def experiment(wrapper, X, y, r):
    """execute n times and calculate the average metrics"""
    n = 10

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    all_prob_gathered = []
    all_gt_gathered = []

    for i in range(n):
        train_wrapper, test_wrapper = sample_dataset(X, y, r)
        wrapper.model.fit(train_wrapper[0], train_wrapper[1])
        y_prob = wrapper.model.predict_proba(test_wrapper[0])[:, 1]
        y_pred = wrapper.model.predict(test_wrapper[0])  # equals to threshold y_prob by 0.5
        res = evaluate(y_prob, y_pred, test_wrapper[1])
        accuracy += res["accuracy"]
        precision += res["precision"]
        recall += res["recall"]
        f1 += res["f1"]
        all_prob_gathered.append(res["prob"])
        all_gt_gathered.append(res["label"])

    auc = draw_ROC(wrapper, all_prob_gathered, all_gt_gathered, n)

    return {
        "accuracy": accuracy / n,
        "precision": precision / n,
        "recall": recall / n,
        "f1": f1 / n,
        "auc": auc
    }


def save_dict_as_csv(data):
    records = []
    for dim, models in data.items():
        for model_name, metrics in models.items():
            row = {"dim": dim, "model": model_name}
            row.update(metrics)
            records.append(row)
    save = pd.DataFrame(records)
    save_path = os.path.join(table_save_path, "result.csv")
    save.to_csv(save_path, index=False)


if __name__ == '__main__':

    candiK = [10, 15, 25, 40, 60, 100]
    all_res = {}
    for k in candiK:
        X, y, r = make_whole_dataset(k)
        model_list = ["svm", "random_forest", "logic_regression", "knn", "mlp"]
        res = {}
        for i in range(len(model_list)):
            modelWrapper = ModelWrapper(model_list[i], k)
            res[modelWrapper.name] = experiment(modelWrapper, X, y, r)

        all_res[k] = res

    save_dict_as_csv(all_res)

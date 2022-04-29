import argparse
import joblib
import json
import os
from typing import Text
import yaml

#from src.data.dataset import get_dataset
#from src.report.report  import visulazation

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Tuple
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


def get_dataset(dataset_path: Text) -> pd.DataFrame:

    return pd.read_csv(dataset_path)


def evaluate(df: pd.DataFrame, target_column: Text, clf: sklearn.base.BaseEstimator) -> Tuple[float, np.array]:
    
   # Get X and Y
    y_test = df.loc[:, target_column].values.astype("float32")
    X_test = df.drop(target_column, axis=1).values
    X_test = X_test.astype("float32")

    prediction = clf.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')
    cm = confusion_matrix(prediction, y_test)

    return f1, cm




def evaluate_model(config_path: Text):


    with open ("params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)

    target_column = config['featurize']['target_column']
    test_df = get_dataset(config['split_train_test']['test_csv'])
    model_name = config['base']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    model = joblib.load(os.path.join(models_folder, model_name))

    f1, cm = evaluate(df=test_df,
                      target_column=target_column,
                      clf=model)

    test_report = {
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    print(test_report)
    filepath = config['evaluate']['metrics_file']
    json.dump(obj=test_report, fp=open(filepath, 'w'), indent=2)
    print("complete evalution")

    metrics={'f1':f1}
    with open(config["evaluate"]["metrics"],'w') as mf:
        json.dump(obj=metrics,fp=mf,indent=4)

    plt=sns.heatmap(cm, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.figure.savefig(config["evaluate"]["confusion_matrix"])
    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate_model(config_path=args.config)
from statistics import mode
import yaml
import argparse
import pandas as pd 
from typing import Text
import joblib
import os
print("test")
from matplotlib.pyplot import text
from numpy import average
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import f1_score,make_scorer
from typing import Dict, Text


class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):

        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)


def get_supported_estimator():

    return {
        'logreg': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier
    }


def train(df: pd.DataFrame, target_column: Text, estimator_name: Text, param_grid: Dict, cv: int):

    estimators = get_supported_estimator()

    if estimator_name not in estimators.keys():

        raise UnsupportedClassifier(estimator_name)

    estimator = estimators[estimator_name]()
    f1_scorer = make_scorer(f1_score, average='weighted')

    clf = GridSearchCV(estimator=estimator,
                       param_grid =  param_grid,
                       cv=cv,
                       verbose=1,
                       scoring=f1_scorer)

    # Get X and Y
    y_train = df.loc[:, target_column].values.astype("float32")
    X_train = df.drop(target_column, axis=1).values

    clf.fit(X_train, y_train)

    return clf


def train_model(config_path:Text)->None:
    with open ("reports/params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)

    estimator_name=config['train']['estimator_name']
    train_df=pd.read_csv(config['data']['trainset_path'])
    model=train(df=train_df,
                target_column=config['featurize']['target_column'],
                estimator_name=estimator_name,
                param_grid=config['train']['estimators'][estimator_name]['param_grid'],
                cv=config['train']['cv']
                )

    model_name = config['base']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    joblib.dump(
        model,
        os.path.join(models_folder, model_name)
    )


if __name__=='__main__':
    args_parser=argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config',required=True)
    args=args_parser.parse_args()
    train_model(config_path=args.config)

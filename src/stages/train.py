from statistics import mode
import yaml
import argparse
import pandas as pd 
from typing import Text
import joblib

print("test")
from src.train.train import train


def train_model(config_path:Text)->None:
    with open ("reports/params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)

    estimator_name=config['train']['estimator_name']
    train_df=pd.read_csv(config['data']['trainset_path'])
    model=train(df=train_df,
                target_column=config['featurize']['target_column'],
                estimator=estimator_name,
                param_grid=config['train']['estimators'][estimator_name]['param_grid'],
                cv=config['train']['cv']
                )
    model_path=config["train"]["model_path"]
    joblib.dump(model,model_path)

if __name__=='__main__':
    args_parser=argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config',required=True)
    args=args_parser.parse_args()
    train_model(config_path=args.config)

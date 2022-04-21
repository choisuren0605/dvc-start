from statistics import mode
import yaml
import argparse
import pandas as pd 
from typing import Text
import joblib
from scr.utils.logs import get_logger
from scr.train.train import train

def train_model(config_path:Text)->None:
    with open ("reports/params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)
    logger=get_logger("TRAIN",log_level=config["base"]["log_level"])
    logger.info('Get estimator name')
    estimator_name=config(['train'])['estimator_name']
    logger.info(f'Estimator:{estimator_name}')

    logger.info('Load train dataset')
    train_df=pd.read_csv(config['data_split']['trainset_path'])
    logger.info('Train_model')
    model=train(df=train_df,
                target_column=config['featurize']['target_column']
                estimator=estimator_name,
                param_grid=config['train']['estimators'][estimator_name]['param_grid'],
                cv=config['train']['cv']
                )
    logger.info(f'Best.score: {model.best_score_}')
    logger.info('Seve model')
    model_path=config["train"]["model_path"]
    joblib.dump(model,model_path)

if __name__=='__main__':
    args_parser=argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config',required=True)
    args=args_parser.parse_args()
    train_model(config_path=args.config)

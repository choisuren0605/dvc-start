import argparse
from asyncio.log import logger
import pandas as pd 
from typing import Text
import yaml


def featurize(config_path: Text)->None:
    with open ("reports/params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)
    
    dataset=pd.read_csv(config["data"]["dataset_csv"])
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    featured_dataset = dataset[[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
    'target']]

    features_path=config['featurize']['featured_dataset_csv']
    featured_dataset.to_csv(features_path, index=False)
    print("Featurize complete")

if __name__=='__main__':
    args_parser=argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config',required=True)
    args=args_parser.parse_args()
    featurize(config_path=args.config)
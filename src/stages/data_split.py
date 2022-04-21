from operator import index
from path import TempDir
import yaml
import argparse
import pandas as pd 
from typing import Text
from sklearn.model_selection import train_test_split


def data_split(config_path: Text)->None:
    with open ("reports/params.yaml") as conf_file:
        config=yaml.safe_load(conf_file)
    
    dataset=pd.read_csv(config['featurize']['features_path'])
    train_dataset, test_dataset = train_test_split(dataset, test_size=config["data_split"]['test_size'], random_state=42)
    train_path=config["data"]["trainset_path"]
    test_path=config["data"]["testset_path"]
    train_dataset.to_csv(train_path,index=False)
    test_dataset.to_csv(test_path, index=False)
    

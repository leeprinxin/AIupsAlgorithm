import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import KFold, StratifiedKFold
import traceback
import os
import sys
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import traceback
import uuid
import re
import tqdm
from typing import Union
import configparser
import os
train_conf = configparser.ConfigParser()
directory_path = os.path.dirname(os.path.abspath(__file__))
train_conf.read(os.path.join(directory_path, "config.ini"), encoding='utf-8')
is_gpu = train_conf.getint('Train', 'GPU')
gpu_id = train_conf.getint('Train', 'gpu_id')
batch_size = train_conf.getint('BPNN', 'batch_size')
epochs = train_conf.getint('BPNN', 'epochs')
disable_early_stopping_rounds = train_conf.getboolean('BPNN', 'disable_early_stopping_rounds')
patience = train_conf.getint('BPNN', 'patience')
print_interval = train_conf.getint('BPNN', "print_interval")

def numpy_to_pandas(y_true: Union[pd.Series, np.ndarray], x_true: Union[pd.DataFrame, np.ndarray], **kwargs):
    """
    將 y_true 和 x_true 轉換成 pd.Series 與 pd.DataFrame。

    Parameters:
    y_true : pd.Series or np.ndarray
        實際的目標值。
    x_true : pd.DataFrame or np.ndarray
        預測的目標值。

    Returns:
     pd.Series, pd.DataFrame
        轉換後的  pd.Series 與 pd.DataFrame。
    """

    if isinstance(y_true, np.ndarray):
        y_true = y_true.reshape(-1)
        y_true = pd.Series(y_true)

    if isinstance(x_true, np.ndarray):
        x_true = pd.DataFrame(x_true)

    return y_true, x_true
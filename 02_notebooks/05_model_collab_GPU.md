
# Prepare environment


```python
import os
from google.colab import drive
drive.mount('/gdrive')
os.symlink('/gdrive/My Drive', '/content/gdrive')
!ls -l /content/gdrive/

```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /gdrive
    total 36
    drwx------ 2 root root 4096 Oct  3  2018 'Colab Notebooks'
    drwx------ 2 root root 4096 Jun  2  2018  DataScience
    drwx------ 2 root root 4096 Jan 12  2014  Dokumenty
    drwx------ 2 root root 4096 Feb  6  2019  Notability
    drwx------ 2 root root 4096 Feb  6  2019  Notes
    drwx------ 2 root root 4096 May 12 21:02 'Sleep as Android'
    drwx------ 2 root root 4096 Apr 21 16:20 'Trekking '
    drwx------ 2 root root 4096 Jun 14 20:30 'Wedding '
    drwx------ 2 root root 4096 Sep 15  2018  WWW



```python
!ls gdrive/DataScience/Projects/201908_credit/
```

    01_data       99_docs		   README.md	     site-packages
    02_notebooks  docker-compose.yaml  requirements.txt
    98_sys_files  install.sh	   run.sh



```python
#!pip install -r gdrive/DataScience/Projects/201908_credit/requirements.txt
#!pip install -r gdrive/DataScience/Projects/201908_credit/requirements.txt
```


```python
!pip install scikit-learn==0.20.2 --upgrade
!pip install scikit-optimize==0.5.2 --upgrade
!pip install catboost
```

    Requirement already up-to-date: scikit-learn==0.20.2 in /usr/local/lib/python3.6/dist-packages (0.20.2)
    Requirement already satisfied, skipping upgrade: scipy>=0.13.3 in /usr/local/lib/python3.6/dist-packages (from scikit-learn==0.20.2) (1.3.1)
    Requirement already satisfied, skipping upgrade: numpy>=1.8.2 in /usr/local/lib/python3.6/dist-packages (from scikit-learn==0.20.2) (1.16.4)
    Requirement already up-to-date: scikit-optimize==0.5.2 in /usr/local/lib/python3.6/dist-packages (0.5.2)
    Requirement already satisfied, skipping upgrade: numpy in /usr/local/lib/python3.6/dist-packages (from scikit-optimize==0.5.2) (1.16.4)
    Requirement already satisfied, skipping upgrade: scipy>=0.14.0 in /usr/local/lib/python3.6/dist-packages (from scikit-optimize==0.5.2) (1.3.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.19.1 in /usr/local/lib/python3.6/dist-packages (from scikit-optimize==0.5.2) (0.20.2)
    Requirement already satisfied: catboost in /usr/local/lib/python3.6/dist-packages (0.16.5)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.6/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: pandas>=0.19.1 in /usr/local/lib/python3.6/dist-packages (from catboost) (0.24.2)
    Requirement already satisfied: plotly in /usr/local/lib/python3.6/dist-packages (from catboost) (3.6.1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from catboost) (1.12.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from catboost) (3.0.3)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.6/dist-packages (from catboost) (1.16.4)
    Requirement already satisfied: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.19.1->catboost) (2.5.3)
    Requirement already satisfied: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas>=0.19.1->catboost) (2018.9)
    Requirement already satisfied: decorator>=4.0.6 in /usr/local/lib/python3.6/dist-packages (from plotly->catboost) (4.4.0)
    Requirement already satisfied: retrying>=1.3.3 in /usr/local/lib/python3.6/dist-packages (from plotly->catboost) (1.3.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from plotly->catboost) (2.21.0)
    Requirement already satisfied: nbformat>=4.2 in /usr/local/lib/python3.6/dist-packages (from plotly->catboost) (4.4.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (2.4.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (1.1.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->plotly->catboost) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->plotly->catboost) (2.8)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->plotly->catboost) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->plotly->catboost) (2019.6.16)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2->plotly->catboost) (2.6.0)
    Requirement already satisfied: traitlets>=4.1 in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2->plotly->catboost) (4.3.2)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2->plotly->catboost) (4.5.0)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2->plotly->catboost) (0.2.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from kiwisolver>=1.0.1->matplotlib->catboost) (41.1.0)



```python
import feather
import os
import warnings
import numpy as np
import pandas as pd
import json
import codecs
import gc 
import sys
from time import time
import itertools

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

# models
from catboost import CatBoostClassifier


# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer

# Data prepare
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline, make_union
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

warnings.filterwarnings("ignore")
seed = 666
```


```python
sys.path.append('gdrive/DataScience/Projects/201908_credit/02_notebooks')
import helpers
```

# Load Data


```python
data_DIR = Path('gdrive/DataScience/Projects/201908_credit/01_data')

df_train = feather.read_dataframe(os.path.join(data_DIR, 'DS_loans_IN_train.feather')).reset_index(drop=True)
df_test = feather.read_dataframe(os.path.join(data_DIR, 'DS_loans_IN_test.feather')).reset_index(drop=True)

df_variables = feather.read_dataframe(os.path.join(data_DIR, 'variables_primary_selection.feather'))

```


```python
#!!! RESET INDEX
train_sample_size = 30000
train_sample_size_tune = int(train_sample_size/3)
test_sample_size = 5000
test_sample_size_tune = int(test_sample_size/3)

df_train = df_train.sample(train_sample_size, random_state=seed).reset_index(drop=True)

df_train_tune = df_train.sample(train_sample_size_tune, random_state=seed).reset_index(drop=True)

```


```python
replacement = {'emp_length':
                  {'< 1 year':0, '9 years':9, '3 years':3, '10+ years':11, '7 years':7,
                   '2 years':2, '4 years':4, '1 year':1, '8 years':8, '5 years':5, None:np.nan,
                   '6 years':6}}
df_train = df_train.replace(replacement)
df_test = df_test.replace(replacement)
```


```python
df_train.replace({'desc': {'': np.nan},'emp_title': {'': np.nan},'title': {'': np.nan}}, inplace=True)
df_train.fillna({'desc': 'novalue', 'emp_title': 'novalue', 'title': 'novalue'}, inplace=True)

df_test.replace({'desc': {'': np.nan},'emp_title': {'': np.nan},'title': {'': np.nan}}, inplace=True)
df_test.fillna({'desc': 'novalue', 'emp_title': 'novalue', 'title': 'novalue'}, inplace=True)

```

REMOVE LOAN STATUS!!!


```python
df_train = df_train.drop(['loan_status','emp_title','desc','title','earliest_cr_line','issue_d','grade','int_rate'],axis=1)
df_test = df_test.drop(['loan_status','emp_title','desc','title','earliest_cr_line','issue_d','grade','int_rate'],axis=1)


```


```python
# convert text to category 
for col in df_train.select_dtypes(include=['object']).columns:
    df_train[col] = df_train[col].replace([' '],['nocat']).fillna('nocat').astype('category')

for col in df_test.select_dtypes(include=['object']).columns:
    df_test[col] = df_test[col].replace([' '],['nocat']).fillna('nocat').astype('category')

```


```python
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper

```


```python
# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    best_score = optimizer.best_score_
    best_score_std = optimizer.cv_results_['std_test_score'][optimizer.best_index_]
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params
```


```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
```


```python
# Converting average precision score into a scorer suitable for model selection
roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

avg_prec = make_scorer(average_precision_score, greater_is_better=True, needs_proba=True)
```


```python
category_features = df_train.select_dtypes(include='category').columns.tolist()
```


```python
#clf = CatBoostClassifier(loss_function='Logloss',
#                         custom_metric = ['Logloss', 'AUC'],
#                         #eval_metric = 'AUC',
#                         task_type='GPU',
#                         cat_features = category_features,
#                         verbose = False)
```


```python
clf = CatBoostClassifier(loss_function='Logloss',
                         task_type='GPU',
                         cat_features = category_features,
                         verbose = False)
```


```python
search_spaces = {'iterations': Integer(10, 300),
                 'depth': Integer(1, 8),
                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 #'border_count': Integer(60, 180),
                 #'ctr_border_count': Integer(60, 180),
                 'l2_leaf_reg': Integer(2, 30),
                 'scale_pos_weight':Real(0.01, 1.0, 'uniform')}
```


```python
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=roc_auc,
                    cv=skf,
                    n_iter=20,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=42)
```


```python
best_params = report_perf(opt, df_train.drop('default', axis=1), df_train.default.to_list(),'CatBoost', 
                          callbacks=[VerboseCallback(1), 
                                     DeadlineStopper(60*30)])
```

    Iteration No: 1 started. Searching for the next optimal point.
    Iteration No: 1 ended. Search finished for the next optimal point.
    Time taken: 103.0042
    Function value obtained: -0.6939
    Current minimum: -0.6939
    Iteration No: 2 ended. Search finished for the next optimal point.
    Time taken: 159.0729
    Function value obtained: -0.6903
    Current minimum: -0.6939
    Iteration No: 3 ended. Search finished for the next optimal point.
    Time taken: 208.0402
    Function value obtained: -0.6952
    Current minimum: -0.6952
    Iteration No: 4 ended. Search finished for the next optimal point.
    Time taken: 264.2281
    Function value obtained: -0.6990
    Current minimum: -0.6990
    Iteration No: 5 ended. Search finished for the next optimal point.
    Time taken: 321.0151
    Function value obtained: -0.6828
    Current minimum: -0.6990
    Iteration No: 6 ended. Search finished for the next optimal point.
    Time taken: 373.6319
    Function value obtained: -0.6813
    Current minimum: -0.6990
    Iteration No: 7 ended. Search finished for the next optimal point.
    Time taken: 427.9183
    Function value obtained: -0.6998
    Current minimum: -0.6998
    Iteration No: 8 ended. Search finished for the next optimal point.
    Time taken: 489.0563
    Function value obtained: -0.6998
    Current minimum: -0.6998
    Iteration No: 9 ended. Search finished for the next optimal point.
    Time taken: 584.0361
    Function value obtained: -0.6885
    Current minimum: -0.6998
    Iteration No: 10 ended. Search finished for the next optimal point.
    Time taken: 684.6574
    Function value obtained: -0.6997
    Current minimum: -0.6998
    Iteration No: 11 ended. Search finished for the next optimal point.
    Time taken: 748.6502
    Function value obtained: -0.6863
    Current minimum: -0.6998
    Iteration No: 12 ended. Search finished for the next optimal point.
    Time taken: 810.3563
    Function value obtained: -0.6979
    Current minimum: -0.6998
    Iteration No: 13 ended. Search finished for the next optimal point.
    Time taken: 856.3104
    Function value obtained: -0.6487
    Current minimum: -0.6998
    Iteration No: 14 ended. Search finished for the next optimal point.
    Time taken: 995.9110
    Function value obtained: -0.6154
    Current minimum: -0.6998
    Iteration No: 15 ended. Search finished for the next optimal point.
    Time taken: 1042.7665
    Function value obtained: -0.6505
    Current minimum: -0.6998
    Iteration No: 16 ended. Search finished for the next optimal point.
    Time taken: 1097.3144
    Function value obtained: -0.6993
    Current minimum: -0.6998
    Iteration No: 17 ended. Search finished for the next optimal point.
    Time taken: 1231.7775
    Function value obtained: -0.6824
    Current minimum: -0.6998
    Iteration No: 18 ended. Search finished for the next optimal point.
    Time taken: 1279.3232
    Function value obtained: -0.6773
    Current minimum: -0.6998
    Iteration No: 19 ended. Search finished for the next optimal point.
    Time taken: 1364.9614
    Function value obtained: -0.6715
    Current minimum: -0.6998
    Iteration No: 20 ended. Search finished for the next optimal point.
    Time taken: 1428.4966
    Function value obtained: -0.6923
    Current minimum: -0.6998
    CatBoost took 1440.77 seconds,  candidates checked: 20, best CV score: 0.700 ± 0.011
    Best parameters:
    {'bagging_temperature': 0.5434030676903125,
     'depth': 7,
     'iterations': 154,
     'l2_leaf_reg': 25,
     'learning_rate': 0.04447541043186938,
     'random_strength': 3.245977736555784e-09,
     'scale_pos_weight': 0.5750700246521092}
    



```python
from functools import partial
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective

```

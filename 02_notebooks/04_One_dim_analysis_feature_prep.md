

```python
import feather
import os
from pathlib import Path
import pandas as pd
import numpy as np
import category_encoders
import re
import gc

from pandas.api.types import CategoricalDtype
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```

# Load data 


```python
%%time

data_DIR = Path('/Users/xszpo/Google Drive/DataScience/Projects/201908_credit/'
                '01_data')

df_train = feather.read_dataframe(os.path.join(data_DIR, 'DS_loans_IN_train.feather'))

df_test = feather.read_dataframe(os.path.join(data_DIR, 'DS_loans_IN_test.feather'))

df_variables = feather.read_dataframe(os.path.join(data_DIR, 'variables_primary_selection.feather'))

```

    CPU times: user 1.24 s, sys: 639 ms, total: 1.88 s
    Wall time: 1.23 s



```python
print(df_train.shape)
print(df_test.shape)
print(df_variables.shape)
```

    (714794, 93)
    (306533, 93)
    (84, 2)


Select top 20 features to build model


```python
features = df_variables.loc[df_variables.index<20]
features_list = features['feature_name'].tolist()
```


```python
df_train = df_train[features_list+['issue_d','default','desc','title']]
df_test = df_test[features_list+['issue_d','default','desc','title']]
```


```python
df_train.columns
```




    Index(['zip_code', 'sub_grade', 'acc_open_past_24mths', 'int_rate',
           'avg_cur_bal', 'dti', 'emp_length', 'addr_state', 'annual_inc',
           'mo_sin_old_rev_tl_op', 'home_ownership', 'bc_util', 'mort_acc',
           'tot_cur_bal', 'bc_open_to_buy', 'term', 'mths_since_recent_bc',
           'total_bc_limit', 'loan_amnt', 'emp_title', 'issue_d', 'default',
           'desc', 'title'],
          dtype='object')




```python
replacement = {'emp_length':
                  {'< 1 year':0, '9 years':9, '3 years':3, '10+ years':11, '7 years':7,
                   '2 years':2, '4 years':4, '1 year':1, '8 years':8, '5 years':5, None:np.nan,
                   '6 years':6}}
df_train = df_train.replace(replacement)
df_test = df_test.replace(replacement)

```


```python
target = ['default']
features_text = ['emp_title','desc','title']
features_category = [i for i in df_train.select_dtypes(include=['object']).columns if i not in features_text]
features_numeric=  [i for i in df_train.select_dtypes(exclude=['object','datetime64[ns]']).columns if i not in ['default']]
features_data =  list(df_train.select_dtypes(include=['datetime64[ns]']).columns)

print("Text features ({}): {} \n".format(len(features_text),", ".join(features_text)))
print("Category features ({}): {} \n".format(len(features_category),", ".join(features_category)))
print("Numeric features ({}): {} \n".format(len(features_numeric),", ".join(features_numeric)))
print("Datetime features ({}): {} \n".format(len(features_data),", ".join(features_data)))

```

    Text features (3): emp_title, desc, title 
    
    Category features (5): zip_code, sub_grade, addr_state, home_ownership, term 
    
    Numeric features (14): acc_open_past_24mths, int_rate, avg_cur_bal, dti, emp_length, annual_inc, mo_sin_old_rev_tl_op, bc_util, mort_acc, tot_cur_bal, bc_open_to_buy, mths_since_recent_bc, total_bc_limit, loan_amnt 
    
    Datetime features (1): issue_d 
    



```python
df_train.select_dtypes(include=['object']).columns
```




    Index(['zip_code', 'sub_grade', 'addr_state', 'home_ownership', 'term',
           'emp_title', 'desc', 'title'],
          dtype='object')



# One dimensional analysis


```python
from collections import Counter

def calculate_IV(column, default_column = 'default', default_val=1, nondefault_val=0, df=df_train, ifprt = True):
    """
    calculate IV - temporary implementation
    """
    total = df.shape[0]
    total_event = np.sum(df[default_column]==default_val)
    total_nonevent= np.sum(df[default_column]==nondefault_val)

    def total_prc(x): return np.round(len(x)/total,4)
    def event_prc(x): return  np.round(len([i for i in list(x) if i==default_val])/total_event,4)
    def nonevent_prc(x): return  np.round(len([i for i in list(x) if i==nondefault_val])/total_nonevent,4)
    def woe(x): return np.log(nonevent_prc(x)/(event_prc(x)+np.finfo(float).eps))
    def iv(x): return (nonevent_prc(x)-event_prc(x))*woe(x)

    wyn = df[[column,default_column]].groupby(column).agg(
        ['count',total_prc,event_prc, nonevent_prc, woe,iv
        ])

    if ifprt:
        print(wyn)
    IV = np.sum(wyn.iloc[:,-1])
    if ifprt:
        print("\n IV value of variable '{}' is {}".format(column,np.round(IV,4)))
    return IV, wyn
```


```python
features_numeric
```




    ['acc_open_past_24mths',
     'int_rate',
     'avg_cur_bal',
     'dti',
     'emp_length',
     'annual_inc',
     'mo_sin_old_rev_tl_op',
     'bc_util',
     'mort_acc',
     'tot_cur_bal',
     'bc_open_to_buy',
     'mths_since_recent_bc',
     'total_bc_limit',
     'loan_amnt']




```python
IV_cat = {}
repl_dic = {}
```

## Category Features


```python
features_category
```




    ['zip_code', 'sub_grade', 'addr_state', 'home_ownership', 'term']



### zip_code


```python
len(df_train['zip_code'].unique())
```




    928



#### Deep embedding for categorical variables (Cat2Vec)


```python
from keras import models
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense

```

    Using TensorFlow backend.



```python
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
feat_zip_code = le1.fit_transform(pd.concat([df_train,df_test],axis=0)['zip_code'])

```


```python
embedding_size = 10
model = models.Sequential()
model.add(Embedding(input_dim = len(np.unique(feat_zip_code)), output_dim = embedding_size, input_length = 1, name="embedding"))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
model.compile(loss = "mse", optimizer = "adam", metrics=["accuracy"])
model.fit(
    x = feat_zip_code, 
    y = pd.concat([df_train,df_test],axis=0)['default'].values  , 
    epochs = 10, batch_size = 512)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0822 22:40:38.154908 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0822 22:40:38.218895 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0822 22:40:38.221333 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0822 22:40:38.281878 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    W0822 22:40:38.806745 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    W0822 22:40:38.931548 4533990848 deprecation_wrapper.py:119] From /Users/xszpo/anaconda3/envs/lending_club/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    


    Epoch 1/10
    1021327/1021327 [==============================] - 4s 4us/step - loss: 0.1438 - acc: 0.8257
    Epoch 2/10
    1021327/1021327 [==============================] - 6s 6us/step - loss: 0.1435 - acc: 0.8257
    Epoch 3/10
    1021327/1021327 [==============================] - 8s 8us/step - loss: 0.1435 - acc: 0.8257
    Epoch 4/10
    1021327/1021327 [==============================] - 5s 5us/step - loss: 0.1434 - acc: 0.8257
    Epoch 5/10
    1021327/1021327 [==============================] - 4s 4us/step - loss: 0.1434 - acc: 0.8257
    Epoch 6/10
    1021327/1021327 [==============================] - 3s 3us/step - loss: 0.1434 - acc: 0.8257
    Epoch 7/10
    1021327/1021327 [==============================] - 3s 3us/step - loss: 0.1434 - acc: 0.8257
    Epoch 8/10
    1021327/1021327 [==============================] - 3s 3us/step - loss: 0.1434 - acc: 0.8257
    Epoch 9/10
    1021327/1021327 [==============================] - 3s 3us/step - loss: 0.1434 - acc: 0.8257
    Epoch 10/10
    1021327/1021327 [==============================] - 3s 3us/step - loss: 0.1434 - acc: 0.8257





    <keras.callbacks.History at 0x121981c50>




```python
model.summary() 
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 1, 10)             9420      
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 10)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 50)                550       
    _________________________________________________________________
    dense_2 (Dense)              (None, 15)                765       
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 16        
    =================================================================
    Total params: 10,751
    Trainable params: 10,751
    Non-trainable params: 0
    _________________________________________________________________



```python
layer = model.get_layer('embedding')
output_embeddings = layer.get_weights()
output_embeddings[0].shape
```




    (942, 10)




```python
from sklearn.cluster import DBSCAN, KMeans
```


```python
from sklearn.cluster import  KMeans
clustering = KMeans(n_clusters=5).fit(output_embeddings[0])
len(np.unique(clustering.labels_))
```




    5




```python
cat_repl_zip_code = {}
for i in zip(le1.classes_,clustering.labels_):
    cat_repl_zip_code[i[0]]=int(i[1])

repl_dic['zip_code'] = cat_repl_zip_code
```


```python
IV_cat['zip_code'], _ = calculate_IV('zip_code', default_column = 'default', default_val=1, nondefault_val=0, 
                 df=df_train[['zip_code','default']].replace(repl_dic['zip_code']))
```

             default                                                     
               count total_prc event_prc nonevent_prc       woe        iv
    zip_code                                                             
    0          98897    0.1384    0.1702       0.1316 -0.257207  0.009928
    1         386101    0.5402    0.5615       0.5356 -0.047224  0.001223
    2          54419    0.0761    0.0553       0.0805  0.375484  0.009462
    3         174244    0.2438    0.2123       0.2504  0.165059  0.006289
    4           1133    0.0016    0.0007       0.0018  0.944462  0.001039
    
     IV value of variable 'zip_code' is 0.0279



```python
IV_cat
```




    {'zip_code': 0.027941173296134353}



## sub_grade


```python
IV_cat['sub_grade'], _ = calculate_IV('sub_grade')
```

              default                                                     
                count total_prc event_prc nonevent_prc       woe        iv
    sub_grade                                                             
    A1          21642    0.0303    0.0048       0.0356  2.003730  0.061715
    A2          18659    0.0261    0.0065       0.0302  1.536040  0.036404
    A3          19019    0.0266    0.0075       0.0306  1.406097  0.032481
    A4          27076    0.0379    0.0133       0.0431  1.175759  0.035038
    A5          35699    0.0499    0.0212       0.0560  0.971351  0.033803
    B1          36598    0.0512    0.0263       0.0565  0.764672  0.023093
    B2          39459    0.0552    0.0318       0.0601  0.636544  0.018014
    B3          45323    0.0634    0.0412       0.0681  0.502539  0.013518
    B4          45031    0.0630    0.0465       0.0665  0.357750  0.007155
    B5          39848    0.0557    0.0457       0.0579  0.236619  0.002887
    C1          43701    0.0611    0.0565       0.0621  0.094505  0.000529
    C2          42043    0.0588    0.0605       0.0585 -0.033617  0.000067
    C3          40525    0.0567    0.0627       0.0554 -0.123782  0.000904
    C4          39166    0.0548    0.0667       0.0523 -0.243209  0.003502
    C5          32707    0.0458    0.0581       0.0432 -0.296325  0.004415
    D1          28515    0.0399    0.0532       0.0371 -0.360441  0.005803
    D2          23311    0.0326    0.0477       0.0294 -0.483937  0.008856
    D3          20515    0.0287    0.0429       0.0257 -0.512381  0.008813
    D4          19978    0.0279    0.0449       0.0244 -0.609855  0.012502
    D5          16658    0.0233    0.0383       0.0201 -0.644730  0.011734
    E1          14552    0.0204    0.0358       0.0171 -0.738869  0.013817
    E2          13384    0.0187    0.0345       0.0154 -0.806592  0.015406
    E3          11342    0.0159    0.0303       0.0128 -0.861703  0.015080
    E4           9389    0.0131    0.0262       0.0104 -0.923954  0.014598
    E5           7640    0.0107    0.0221       0.0083 -0.979322  0.013515
    F1           5827    0.0082    0.0170       0.0063 -0.992664  0.010622
    F2           4282    0.0060    0.0135       0.0044 -1.121085  0.010202
    F3           3501    0.0049    0.0113       0.0035 -1.172040  0.009142
    F4           2769    0.0039    0.0097       0.0027 -1.278874  0.008952
    F5           2149    0.0030    0.0076       0.0020 -1.335001  0.007476
    G1           1497    0.0021    0.0051       0.0015 -1.223775  0.004406
    G2           1147    0.0016    0.0038       0.0011 -1.239691  0.003347
    G3            792    0.0011    0.0029       0.0007 -1.421386  0.003127
    G4            553    0.0008    0.0021       0.0005 -1.435085  0.002296
    G5            497    0.0007    0.0018       0.0005 -1.280934  0.001665
    
     IV value of variable 'sub_grade' is 0.4549


Check if resaults from scikit-learn-contrib, category_encoders.WOEEncoder are similar to my.   
They ARE similar to my resault, so I could use ready implementation to my pipeline.



```python
enc = category_encoders.WOEEncoder(cols=['sub_grade_woe'], randomized=False)
enc.fit_transform(df_train.assign(sub_grade_woe = lambda x:x['sub_grade'])[['sub_grade_woe','sub_grade']],
                  df_train['default'].values).groupby(['sub_grade']).agg('max').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_grade_woe</th>
    </tr>
    <tr>
      <th>sub_grade</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A1</th>
      <td>-1.998251</td>
    </tr>
    <tr>
      <th>A2</th>
      <td>-1.540892</td>
    </tr>
    <tr>
      <th>A3</th>
      <td>-1.412218</td>
    </tr>
    <tr>
      <th>A4</th>
      <td>-1.171590</td>
    </tr>
    <tr>
      <th>A5</th>
      <td>-0.969765</td>
    </tr>
  </tbody>
</table>
</div>



## addr_state


```python
IV_cat['addr_state'], dat = calculate_IV('addr_state', df=df_train)
```

               default                                                         
                 count total_prc event_prc nonevent_prc           woe        iv
    addr_state                                                                 
    AK            1781    0.0025    0.0026       0.0025 -3.922071e-02  0.000004
    AL            8909    0.0125    0.0151       0.0119 -2.381563e-01  0.000762
    AR            5327    0.0075    0.0090       0.0071 -2.371298e-01  0.000451
    AZ           16550    0.0232    0.0235       0.0231 -1.716780e-02  0.000007
    CA          103243    0.1444    0.1441       0.1445  2.772005e-03  0.000001
    CO           15121    0.0212    0.0171       0.0220  2.519640e-01  0.001235
    CT           10758    0.0151    0.0128       0.0155  1.913949e-01  0.000517
    DC            1915    0.0027    0.0016       0.0029  5.947071e-01  0.000773
    DE            2065    0.0029    0.0028       0.0029  3.509132e-02  0.000004
    FL           49202    0.0688    0.0742       0.0677 -9.167797e-02  0.000596
    GA           23529    0.0329    0.0301       0.0335  1.070203e-01  0.000364
    HI            3665    0.0051    0.0054       0.0051 -5.715841e-02  0.000017
    IA              10    0.0000    0.0000       0.0000          -inf       NaN
    ID             251    0.0004    0.0003       0.0004  2.876821e-01  0.000029
    IL           28670    0.0401    0.0353       0.0411  1.521252e-01  0.000882
    IN           11312    0.0158    0.0169       0.0156 -8.004271e-02  0.000104
    KS            6408    0.0090    0.0073       0.0093  2.421401e-01  0.000484
    KY            6938    0.0097    0.0106       0.0095 -1.095622e-01  0.000121
    LA            8548    0.0120    0.0133       0.0117 -1.281752e-01  0.000205
    MA           16472    0.0230    0.0215       0.0234  8.468309e-02  0.000161
    MD           16701    0.0234    0.0241       0.0232 -3.805956e-02  0.000034
    ME             719    0.0010    0.0007       0.0011  4.519851e-01  0.000181
    MI           18405    0.0257    0.0262       0.0257 -1.926842e-02  0.000010
    MN           12735    0.0178    0.0176       0.0179  1.690181e-02  0.000005
    MO           11522    0.0161    0.0170       0.0159 -6.689423e-02  0.000074
    MS            3339    0.0047    0.0058       0.0044 -2.762534e-01  0.000387
    MT            2094    0.0029    0.0025       0.0030  1.823216e-01  0.000091
    NC           19968    0.0279    0.0290       0.0277 -4.586342e-02  0.000060
    ND             581    0.0008    0.0008       0.0008 -2.775558e-13 -0.000000
    NE            1263    0.0018    0.0021       0.0017 -2.113091e-01  0.000085
    NH            3464    0.0048    0.0035       0.0051  3.764776e-01  0.000602
    NJ           26736    0.0374    0.0392       0.0370 -5.775883e-02  0.000127
    NM            4029    0.0056    0.0060       0.0056 -6.899287e-02  0.000028
    NV           10077    0.0141    0.0164       0.0136 -1.872115e-01  0.000524
    NY           59131    0.0827    0.0894       0.0813 -9.497467e-02  0.000769
    OH           23913    0.0335    0.0351       0.0331 -5.866785e-02  0.000117
    OK            6626    0.0093    0.0110       0.0089 -2.118440e-01  0.000445
    OR            8794    0.0123    0.0094       0.0129  3.165176e-01  0.001108
    PA           25046    0.0350    0.0363       0.0348 -4.220035e-02  0.000063
    RI            3085    0.0043    0.0037       0.0044  1.732717e-01  0.000121
    SC            8650    0.0121    0.0101       0.0125  2.131932e-01  0.000512
    SD            1511    0.0021    0.0024       0.0021 -1.335314e-01  0.000040
    TN           10608    0.0148    0.0161       0.0146 -9.779774e-02  0.000147
    TX           57727    0.0808    0.0783       0.0813  3.759841e-02  0.000113
    UT            4967    0.0069    0.0066       0.0070  5.884050e-02  0.000024
    VA           21025    0.0294    0.0296       0.0294 -6.779687e-03  0.000001
    VT            1448    0.0020    0.0015       0.0021  3.364722e-01  0.000202
    WA           15436    0.0216    0.0178       0.0224  2.298625e-01  0.001057
    WI            9452    0.0132    0.0123       0.0134  8.565544e-02  0.000094
    WV            3441    0.0048    0.0040       0.0050  2.231436e-01  0.000223
    WY            1627    0.0023    0.0020       0.0023  1.397619e-01  0.000042
    
     IV value of variable 'addr_state' is 0.014



```python
dat = pd.DataFrame.from_dict({'addr_state':list(dat.index),'woe':list(dat.iloc[:,-2])})
```


```python
import plotly.graph_objects as go


fig = go.Figure(data=go.Choropleth(
    locations=dat['addr_state'], # Spatial coordinates
    z = dat['woe'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'RdBu',
    #Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,
    #Hot,Blackbody,Earth,Electric,Viridis,Cividis
    colorbar_title = "WOE",
))

fig.update_layout(
    title_text = 'WOE by state',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v1.49.1
* Copyright 2012-2019, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



<div>
        
        
            <div id="520cd1a2-9ade-4774-a0b4-79467f143fd1" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("520cd1a2-9ade-4774-a0b4-79467f143fd1")) {
                    Plotly.newPlot(
                        '520cd1a2-9ade-4774-a0b4-79467f143fd1',
                        [{"colorbar": {"title": {"text": "WOE"}}, "colorscale": [[0.0, "rgb(103,0,31)"], [0.1, "rgb(178,24,43)"], [0.2, "rgb(214,96,77)"], [0.3, "rgb(244,165,130)"], [0.4, "rgb(253,219,199)"], [0.5, "rgb(247,247,247)"], [0.6, "rgb(209,229,240)"], [0.7, "rgb(146,197,222)"], [0.8, "rgb(67,147,195)"], [0.9, "rgb(33,102,172)"], [1.0, "rgb(5,48,97)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "type": "choropleth", "z": [-0.039220713153366595, -0.23815634370340955, -0.23712979328897416, -0.01716780362237493, 0.0027720045470087202, 0.2519639898496887, 0.19139485299961204, 0.5947071077465539, 0.03509131981119066, -0.09167797025510853, 0.1070202670761834, -0.057158413839989666, null, 0.28768207245104094, 0.15212515756293155, -0.08004270767354948, 0.2421400520048343, -0.10956220251154734, -0.1281751934240143, 0.08468308723002849, -0.03805956182435436, 0.45198512374274014, -0.01926841886588552, 0.01690181080259059, -0.06689423483004328, -0.27625337662819627, 0.18232155679386577, -0.04586341679318898, -2.7755575615632766e-13, -0.21130909366731265, 0.37647757123484865, -0.05775883415219806, -0.06899287148698854, -0.18721154208816013, -0.09497466562570603, -0.05866784808880933, -0.21184399606029658, 0.31651762209164463, -0.04220035449038255, 0.17327172127397664, 0.21319322046101977, -0.1335313926246151, -0.09779774327614034, 0.03759841355700459, 0.058840500022899894, -0.006779686985386293, 0.3364722366210649, 0.22986250156294216, 0.08565544457847588, 0.22314355131415434, 0.1397619423750476]}],
                        {"geo": {"scope": "usa"}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "WOE by state"}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('520cd1a2-9ade-4774-a0b4-79467f143fd1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


## home_ownership


```python
IV_cat['home_ownership'], dat = calculate_IV('home_ownership')
```

                   default                                                 \
                     count total_prc event_prc nonevent_prc           woe   
    home_ownership                                                          
    ANY                  3    0.0000    0.0000       0.0000          -inf   
    MORTGAGE        357394    0.5000    0.4365       0.5134  1.622669e-01   
    NONE                35    0.0000    0.0000       0.0001  2.683331e+01   
    OTHER              126    0.0002    0.0002       0.0002 -1.110223e-12   
    OWN              72552    0.1015    0.1037       0.1010 -2.638160e-02   
    RENT            284684    0.3983    0.4596       0.3853 -1.763343e-01   
    
                              
                          iv  
    home_ownership            
    ANY                  NaN  
    MORTGAGE        0.012478  
    NONE            0.002683  
    OTHER          -0.000000  
    OWN             0.000071  
    RENT            0.013102  
    
     IV value of variable 'home_ownership' is 0.0283


## term


```python
IV_cat['term'], dat = calculate_IV('term')
```

               default                                                     
                 count total_prc event_prc nonevent_prc       woe        iv
    term                                                                   
     36 months  501467    0.7016    0.5707       0.7292  0.245084  0.038846
     60 months  213327    0.2984    0.4293       0.2708 -0.460775  0.073033
    
     IV value of variable 'term' is 0.1119


## Replacements - DICTIONARY


```python
import json
import codecs
with codecs.open(os.path.join(data_DIR, 'replacement_dictionary.json'),'w') as f:
    json.dump(repl_dic,f)
```

# Numeric features


```python
gc.collect()
```




    441




```python
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16,26))
gs = matplotlib.gridspec.GridSpec(nrows=5, 
                       ncols=3, 
                       figure=fig, 
                       wspace=0.3,
                       hspace=0.3)

ax_dic = {}

counter = 0
for row in range(5):
    for col in range(3):
        if counter < len(features_numeric):
            col_name = features_numeric[counter]
            ax_dic[counter] = fig.add_subplot(gs[row, col])
            sns.regplot(x=col_name, y="default",logistic=True,y_jitter=.03,
                        data=df_train.sample(1000, random_state=666), x_bins=10, ax=ax_dic[counter])
            ax_dic[counter].set_title(col_name,fontsize=18)
            counter +=1
        
```


![png](04_One_dim_analysis_feature_prep_files/04_One_dim_analysis_feature_prep_46_0.png)



```python
percent_missing = round(df_train[features_numeric].isnull().sum() * 100 / df_train.shape[0],2)
min_ = df_train[features_numeric].min()
max_ = df_train[features_numeric].max()

missing_value_df = pd.DataFrame({'column_name': features_numeric,
                                 'percent_missing': percent_missing,
                                'min':min_,
                                 'max':max_
                                })
missing_value_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_name</th>
      <th>percent_missing</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_open_past_24mths</th>
      <td>acc_open_past_24mths</td>
      <td>4.89</td>
      <td>0.00</td>
      <td>64.00</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>int_rate</td>
      <td>0.00</td>
      <td>5.32</td>
      <td>28.99</td>
    </tr>
    <tr>
      <th>avg_cur_bal</th>
      <td>avg_cur_bal</td>
      <td>6.86</td>
      <td>0.00</td>
      <td>958084.00</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>dti</td>
      <td>0.00</td>
      <td>-1.00</td>
      <td>999.00</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>emp_length</td>
      <td>5.26</td>
      <td>0.00</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>annual_inc</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9550000.00</td>
    </tr>
    <tr>
      <th>mo_sin_old_rev_tl_op</th>
      <td>mo_sin_old_rev_tl_op</td>
      <td>6.86</td>
      <td>3.00</td>
      <td>852.00</td>
    </tr>
    <tr>
      <th>bc_util</th>
      <td>bc_util</td>
      <td>5.84</td>
      <td>0.00</td>
      <td>339.60</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>mort_acc</td>
      <td>4.89</td>
      <td>0.00</td>
      <td>52.00</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>tot_cur_bal</td>
      <td>6.86</td>
      <td>0.00</td>
      <td>8000078.00</td>
    </tr>
    <tr>
      <th>bc_open_to_buy</th>
      <td>bc_open_to_buy</td>
      <td>5.78</td>
      <td>0.00</td>
      <td>559912.00</td>
    </tr>
    <tr>
      <th>mths_since_recent_bc</th>
      <td>mths_since_recent_bc</td>
      <td>5.72</td>
      <td>0.00</td>
      <td>616.00</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>total_bc_limit</td>
      <td>4.89</td>
      <td>0.00</td>
      <td>1105500.00</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>loan_amnt</td>
      <td>0.00</td>
      <td>500.00</td>
      <td>40000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = [3]+[3]+[float("inf")]+[-float("inf")]
sorted(list(set(x)))


```




    [-inf, 3, inf]




```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress
import pandas as pd


class DecisionTreeDiscretizer_DF(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_depth = 3, min_samples_prc_leaf=0.1, bins = None, **kwargs):
        self.max_depth = max_depth
        self.min_samples_prc_leaf = min_samples_prc_leaf
        self.bins = bins
        self.kwargs = kwargs
        
    def fit(self, x, y = None):
        
        #type(x)!=pd.core.frame.DataFrame:
        #    raise ValueError('{} works only with Pandas Data Frame') 
        
        if type(x)==pd.core.frame.DataFrame:
            self.columnNames = x.columns
            self.numberofcolumns = x.shape[1]
            
        if type(x)==pd.core.series.Series:
            self.columnNames = [x.name]
            self.numberofcolumns = 1

        if type(y)==list:
            min_samples_leaf = int(self.min_samples_prc_leaf*len(y))
        else:
            min_samples_leaf = int(self.min_samples_prc_leaf*y.shape[0])
                        
        self.trees = {}
        
        if not self.bins:
            self.bins = {}
        
        for nr_col,name in enumerate(self.columnNames):
            if name not in self.bins.keys():
                self.bins[name] = {}
            
            #self.bins[nr_col]['name'] = self.columnNames[nr_col]
            
            if self.numberofcolumns ==1:
                _df = x.copy()
            else:
                _df = x[name].copy()
            
            _df = _df.to_frame()
            _df['target'] = y
            _df_nona = _df.dropna().copy()
            
            if "bins" not in self.bins[name]:
                self.trees[name] = DecisionTreeClassifier(
                    criterion = 'gini', 
                    random_state=666, 
                    max_depth=self.max_depth, 
                    min_samples_leaf = min_samples_leaf)

                #index 0 becouse _df is only one feature and target
                self.trees[name].fit(_df_nona.iloc[:,0].to_frame(), _df_nona['target'])

                self.bins[name]["bins"] = [-float("inf")]+ \
                        list(sorted(set(self.trees[name].tree_.threshold)))[1:]+[float("inf")]
            else:
                self.bins[name]["bins"] = sorted(list(set(self.bins[name]["bins"])))
                if self.bins[name]["bins"][0] != -float("inf"):
                    self.bins[name]["bins"] = [-float("inf")]+self.bins[name]["bins"]
                if self.bins[name]["bins"][-1] != float("inf"):
                    self.bins[name]["bins"] = self.bins[name]["bins"]+[float("inf")]
        
            #create lower bin bound
            self.bins[name]["bins_l"] = np.array(self.bins[name]["bins"][:-1]).reshape(1,-1)
            #create upper bin bound
            self.bins[name]["bins_u"] = np.array(self.bins[name]["bins"][1:]).reshape(1,-1)
            #creat bin names
            self.bins[name]["bin_names"] = ["NULL"]+ \
                ["["+str(round(i[0],5))+"<->"+str(round(i[1],5))+")" for i in zip(
                    list(self.bins[name]["bins_l"].reshape(-1)),
                    list(self.bins[name]["bins_u"].reshape(-1)))]
        return self
    
    def get_feature_names(self):
        if hasattr(self, "columnNames"):
            return self.columnNames
        else:
            return None  
    
    def transform(self, x):
        
        if type(x)==pd.core.frame.DataFrame:
            _transform_columnNames = x.columns
            _transform_numberofcolumns = x.shape[1]
            
        if type(x)==pd.core.series.Series:
            _transform_columnNames = [x.name]
            _transform_numberofcolumns = 1        
                
        DF = pd.DataFrame()
                
        for nr_col,name in enumerate(_transform_columnNames):
            #select data to discretize and convert to np array
            if _transform_numberofcolumns == 1:
                _data_to_disc = np.array(x).reshape(-1,1)
            else:
                _data_to_disc = np.array(x[name]).reshape(-1,1)
            
            #operacja maciezowa: 
            # 1. Czy wartosc jest wieksza niz donla granica, 2. Czy jest mniejsza niz donla grnaica
            # 3. Wskaz kolumna ktora spelnia punkt 1 i 2 
            _selected_bin = np.logical_and(_data_to_disc>=self.bins[name]["bins_l"],
                                          _data_to_disc<self.bins[name]["bins_u"])
            #
            _selected_bin_arg = np.argmax(_selected_bin,axis=1).reshape(-1,1)
            #fill nan with -99
            _selected_bin_arg[np.isnan(_data_to_disc)] = -99
            
            #create values to change
            if len(_selected_bin_arg[np.isnan(_data_to_disc)])==0:
                _old_values = [-99]+list(np.sort(np.unique(_selected_bin_arg)))
            else:
                _old_values = list(np.sort(np.unique(_selected_bin_arg)))
                
            _name_dic = {}
            
            for i in zip(_old_values,self.bins[name]["bin_names"]):
                _name_dic[i[0]] = i[1] 
                
            _s = pd.Series(pd.Categorical(_selected_bin_arg[:,0], ordered=True))
            #if len(_selected_bin_arg[np.isnan(_data_to_disc)])==0:
            #    _s.cat.add_categories(-99)
            DF[name] = _s.cat.rename_categories(_name_dic)
            
        return DF


```


```python
tr = DecisionTreeDiscretizer_DF(max_depth = 3, min_samples_prc_leaf=0.05)
#tr.fit_transform(df_train['loan_amnt'],df_train['default'] ).head()
discretized =  tr.fit_transform(df_train[features_numeric],df_train['default'] )

```


```python
discretized['default'] = df_train['default']
IV_nuemric = {}
for name in discretized.columns[:-1]:
    IV_value, _ = calculate_IV(name, df=discretized, ifprt = False)
    IV_nuemric[name] = IV_value

```

features_numeric


```python
import operator

sns.set_style("whitegrid")
fig = plt.figure(figsize=(16,40))
nrows= 7
ncol = 2
gs = matplotlib.gridspec.GridSpec(nrows=nrows, 
                       ncols=ncol, 
                       figure=fig, 
                       wspace=0.3,
                       hspace=0.3)

ax_dic = {}
features_iv_sort = [i[0] for i in sorted(IV_nuemric.items(), key=operator.itemgetter(1),reverse=True)]
counter = 0
for row in range(nrows):
    for col in range(ncol):
        if counter < len(features_iv_sort):
            col_name = features_iv_sort[counter]
            ax_dic[counter] = fig.add_subplot(gs[row, col])
            #sns.regplot(x=col_name, y="default",logistic=True,y_jitter=.03,
            #            data=df_train.sample(1000, random_state=666), x_bins=10, ax=ax_dic[counter])
            sns.barplot(y=col_name, x="default",data=discretized.sample(1000, random_state=666), palette= 'Reds')
            #Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,
            #Hot,Blackbody,Earth,Electric,Viridis,Cividis
            ax_dic[counter].set_title("%s [IV %.3f]"% (col_name,IV_nuemric[col_name]),fontsize=16)
            counter +=1
        
```


![png](04_One_dim_analysis_feature_prep_files/04_One_dim_analysis_feature_prep_53_0.png)



```python
discretized['loan_amnt'].unique()
```




    [[10012.5<->14987.5), [-inf<->7012.5), [9987.5<->10012.5), [14987.5<->15012.5), [15012.5<->19962.5), [19962.5<->inf), [8012.5<->9987.5), [7012.5<->8012.5)]
    Categories (8, object): [[-inf<->7012.5) < [7012.5<->8012.5) < [8012.5<->9987.5) < [9987.5<->10012.5) < [10012.5<->14987.5) < [14987.5<->15012.5) < [15012.5<->19962.5) < [19962.5<->inf)]




```python
features_numeric
```




    ['acc_open_past_24mths',
     'int_rate',
     'avg_cur_bal',
     'dti',
     'emp_length',
     'annual_inc',
     'mo_sin_old_rev_tl_op',
     'bc_util',
     'mort_acc',
     'tot_cur_bal',
     'bc_open_to_buy',
     'mths_since_recent_bc',
     'total_bc_limit',
     'loan_amnt']




```python
bins_prv = {}
man_bins = {}
for i in tr.bins.keys():
    bins_prv[i] = tr.bins[i]['bins']
for i in tr.bins.keys():
    man_bins[i] = {}

man_bins['int_rate']['bins']=[-np.inf,12,20,np.inf]
man_bins['acc_open_past_24mths']['bins']=[-np.inf,3,9,np.inf]
man_bins['dti']['bins'] = [10,33]
man_bins['bc_open_to_buy']['bins'] = [5000,2000]
man_bins['total_bc_limit']['bins'] = [30000,60000]
man_bins['avg_cur_bal']['bins'] = [10000,30000]
man_bins['tot_cur_bal']['bins'] = [20000,130000]
man_bins['annual_inc']['bins'] = [125000]

```


```python
man_bins
```




    {'acc_open_past_24mths': {'bins': [-inf, 3, 9, inf]},
     'int_rate': {'bins': [-inf, 12, 20, inf]},
     'avg_cur_bal': {'bins': [10000, 30000]},
     'dti': {'bins': [10, 33]},
     'emp_length': {},
     'annual_inc': {'bins': [125000]},
     'mo_sin_old_rev_tl_op': {},
     'bc_util': {},
     'mort_acc': {},
     'tot_cur_bal': {'bins': [20000, 130000]},
     'bc_open_to_buy': {'bins': [5000, 2000]},
     'mths_since_recent_bc': {},
     'total_bc_limit': {'bins': [30000, 60000]},
     'loan_amnt': {}}




```python
tr_n = DecisionTreeDiscretizer_DF(max_depth = 3, min_samples_prc_leaf=0.05, bins=man_bins)
#tr.fit_transform(df_train['loan_amnt'],df_train['default'] ).head()
discretized_n =  tr_n.fit_transform(df_train[features_numeric],df_train['default'] )
discretized_n['default'] = df_train['default']
IV_nuemric_n = {}
for name in discretized_n.columns[:-1]:
    IV_value, _ = calculate_IV(name, df=discretized_n, ifprt = False)
    IV_nuemric_n[name] = IV_value

```


```python
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16,40))
nrows= 7
ncol = 2
gs = matplotlib.gridspec.GridSpec(nrows=nrows, 
                       ncols=ncol, 
                       figure=fig, 
                       wspace=0.9,
                       hspace=0.9)
sns.set(font_scale = 1.5)

ax_dic = {}
features_iv_sort = [i[0] for i in sorted(IV_nuemric.items(), key=operator.itemgetter(1),reverse=True)]
counter = 0
for row in range(nrows):
    for col in range(ncol):
        if counter < len(features_iv_sort):
            col_name = features_iv_sort[counter]
            ax_dic[counter] = fig.add_subplot(gs[row, col])
            #sns.regplot(x=col_name, y="default",logistic=True,y_jitter=.03,
            #            data=df_train.sample(1000, random_state=666), x_bins=10, ax=ax_dic[counter])
            sns.barplot(y=col_name, x="default",data=discretized_n.sample(1000, random_state=666), palette= 'Reds')
            #Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,
            #Hot,Blackbody,Earth,Electric,Viridis,Cividis
            ax_dic[counter].set_title("%s [IV %.3f] \n"% (col_name,IV_nuemric_n[col_name]),fontsize=24)
            ax_dic[counter].set_ylabel('')  
            ax_dic[counter].set_xlabel('default rate [%]') 
            counter +=1
        
```


![png](04_One_dim_analysis_feature_prep_files/04_One_dim_analysis_feature_prep_59_0.png)



```python
IV_cat
```




    {'zip_code': 0.027941173296134353,
     'sub_grade': 0.45488374504047996,
     'addr_state': 0.014000589581223443,
     'home_ownership': 0.028334523770828172,
     'term': 0.11187877821349718}




```python
IV_nuemric
```




    {'acc_open_past_24mths': 0.08746038723921441,
     'int_rate': 0.4193670952941311,
     'avg_cur_bal': 0.04748587628704645,
     'dti': 0.06137983285388187,
     'emp_length': 0.008685111335620756,
     'annual_inc': 0.03574374513150312,
     'mo_sin_old_rev_tl_op': 0.026562616545743396,
     'bc_util': 0.021225004030360692,
     'mort_acc': 0.030779604411806168,
     'tot_cur_bal': 0.03779414254988935,
     'bc_open_to_buy': 0.05694550828005637,
     'mths_since_recent_bc': 0.03217104578411213,
     'total_bc_limit': 0.04983740294599484,
     'loan_amnt': 0.017006957584100063}




```python
IV_nuemric_n
```




    {'acc_open_past_24mths': 0.07060656452169317,
     'int_rate': 0.3121249421823555,
     'avg_cur_bal': 0.04248061141413449,
     'dti': 0.03465820035160508,
     'emp_length': 0.008685111335620756,
     'annual_inc': 0.013152702492669806,
     'mo_sin_old_rev_tl_op': 0.026562616545743396,
     'bc_util': 0.021225004030360692,
     'mort_acc': 0.030779604411806168,
     'tot_cur_bal': 0.029082408639124695,
     'bc_open_to_buy': 0.036920335976666205,
     'mths_since_recent_bc': 0.03217104578411213,
     'total_bc_limit': 0.036804743242539295,
     'loan_amnt': 0.017006957584100063}



# Data features


```python
features_data
```




    ['issue_d']




```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress
import pandas as pd


class DataFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, x, y = None):
        if type(x)==pd.core.frame.DataFrame:
            self.columnNames = x.columns
            self.numberofcolumns = x.shape[1]
            
        if type(x)==pd.core.series.Series:
            self.columnNames = [x.name]
            self.numberofcolumns = 1
        return self
    
    def get_feature_names(self):
        if hasattr(self, "columnNames"):
            return self.columnNames
        else:
            return None  
    
    def transform(self, x):
        
        DF = pd.DataFrame()
        
        if type(x)==pd.core.frame.DataFrame:
            _transform_columnNames = x.columns
            
            for nr_col,name in enumerate(_transform_columnNames):
                #select data to discretize and convert to np array
                DF[name+"_month"] = x['name'].dt.month.tolist()
                DF[name+"_quarter"] = x['name'].dt.quarter.tolist()
            
        if type(x)==pd.core.series.Series:
            name = x.name
            DF[name+"_month"] = x.dt.month.tolist()
            DF[name+"_quarter"] = x.dt.quarter.tolist()

        return DF


```


```python
dat =  DataFeatures()
dat = dat.fit_transform(df_train.issue_d)

```


```python
tr_d = DecisionTreeDiscretizer_DF(max_depth = 3, min_samples_prc_leaf=0.05)
#tr.fit_transform(df_train['loan_amnt'],df_train['default'] ).head()
discretized_d =  tr_d.fit_transform(dat,df_train['default'] )
discretized_d['default'] = df_train['default']
IV_nuemric_d = {}
for name in discretized_d.columns[:-1]:
    IV_value, _ = calculate_IV(name, df=discretized_d, ifprt = False)
    IV_nuemric_d[name] = IV_value
```


```python
IV_nuemric_d
```




    {'issue_d_month': 0.0002987427292848985,
     'issue_d_quarter': 0.00021091782514884996}



# Text features


```python
import lightgbm as lgb

```


```python
import lightgbm as lgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer,RobustScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge

seed = 6666
max_features_Vectorizer = 10000

pipe = make_pipeline(
    ColumnTransformer([
        ('emp_title', TfidfVectorizer(lowercase=True, 
                               ngram_range=(1, 2), 
                               max_features=max_features_Vectorizer, 
                               dtype=np.float32,
                               use_idf=True),'emp_title'),
        ('desc', TfidfVectorizer(lowercase=True, 
                               ngram_range=(1, 2), 
                               max_features=max_features_Vectorizer, 
                               dtype=np.float32,
                               use_idf=True),'desc'),
        ('title', TfidfVectorizer(lowercase=True, 
                               ngram_range=(1, 2), 
                               max_features=max_features_Vectorizer, 
                               dtype=np.float32,
                               use_idf=True),'title'),
    ]),
    lgb.LGBMClassifier()
    )
```


```python
df_train.replace({'desc': {'': np.nan},'emp_title': {'': np.nan},'title': {'': np.nan}}, inplace=True)
df_train.fillna({'desc': 'novalue', 'emp_title': 'novalue', 'title': 'novalue'}, inplace=True)

df_test.replace({'desc': {'': np.nan},'emp_title': {'': np.nan},'title': {'': np.nan}}, inplace=True)
df_test.fillna({'desc': 'novalue', 'emp_title': 'novalue', 'title': 'novalue'}, inplace=True)

```


```python
df_train.columns
```




    Index(['zip_code', 'sub_grade', 'acc_open_past_24mths', 'int_rate',
           'avg_cur_bal', 'dti', 'emp_length', 'addr_state', 'annual_inc',
           'mo_sin_old_rev_tl_op', 'home_ownership', 'bc_util', 'mort_acc',
           'tot_cur_bal', 'bc_open_to_buy', 'term', 'mths_since_recent_bc',
           'total_bc_limit', 'loan_amnt', 'emp_title', 'issue_d', 'default',
           'desc', 'title'],
          dtype='object')




```python
%%time
save = pipe.fit(df_train, df_train.default)
```

    CPU times: user 3min 30s, sys: 5.6 s, total: 3min 35s
    Wall time: 1min 23s



```python
from sklearn import metrics

y_pred_train = pipe.predict_proba(df_train)[:,1]
fpr_train, tpr_train, thresholds_train = metrics.roc_curve(df_train['default'], y_pred_train, pos_label=1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)
print("Train AUC= %.3f"% roc_auc_train)
    
y_pred_test = pipe.predict_proba(df_test)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(df_test['default'], y_pred_test, pos_label=1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)
print("Test AUC= %.3f"% roc_auc_test)  
```

    Train AUC= 0.605
    Test AUC= 0.578



```python
import eli5
eli5.show_weights(pipe.named_steps['lgbmclassifier'], 
                  feature_names = pipe.named_steps['columntransformer'].get_feature_names(),
                 top=30)
```





    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>



    

    

    

    

    

    


    

    

    

    

    

    


    

    

    

    

    
        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>
    
        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0589
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__credit
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 81.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0537
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__novalue
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 87.51%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0300
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__driver
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 87.91%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0287
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__debt
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 89.87%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0223
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__interest
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 90.02%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0218
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__card
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 90.57%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0201
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__director
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 91.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0187
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__business
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 91.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0164
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__engineer
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 92.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0139
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__senior
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 93.91%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0108
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__rate
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 93.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0107
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__operator
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 93.95%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0107
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__software
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 94.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0104
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__analyst
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 94.15%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0102
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__attorney
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 94.69%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0088
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__bills
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0080
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__physician
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.26%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0075
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__manager
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.61%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0068
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__vp
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0064
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__vice president
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.78%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0064
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__other
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.98%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0059
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__business
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0059
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__professor
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0052
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__novalue
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.43%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0050
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                title__pool
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.49%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__tech
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.53%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0048
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__architect
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.56%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0048
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__added
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0043
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                desc__need
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0042
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                emp_title__associate
            </td>
        </tr>
    
    
        
            <tr style="background-color: hsl(120, 100.00%, 96.84%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 29970 more &hellip;</i>
                </td>
            </tr>
        
    
    </tbody>
</table>
    

    


    

    

    

    

    

    






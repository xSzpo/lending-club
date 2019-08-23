

```python
import pandas as pd
import codecs
from pandarallel import pandarallel
import datetime
import numpy as np
import feather
from pathlib import Path
import os

pd.options.display.max_columns = 999
pandarallel.initialize(progress_bar=False)

```

    New pandarallel memory created - Size: 2000 MB
    Pandarallel will run on 4 workers



```python
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
```


```python
with codecs.open('../01_data/variables_browseNotes_select.txt','r') as f:
    variables = [i[:-1] for i in f.readlines()]
```


```python
%%time

df = pd.read_csv('../01_data/loan.csv', 
                 low_memory=False,
                 usecols=variables,
                 parse_dates=['issue_d','earliest_cr_line'],
                 date_parser=lambda col: pd.to_datetime(col, format="%b-%Y", errors='coerce').date()
                )
```

    CPU times: user 15min 34s, sys: 10.4 s, total: 15min 45s
    Wall time: 16min 12s



```python
df.dtypes
```




    id                                            float64
    member_id                                     float64
    loan_amnt                                       int64
    funded_amnt                                     int64
    term                                           object
    int_rate                                      float64
    installment                                   float64
    grade                                          object
    sub_grade                                      object
    emp_title                                      object
    emp_length                                     object
    home_ownership                                 object
    annual_inc                                    float64
    issue_d                                datetime64[ns]
    loan_status                                    object
    url                                           float64
    desc                                           object
    purpose                                        object
    title                                          object
    zip_code                                       object
    addr_state                                     object
    dti                                           float64
    delinq_2yrs                                   float64
    earliest_cr_line                       datetime64[ns]
    inq_last_6mths                                float64
    mths_since_last_delinq                        float64
    mths_since_last_record                        float64
    open_acc                                      float64
    pub_rec                                       float64
    revol_bal                                       int64
    revol_util                                    float64
    total_acc                                     float64
    initial_list_status                            object
    collections_12_mths_ex_med                    float64
    mths_since_last_major_derog                   float64
    application_type                               object
    annual_inc_joint                              float64
    dti_joint                                     float64
    verification_status_joint                      object
    acc_now_delinq                                float64
    tot_coll_amt                                  float64
    tot_cur_bal                                   float64
    open_acc_6m                                   float64
    open_act_il                                   float64
    open_il_12m                                   float64
    open_il_24m                                   float64
    mths_since_rcnt_il                            float64
    total_bal_il                                  float64
    il_util                                       float64
    open_rv_12m                                   float64
    open_rv_24m                                   float64
    max_bal_bc                                    float64
    all_util                                      float64
    total_rev_hi_lim                              float64
    inq_fi                                        float64
    total_cu_tl                                   float64
    inq_last_12m                                  float64
    acc_open_past_24mths                          float64
    avg_cur_bal                                   float64
    bc_open_to_buy                                float64
    bc_util                                       float64
    chargeoff_within_12_mths                      float64
    delinq_amnt                                   float64
    mo_sin_old_rev_tl_op                          float64
    mo_sin_rcnt_rev_tl_op                         float64
    mo_sin_rcnt_tl                                float64
    mort_acc                                      float64
    mths_since_recent_bc                          float64
    mths_since_recent_inq                         float64
    mths_since_recent_revol_delinq                float64
    num_accts_ever_120_pd                         float64
    num_actv_bc_tl                                float64
    num_actv_rev_tl                               float64
    num_bc_sats                                   float64
    num_bc_tl                                     float64
    num_il_tl                                     float64
    num_op_rev_tl                                 float64
    num_rev_accts                                 float64
    num_rev_tl_bal_gt_0                           float64
    num_sats                                      float64
    num_tl_120dpd_2m                              float64
    num_tl_30dpd                                  float64
    num_tl_90g_dpd_24m                            float64
    num_tl_op_past_12m                            float64
    pct_tl_nvr_dlq                                float64
    percent_bc_gt_75                              float64
    pub_rec_bankruptcies                          float64
    tax_liens                                     float64
    tot_hi_cred_lim                               float64
    total_bal_ex_mort                             float64
    total_bc_limit                                float64
    total_il_high_credit_limit                    float64
    revol_bal_joint                               float64
    sec_app_earliest_cr_line                       object
    sec_app_inq_last_6mths                        float64
    sec_app_mort_acc                              float64
    sec_app_open_acc                              float64
    sec_app_revol_util                            float64
    sec_app_open_act_il                           float64
    sec_app_num_rev_accts                         float64
    sec_app_chargeoff_within_12_mths              float64
    sec_app_collections_12_mths_ex_med            float64
    sec_app_mths_since_last_major_derog           float64
    disbursement_method                            object
    dtype: object




```python
def include_exclude(col):
    if col <= pd.to_datetime('2016.03'):
        return True
    else:
        return False
def assign_random():
    return np.random.rand()
```


```python
df = df.assign(include = lambda x: x['issue_d'].parallel_apply(include_exclude))
```


```python
from collections import Counter
Counter(df.include.tolist())
```




    Counter({False: 1239341, True: 1021327})




```python
np.random.seed(seed=666)
df['random_value'] = np.random.rand(df.shape[0])
```


```python
default_map = {
    'Fully Paid': 0,
    'Default' : 1,
    'In Grace Period': 0,
    'Does not meet the credit policy. Status:Fully Paid' : 0,
    'Charged Off': 1,
    'Late (31-120 days)' : 1,
    'Current' : 0,
    'Does not meet the credit policy. Status:Charged Off' : 1,
    'Late (16-30 days)' : 0
}

def default_flag(value):
    return default_map[value]
```


```python
df = df.assign(default = lambda x: x['loan_status'].parallel_apply(default_flag))
```


```python
df.head()
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
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>url</th>
      <th>desc</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>collections_12_mths_ex_med</th>
      <th>mths_since_last_major_derog</th>
      <th>application_type</th>
      <th>annual_inc_joint</th>
      <th>dti_joint</th>
      <th>verification_status_joint</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>open_acc_6m</th>
      <th>open_act_il</th>
      <th>open_il_12m</th>
      <th>open_il_24m</th>
      <th>mths_since_rcnt_il</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
      <th>acc_open_past_24mths</th>
      <th>avg_cur_bal</th>
      <th>bc_open_to_buy</th>
      <th>bc_util</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>mo_sin_old_rev_tl_op</th>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <th>mo_sin_rcnt_tl</th>
      <th>mort_acc</th>
      <th>mths_since_recent_bc</th>
      <th>mths_since_recent_inq</th>
      <th>mths_since_recent_revol_delinq</th>
      <th>num_accts_ever_120_pd</th>
      <th>num_actv_bc_tl</th>
      <th>num_actv_rev_tl</th>
      <th>num_bc_sats</th>
      <th>num_bc_tl</th>
      <th>num_il_tl</th>
      <th>num_op_rev_tl</th>
      <th>num_rev_accts</th>
      <th>num_rev_tl_bal_gt_0</th>
      <th>num_sats</th>
      <th>num_tl_120dpd_2m</th>
      <th>num_tl_30dpd</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>revol_bal_joint</th>
      <th>sec_app_earliest_cr_line</th>
      <th>sec_app_inq_last_6mths</th>
      <th>sec_app_mort_acc</th>
      <th>sec_app_open_acc</th>
      <th>sec_app_revol_util</th>
      <th>sec_app_open_act_il</th>
      <th>sec_app_num_rev_accts</th>
      <th>sec_app_chargeoff_within_12_mths</th>
      <th>sec_app_collections_12_mths_ex_med</th>
      <th>sec_app_mths_since_last_major_derog</th>
      <th>disbursement_method</th>
      <th>include</th>
      <th>random_value</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2500</td>
      <td>2500</td>
      <td>36 months</td>
      <td>13.56</td>
      <td>84.92</td>
      <td>C</td>
      <td>C1</td>
      <td>Chef</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>55000.0</td>
      <td>2018-12-01</td>
      <td>Current</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>109xx</td>
      <td>NY</td>
      <td>18.24</td>
      <td>0.0</td>
      <td>2001-04-01</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>4341</td>
      <td>10.3</td>
      <td>34.0</td>
      <td>w</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16901.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>12560.0</td>
      <td>69.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2137.0</td>
      <td>28.0</td>
      <td>42000.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>1878.0</td>
      <td>34360.0</td>
      <td>5.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>212.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>7.0</td>
      <td>18.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>60124.0</td>
      <td>16901.0</td>
      <td>36500.0</td>
      <td>18124.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>False</td>
      <td>0.700437</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000</td>
      <td>30000</td>
      <td>60 months</td>
      <td>18.94</td>
      <td>777.23</td>
      <td>D</td>
      <td>D2</td>
      <td>Postmaster</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>90000.0</td>
      <td>2018-12-01</td>
      <td>Current</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>713xx</td>
      <td>LA</td>
      <td>26.52</td>
      <td>0.0</td>
      <td>1987-06-01</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>75.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>12315</td>
      <td>24.2</td>
      <td>44.0</td>
      <td>w</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1208.0</td>
      <td>321915.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>87153.0</td>
      <td>88.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>998.0</td>
      <td>57.0</td>
      <td>50800.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>24763.0</td>
      <td>13761.0</td>
      <td>8.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>372872.0</td>
      <td>99468.0</td>
      <td>15000.0</td>
      <td>94072.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>False</td>
      <td>0.844187</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5000</td>
      <td>5000</td>
      <td>36 months</td>
      <td>17.97</td>
      <td>180.69</td>
      <td>D</td>
      <td>D1</td>
      <td>Administrative</td>
      <td>6 years</td>
      <td>MORTGAGE</td>
      <td>59280.0</td>
      <td>2018-12-01</td>
      <td>Current</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>490xx</td>
      <td>MI</td>
      <td>10.51</td>
      <td>0.0</td>
      <td>2011-04-01</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>4599</td>
      <td>19.1</td>
      <td>13.0</td>
      <td>w</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>110299.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>7150.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>24100.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>18383.0</td>
      <td>13800.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>92.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>77.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>136927.0</td>
      <td>11749.0</td>
      <td>13800.0</td>
      <td>10000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>False</td>
      <td>0.676514</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4000</td>
      <td>4000</td>
      <td>36 months</td>
      <td>18.94</td>
      <td>146.51</td>
      <td>D</td>
      <td>D2</td>
      <td>IT Supervisor</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>92000.0</td>
      <td>2018-12-01</td>
      <td>Current</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>985xx</td>
      <td>WA</td>
      <td>16.74</td>
      <td>0.0</td>
      <td>2006-02-01</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5468</td>
      <td>78.1</td>
      <td>13.0</td>
      <td>w</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>686.0</td>
      <td>305049.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30683.0</td>
      <td>68.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3761.0</td>
      <td>70.0</td>
      <td>7000.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30505.0</td>
      <td>1239.0</td>
      <td>75.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>154.0</td>
      <td>64.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>64.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>385183.0</td>
      <td>36151.0</td>
      <td>5000.0</td>
      <td>44984.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>False</td>
      <td>0.727858</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000</td>
      <td>30000</td>
      <td>60 months</td>
      <td>16.14</td>
      <td>731.78</td>
      <td>C</td>
      <td>C4</td>
      <td>Mechanic</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>57250.0</td>
      <td>2018-12-01</td>
      <td>Current</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>212xx</td>
      <td>MD</td>
      <td>26.35</td>
      <td>0.0</td>
      <td>2000-12-01</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>829</td>
      <td>3.6</td>
      <td>26.0</td>
      <td>w</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>116007.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>28845.0</td>
      <td>89.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>516.0</td>
      <td>54.0</td>
      <td>23100.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>9667.0</td>
      <td>8471.0</td>
      <td>8.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>216.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>92.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>157548.0</td>
      <td>29674.0</td>
      <td>9300.0</td>
      <td>32332.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>False</td>
      <td>0.951458</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Save excluded data


```python
feather.write_dataframe(df.query('include == False').drop('include', axis=1), '../01_data/DS_loans_OUT.feather')

```

Included data


```python
df_IN = df.query('include == True').drop('include', axis=1)
```

Drop irrelevant features


```python
df_IN.select_dtypes(include=['object']).describe().T
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>term</th>
      <td>1021327</td>
      <td>2</td>
      <td>36 months</td>
      <td>717289</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>1021327</td>
      <td>7</td>
      <td>B</td>
      <td>294820</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>1021327</td>
      <td>35</td>
      <td>B3</td>
      <td>64847</td>
    </tr>
    <tr>
      <th>emp_title</th>
      <td>960873</td>
      <td>325103</td>
      <td>Teacher</td>
      <td>16015</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>967548</td>
      <td>11</td>
      <td>10+ years</td>
      <td>337941</td>
    </tr>
    <tr>
      <th>home_ownership</th>
      <td>1021327</td>
      <td>6</td>
      <td>MORTGAGE</td>
      <td>510420</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>1021327</td>
      <td>9</td>
      <td>Fully Paid</td>
      <td>749471</td>
    </tr>
    <tr>
      <th>desc</th>
      <td>126054</td>
      <td>124491</td>
      <td></td>
      <td>250</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>1021327</td>
      <td>14</td>
      <td>debt_consolidation</td>
      <td>600940</td>
    </tr>
    <tr>
      <th>title</th>
      <td>1010554</td>
      <td>63155</td>
      <td>Debt consolidation</td>
      <td>484444</td>
    </tr>
    <tr>
      <th>zip_code</th>
      <td>1021327</td>
      <td>942</td>
      <td>945xx</td>
      <td>11163</td>
    </tr>
    <tr>
      <th>addr_state</th>
      <td>1021327</td>
      <td>51</td>
      <td>CA</td>
      <td>147858</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>1021327</td>
      <td>2</td>
      <td>w</td>
      <td>543473</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>1021327</td>
      <td>2</td>
      <td>Individual</td>
      <td>1018155</td>
    </tr>
    <tr>
      <th>verification_status_joint</th>
      <td>3172</td>
      <td>1</td>
      <td>Not Verified</td>
      <td>3172</td>
    </tr>
    <tr>
      <th>sec_app_earliest_cr_line</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>disbursement_method</th>
      <td>1021327</td>
      <td>2</td>
      <td>Cash</td>
      <td>1019594</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_IN.select_dtypes(exclude=['object']).describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>member_id</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>1021327.0</td>
      <td>14864.527424</td>
      <td>8496.667933</td>
      <td>500.000000</td>
      <td>8000.000000</td>
      <td>13000.000000</td>
      <td>20000.000000</td>
      <td>40000.00</td>
    </tr>
    <tr>
      <th>funded_amnt</th>
      <td>1021327.0</td>
      <td>14852.896159</td>
      <td>8492.024793</td>
      <td>500.000000</td>
      <td>8000.000000</td>
      <td>13000.000000</td>
      <td>20000.000000</td>
      <td>40000.00</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>1021327.0</td>
      <td>13.144345</td>
      <td>4.451256</td>
      <td>5.320000</td>
      <td>9.750000</td>
      <td>12.990000</td>
      <td>15.990000</td>
      <td>28.99</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>1021327.0</td>
      <td>439.859617</td>
      <td>246.668097</td>
      <td>4.930000</td>
      <td>261.530000</td>
      <td>384.450000</td>
      <td>577.655000</td>
      <td>1536.95</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>1021323.0</td>
      <td>75739.112984</td>
      <td>66630.445505</td>
      <td>0.000000</td>
      <td>46000.000000</td>
      <td>65000.000000</td>
      <td>90000.000000</td>
      <td>9550000.00</td>
    </tr>
    <tr>
      <th>url</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>1021308.0</td>
      <td>18.319798</td>
      <td>8.645885</td>
      <td>-1.000000</td>
      <td>12.000000</td>
      <td>17.790000</td>
      <td>24.170000</td>
      <td>999.00</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>1021298.0</td>
      <td>0.318246</td>
      <td>0.868505</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>39.00</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>1021297.0</td>
      <td>0.679161</td>
      <td>0.984443</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>502420.0</td>
      <td>34.028608</td>
      <td>21.903673</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>31.000000</td>
      <td>50.000000</td>
      <td>192.00</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>160735.0</td>
      <td>69.565365</td>
      <td>27.629707</td>
      <td>0.000000</td>
      <td>51.000000</td>
      <td>70.000000</td>
      <td>90.000000</td>
      <td>129.00</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>1021298.0</td>
      <td>11.623319</td>
      <td>5.381174</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>11.000000</td>
      <td>14.000000</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>1021298.0</td>
      <td>0.200998</td>
      <td>0.593733</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>86.00</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>1021327.0</td>
      <td>17142.892489</td>
      <td>22970.835300</td>
      <td>0.000000</td>
      <td>6461.000000</td>
      <td>11926.000000</td>
      <td>20977.000000</td>
      <td>2904836.00</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>1020754.0</td>
      <td>54.656617</td>
      <td>23.907140</td>
      <td>0.000000</td>
      <td>37.100000</td>
      <td>55.400000</td>
      <td>73.200000</td>
      <td>892.30</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>1021298.0</td>
      <td>25.277872</td>
      <td>11.874104</td>
      <td>1.000000</td>
      <td>17.000000</td>
      <td>24.000000</td>
      <td>32.000000</td>
      <td>176.00</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>1021182.0</td>
      <td>0.015280</td>
      <td>0.138814</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>259797.0</td>
      <td>43.255796</td>
      <td>21.256968</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>43.000000</td>
      <td>61.000000</td>
      <td>197.00</td>
    </tr>
    <tr>
      <th>annual_inc_joint</th>
      <td>3172.0</td>
      <td>108837.317267</td>
      <td>46290.852667</td>
      <td>17950.000000</td>
      <td>77000.000000</td>
      <td>102000.000000</td>
      <td>132000.000000</td>
      <td>500000.00</td>
    </tr>
    <tr>
      <th>dti_joint</th>
      <td>3168.0</td>
      <td>18.556029</td>
      <td>7.209494</td>
      <td>1.110000</td>
      <td>13.437500</td>
      <td>18.225000</td>
      <td>23.280000</td>
      <td>63.66</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>1021298.0</td>
      <td>0.005250</td>
      <td>0.079012</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>951051.0</td>
      <td>230.413535</td>
      <td>9588.656577</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9152545.00</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>951051.0</td>
      <td>140552.256184</td>
      <td>155130.076844</td>
      <td>0.000000</td>
      <td>30159.000000</td>
      <td>81335.000000</td>
      <td>209503.000000</td>
      <td>8000078.00</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>155197.0</td>
      <td>1.054138</td>
      <td>1.211721</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>16.00</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>155198.0</td>
      <td>2.815281</td>
      <td>2.993564</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>48.00</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>155198.0</td>
      <td>0.749153</td>
      <td>0.990731</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>155198.0</td>
      <td>1.632012</td>
      <td>1.673988</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>150996.0</td>
      <td>21.540776</td>
      <td>27.215913</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>24.000000</td>
      <td>446.00</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>155198.0</td>
      <td>35821.015548</td>
      <td>42298.831529</td>
      <td>0.000000</td>
      <td>9711.250000</td>
      <td>24215.500000</td>
      <td>46754.000000</td>
      <td>878459.00</td>
    </tr>
    <tr>
      <th>il_util</th>
      <td>134625.0</td>
      <td>70.857129</td>
      <td>23.130439</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>74.000000</td>
      <td>87.000000</td>
      <td>558.00</td>
    </tr>
    <tr>
      <th>open_rv_12m</th>
      <td>155198.0</td>
      <td>1.375875</td>
      <td>1.523030</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>open_rv_24m</th>
      <td>155198.0</td>
      <td>2.936269</td>
      <td>2.630192</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>44.00</td>
    </tr>
    <tr>
      <th>max_bal_bc</th>
      <td>155198.0</td>
      <td>6142.875449</td>
      <td>6095.782248</td>
      <td>0.000000</td>
      <td>2507.000000</td>
      <td>4642.000000</td>
      <td>7979.750000</td>
      <td>776843.00</td>
    </tr>
    <tr>
      <th>all_util</th>
      <td>155192.0</td>
      <td>60.685828</td>
      <td>19.954672</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>62.000000</td>
      <td>75.000000</td>
      <td>198.00</td>
    </tr>
    <tr>
      <th>total_rev_hi_lim</th>
      <td>951051.0</td>
      <td>32672.489184</td>
      <td>37504.913242</td>
      <td>0.000000</td>
      <td>14100.000000</td>
      <td>24100.000000</td>
      <td>40500.000000</td>
      <td>9999999.00</td>
    </tr>
    <tr>
      <th>inq_fi</th>
      <td>155198.0</td>
      <td>0.951688</td>
      <td>1.492206</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>total_cu_tl</th>
      <td>155197.0</td>
      <td>1.543148</td>
      <td>2.779478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>79.00</td>
    </tr>
    <tr>
      <th>inq_last_12m</th>
      <td>155197.0</td>
      <td>2.193670</td>
      <td>2.469732</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>acc_open_past_24mths</th>
      <td>971297.0</td>
      <td>4.483550</td>
      <td>3.055239</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>64.00</td>
    </tr>
    <tr>
      <th>avg_cur_bal</th>
      <td>951039.0</td>
      <td>13354.872523</td>
      <td>15922.473224</td>
      <td>0.000000</td>
      <td>3160.000000</td>
      <td>7462.000000</td>
      <td>18535.000000</td>
      <td>958084.00</td>
    </tr>
    <tr>
      <th>bc_open_to_buy</th>
      <td>962248.0</td>
      <td>9264.317049</td>
      <td>14331.035948</td>
      <td>0.000000</td>
      <td>1258.000000</td>
      <td>4122.000000</td>
      <td>11071.000000</td>
      <td>559912.00</td>
    </tr>
    <tr>
      <th>bc_util</th>
      <td>961667.0</td>
      <td>63.333926</td>
      <td>27.186503</td>
      <td>0.000000</td>
      <td>43.400000</td>
      <td>67.500000</td>
      <td>87.000000</td>
      <td>339.60</td>
    </tr>
    <tr>
      <th>chargeoff_within_12_mths</th>
      <td>1021182.0</td>
      <td>0.009067</td>
      <td>0.108588</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>delinq_amnt</th>
      <td>1021298.0</td>
      <td>12.374551</td>
      <td>701.650715</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>159177.00</td>
    </tr>
    <tr>
      <th>mo_sin_old_rev_tl_op</th>
      <td>951050.0</td>
      <td>185.307965</td>
      <td>94.054011</td>
      <td>3.000000</td>
      <td>120.000000</td>
      <td>168.000000</td>
      <td>234.000000</td>
      <td>852.00</td>
    </tr>
    <tr>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <td>951050.0</td>
      <td>13.426090</td>
      <td>16.693808</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>372.00</td>
    </tr>
    <tr>
      <th>mo_sin_rcnt_tl</th>
      <td>951051.0</td>
      <td>8.135296</td>
      <td>9.138431</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>283.00</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>971297.0</td>
      <td>1.746994</td>
      <td>2.066175</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>52.00</td>
    </tr>
    <tr>
      <th>mths_since_recent_bc</th>
      <td>962876.0</td>
      <td>24.766935</td>
      <td>31.229792</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>14.000000</td>
      <td>30.000000</td>
      <td>616.00</td>
    </tr>
    <tr>
      <th>mths_since_recent_inq</th>
      <td>870835.0</td>
      <td>6.845424</td>
      <td>5.909132</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>25.00</td>
    </tr>
    <tr>
      <th>mths_since_recent_revol_delinq</th>
      <td>338733.0</td>
      <td>35.761674</td>
      <td>22.420422</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>33.000000</td>
      <td>53.000000</td>
      <td>197.00</td>
    </tr>
    <tr>
      <th>num_accts_ever_120_pd</th>
      <td>951051.0</td>
      <td>0.486725</td>
      <td>1.261724</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>39.00</td>
    </tr>
    <tr>
      <th>num_actv_bc_tl</th>
      <td>951051.0</td>
      <td>3.739758</td>
      <td>2.245597</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>36.00</td>
    </tr>
    <tr>
      <th>num_actv_rev_tl</th>
      <td>951051.0</td>
      <td>5.823705</td>
      <td>3.304920</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>52.00</td>
    </tr>
    <tr>
      <th>num_bc_sats</th>
      <td>962737.0</td>
      <td>4.760361</td>
      <td>2.897069</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>63.00</td>
    </tr>
    <tr>
      <th>num_bc_tl</th>
      <td>951051.0</td>
      <td>8.368157</td>
      <td>4.827501</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>70.00</td>
    </tr>
    <tr>
      <th>num_il_tl</th>
      <td>951051.0</td>
      <td>8.482174</td>
      <td>7.297438</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>150.00</td>
    </tr>
    <tr>
      <th>num_op_rev_tl</th>
      <td>951051.0</td>
      <td>8.354666</td>
      <td>4.480389</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>83.00</td>
    </tr>
    <tr>
      <th>num_rev_accts</th>
      <td>951050.0</td>
      <td>14.972217</td>
      <td>8.084461</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>118.00</td>
    </tr>
    <tr>
      <th>num_rev_tl_bal_gt_0</th>
      <td>951051.0</td>
      <td>5.786709</td>
      <td>3.238482</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>45.00</td>
    </tr>
    <tr>
      <th>num_sats</th>
      <td>962737.0</td>
      <td>11.705742</td>
      <td>5.392205</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>11.000000</td>
      <td>14.000000</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>num_tl_120dpd_2m</th>
      <td>916084.0</td>
      <td>0.000790</td>
      <td>0.029798</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>num_tl_30dpd</th>
      <td>951051.0</td>
      <td>0.003722</td>
      <td>0.065247</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>num_tl_90g_dpd_24m</th>
      <td>951051.0</td>
      <td>0.088703</td>
      <td>0.489707</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>39.00</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>951051.0</td>
      <td>2.076652</td>
      <td>1.759504</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>950898.0</td>
      <td>94.270518</td>
      <td>8.538346</td>
      <td>0.000000</td>
      <td>91.700000</td>
      <td>97.900000</td>
      <td>100.000000</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>961839.0</td>
      <td>49.026498</td>
      <td>35.546459</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>50.000000</td>
      <td>80.000000</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>1019962.0</td>
      <td>0.120741</td>
      <td>0.362538</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>1021222.0</td>
      <td>0.051403</td>
      <td>0.399393</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>85.00</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>951051.0</td>
      <td>172774.648381</td>
      <td>175109.839979</td>
      <td>0.000000</td>
      <td>49523.000000</td>
      <td>112447.000000</td>
      <td>249894.500000</td>
      <td>9999999.00</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>971297.0</td>
      <td>49817.290124</td>
      <td>47074.370969</td>
      <td>0.000000</td>
      <td>21444.000000</td>
      <td>37712.000000</td>
      <td>62599.000000</td>
      <td>2921551.00</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>971297.0</td>
      <td>21414.096784</td>
      <td>21184.943721</td>
      <td>0.000000</td>
      <td>7700.000000</td>
      <td>15000.000000</td>
      <td>27900.000000</td>
      <td>1105500.00</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>951051.0</td>
      <td>41357.970908</td>
      <td>42600.068775</td>
      <td>0.000000</td>
      <td>14329.000000</td>
      <td>31078.000000</td>
      <td>55644.500000</td>
      <td>2101913.00</td>
    </tr>
    <tr>
      <th>revol_bal_joint</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_inq_last_6mths</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_mort_acc</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_open_acc</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_revol_util</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_open_act_il</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_num_rev_accts</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_chargeoff_within_12_mths</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_collections_12_mths_ex_med</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_mths_since_last_major_derog</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>random_value</th>
      <td>1021327.0</td>
      <td>0.500098</td>
      <td>0.288678</td>
      <td>0.000001</td>
      <td>0.250038</td>
      <td>0.500058</td>
      <td>0.749982</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>default</th>
      <td>1021327.0</td>
      <td>0.174301</td>
      <td>0.379368</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
drop_columns_all_nan = ['id','member_id','url','revol_bal_joint','sec_app_inq_last_6mths','sec_app_mort_acc',
                        'sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts',
                        'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med',
                        'sec_app_mths_since_last_major_derog']
```


```python
df_IN =df_IN.drop(drop_columns_all_nan, axis=1)
```


```python
df.shape
```




    (2260668, 107)




```python
df_IN.shape
```




    (1021327, 93)




```python
feather.write_dataframe(df_IN.query('random_value < 0.7'), '../01_data/DS_loans_IN_train.feather')
feather.write_dataframe(df_IN.query('random_value >= 0.7'), '../01_data/DS_loans_IN_test.feather')

```

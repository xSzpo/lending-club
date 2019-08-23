
## Set Environment and load data


```python
import os
import sys
import pandas as pd
import codecs
import re
import unicodedata

from pyspark.sql import SparkSession
import pyspark.sql.functions as f

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# setup env inside docker

# add path with python packages installed outside of the docker
sys.path.append('/home/jovyan/work/98_sys_files/site-packages')
#!pip install pyallegro -t /home/jovyan/work/98_sys_files/site-packages 

# add jar driver to connect to sqlite
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/jovyan/work/98_sys_files/jar/sqlite-jdbc-3.28.0.jar pyspark-shell'
```


```python
#run spark sesion 
spark = SparkSession.builder \
    .appName('lending-club-loan DataFrame') \
    .master('local[*]') \
    .getOrCreate()

# load schema
with codecs.open("/home/jovyan/work/01_data/jdbc_schema","r") as f:
    ncschema = f.read()

# load data from sqlite
driver = "org.sqlite.JDBC"
path = '/home/jovyan/work/01_data/db/database.sqlite'
url = "jdbc:sqlite:" + path
tablename = "loan"

DF = spark.read \
.option("driver", driver) \
.option("customSchema", ncschema) \
.jdbc(url, tablename)

# register sql table
DF.createOrReplaceTempView("loan")
```

https://www.kaggle.com/wendykan/lending-club-loan-data

## A glance at single record


```python
DF.limit(1).show(vertical=True)
```

    -RECORD 0--------------------------------------------------------
     id                                         |                    
     member_id                                  |                    
     loan_amnt                                  | 2500               
     funded_amnt                                | 2500               
     funded_amnt_inv                            | 2500               
     term                                       |  36 months         
     int_rate                                   | 13.56              
     installment                                | 84.92              
     grade                                      | C                  
     sub_grade                                  | C1                 
     emp_title                                  | Chef               
     emp_length                                 | 10+ years          
     home_ownership                             | RENT               
     annual_inc                                 | 55000              
     verification_status                        | Not Verified       
     issue_d                                    | Dec-2018           
     loan_status                                | Current            
     pymnt_plan                                 | n                  
     url                                        |                    
     desc                                       |                    
     purpose                                    | debt_consolidation 
     title                                      | Debt consolidation 
     zip_code                                   | 109xx              
     addr_state                                 | NY                 
     dti                                        | 18.24              
     delinq_2yrs                                | 0                  
     earliest_cr_line                           | Apr-2001           
     inq_last_6mths                             | 1                  
     mths_since_last_delinq                     | 0                  
     mths_since_last_record                     | 45                 
     open_acc                                   | 9                  
     pub_rec                                    | 1                  
     revol_bal                                  | 4341               
     revol_util                                 | 10.3               
     total_acc                                  | 34                 
     initial_list_status                        | w                  
     out_prncp                                  | 2386.02            
     out_prncp_inv                              | 2386.02            
     total_pymnt                                | 167.02             
     total_pymnt_inv                            | 167.02             
     total_rec_prncp                            | 113.98             
     total_rec_int                              | 53.04              
     total_rec_late_fee                         | 0.00               
     recoveries                                 | 0.00               
     collection_recovery_fee                    | 0.00               
     last_pymnt_d                               | Feb-2019           
     last_pymnt_amnt                            | 84.92              
     next_pymnt_d                               | Mar-2019           
     last_credit_pull_d                         | Feb-2019           
     collections_12_mths_ex_med                 | 0                  
     mths_since_last_major_derog                | 0                  
     policy_code                                | 1                  
     application_type                           | Individual         
     annual_inc_joint                           | 0                  
     dti_joint                                  |                    
     verification_status_joint                  |                    
     acc_now_delinq                             | 0                  
     tot_coll_amt                               | 0                  
     tot_cur_bal                                | 16901              
     open_acc_6m                                | 2                  
     open_act_il                                | 2                  
     open_il_12m                                | 1                  
     open_il_24m                                | 2                  
     mths_since_rcnt_il                         | 2                  
     total_bal_il                               | 12560              
     il_util                                    | 69                 
     open_rv_12m                                | 2                  
     open_rv_24m                                | 7                  
     max_bal_bc                                 | 2137               
     all_util                                   | 28                 
     total_rev_hi_lim                           | 42000              
     inq_fi                                     | 1                  
     total_cu_tl                                | 11                 
     inq_last_12m                               | 2                  
     acc_open_past_24mths                       | 9                  
     avg_cur_bal                                | 1878               
     bc_open_to_buy                             | 34360              
     bc_util                                    | 5.9                
     chargeoff_within_12_mths                   | 0                  
     delinq_amnt                                | 0                  
     mo_sin_old_il_acct                         | 140                
     mo_sin_old_rev_tl_op                       | 212                
     mo_sin_rcnt_rev_tl_op                      | 1                  
     mo_sin_rcnt_tl                             | 1                  
     mort_acc                                   | 0                  
     mths_since_recent_bc                       | 1                  
     mths_since_recent_bc_dlq                   | 0                  
     mths_since_recent_inq                      | 2                  
     mths_since_recent_revol_delinq             | 0                  
     num_accts_ever_120_pd                      | 0                  
     num_actv_bc_tl                             | 2                  
     num_actv_rev_tl                            | 5                  
     num_bc_sats                                | 3                  
     num_bc_tl                                  | 3                  
     num_il_tl                                  | 16                 
     num_op_rev_tl                              | 7                  
     num_rev_accts                              | 18                 
     num_rev_tl_bal_gt_0                        | 5                  
     num_sats                                   | 9                  
     num_tl_120dpd_2m                           | 0                  
     num_tl_30dpd                               | 0                  
     num_tl_90g_dpd_24m                         | 0                  
     num_tl_op_past_12m                         | 3                  
     pct_tl_nvr_dlq                             | 100                
     percent_bc_gt_75                           | 0                  
     pub_rec_bankruptcies                       | 1                  
     tax_liens                                  | 0                  
     tot_hi_cred_lim                            | 60124              
     total_bal_ex_mort                          | 16901              
     total_bc_limit                             | 36500              
     total_il_high_credit_limit                 | 18124              
     revol_bal_joint                            | 0                  
     sec_app_earliest_cr_line                   |                    
     sec_app_inq_last_6mths                     | 0                  
     sec_app_mort_acc                           | 0                  
     sec_app_open_acc                           | 0                  
     sec_app_revol_util                         | 0                  
     sec_app_open_act_il                        | 0                  
     sec_app_num_rev_accts                      | 0                  
     sec_app_chargeoff_within_12_mths           | 0                  
     sec_app_collections_12_mths_ex_med         | 0                  
     sec_app_mths_since_last_major_derog        | 0                  
     hardship_flag                              | N                  
     hardship_type                              |                    
     hardship_reason                            |                    
     hardship_status                            |                    
     deferral_term                              | 0                  
     hardship_amount                            |                    
     hardship_start_date                        |                    
     hardship_end_date                          |                    
     payment_plan_start_date                    |                    
     hardship_length                            | 0                  
     hardship_dpd                               | 0                  
     hardship_loan_status                       |                    
     orig_projected_additional_accrued_interest |                    
     hardship_payoff_balance_amount             |                    
     hardship_last_payment_amount               |                    
     disbursement_method                        | Cash               
     debt_settlement_flag                       | N                  
     debt_settlement_flag_date                  |                    
     settlement_status                          |                    
     settlement_date                            |                    
     settlement_amount                          |                    
     settlement_percentage                      | 0                  
     settlement_term                            | 0                  
    


## Target

In this task target is a DEFAULT event defined as a `loan_status` which is taking on the following levels:   
* Charged off
* Default
* Does not meet the credit policy. Status: Charged Off
* Late (31-120 days)

[statuses meaning](https://help.lendingclub.com/hc/en-us/articles/216109367-What-do-the-different-Note-statuses-mean-) 

Few words about DEFAULT theory:
* DEFAULT is a set of events/status that bank classify as an incapability to pay back a loan.   
* To predict beforehand if client is at risk of default, bank use different tools that assess this probability.   
* **Reactive/ application scoring**   - the main aim of reactive scoring is to forecast the credit quality of loan applications submitted by customers. It attempts to predict the applicant’s probability of default if the application were accepted.   
* **Behavioral scoring** is used to review contracts that have already been formalized by incorporating information on customer behavior and on the contract itself. Unlike reactive scoring, it is an analysis, i.e. once the contract has been granted. 
* **Proactive scoring** tools take into account the same variables as behavioral scorings, but they have a different purpose, as they provide an overall ranking of the customer, rather than of a specific transaction. This customer perspective is supplemented by adjustments that depend on the type of product. 



<img src="https://i.imgur.com/D1Aszpq.jpg" width="700" >

Source: https://shareholdersandinvestors.bbva.com/microsites/bbva2012/en/Riskmanagement/ProbabilityofdefaultPD.html    



```python
default_map = {
    'Fully Paid': '0',
    'Default' : '1',
    'In Grace Period': '0',
    'Does not meet the credit policy. Status:Fully Paid' : '0',
    'Charged Off': '1',
    'Late (31-120 days)' : '1',
    'Current' : '0',
    'Does not meet the credit policy. Status:Charged Off' : '1',
    'Late (16-30 days)' : '0'
}
```


```python
print("Loan statuses distribution")
spark.sql(
    "SELECT loan_status,loan_status as default, count(*) as cnt from loan group by loan_status order by cnt desc") \
    .replace(default_map, subset="default").show(truncate=False)

```

    +---------------------------------------------------+-------+-------+
    |loan_status                                        |default|cnt    |
    +---------------------------------------------------+-------+-------+
    |Fully Paid                                         |0      |1041952|
    |Current                                            |0      |919695 |
    |Charged Off                                        |1      |261655 |
    |Late (31-120 days)                                 |1      |21897  |
    |In Grace Period                                    |0      |8952   |
    |Late (16-30 days)                                  |0      |3737   |
    |Does not meet the credit policy. Status:Fully Paid |0      |1988   |
    |Does not meet the credit policy. Status:Charged Off|1      |761    |
    |Default                                            |1      |31     |
    +---------------------------------------------------+-------+-------+
    



```python
import pyspark.sql.functions as f

default_in_time = DF.withColumn('default', f.col('loan_status')) \
    .replace(default_map, subset="default") \
    .select(f.col('default'), f.to_date("issue_d",'MMM-yyyy').alias("issue_d")) \
    .groupBy(['issue_d']) \
    .agg(f.sum('default').alias('default_cnt'), (f.sum('default')/f.count('default')).alias('default_rate')
        ,f.count('default').alias("cnt")).orderBy('issue_d').toPandas()
```


```python
current_in_time = spark.sql("select distinct issue_d, count(*) as cnt from loan where loan_status = 'Current' group by issue_d order by issue_d") \
    .withColumn("issue_d",f.to_date("issue_d",'MMM-yyyy')).toPandas()

```


```python
from matplotlib.ticker import FuncFormatter
import pyspark.sql.functions as f

sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

sns.lineplot(data=default_in_time, x='issue_d', y='default_rate', color='navy', alpha=1, linewidth=2.5, ax=ax1, label='% of clients at default')
sns.lineplot(data=default_in_time, x='issue_d', y='cnt', color='black',alpha=0.4, dashes=True, linewidth=2.5, ax=ax2, label='number of records')
sns.lineplot(data=current_in_time, x='issue_d', y='cnt', color='#a3a3c2',alpha=0.6, dashes=True, linewidth=2.5, ax=ax2, label='number of records - status current')

ax2.lines[0].set_linestyle("--")
ax2.lines[1].set_linestyle("--")

ax1.legend(loc = (.05,.25), frameon = False)
ax2.legend(loc = (.05, .15), frameon = False)

ax1.set(xlabel='issue date', ylabel='default rate')
ax2.set(ylabel='number of records')

plt.title('Loan data: percentage of clinets at DEFAULT and number of records in time', y=1.05, fontsize = 16)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0f}'.format(y))) 

```


![png](01_Target_docker_spark_files/01_Target_docker_spark_14_0.png)


After 2016 default ratio is sharply decreasing contrariwise to the number of loans in status `current`.    
The younger the loan the lower risk of default. 


```python
_tmp = spark.sql('select term, count(*) as cnt from loan group by term').toPandas()
```


```python
_tmp.assign(prc = lambda x:x['cnt']/(1609754+650914))
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
      <th>term</th>
      <th>cnt</th>
      <th>prc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36 months</td>
      <td>1609754</td>
      <td>0.71207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60 months</td>
      <td>650914</td>
      <td>0.28793</td>
    </tr>
  </tbody>
</table>
</div>



Acording to the website [www.lendingclub.com](https://www.lendingclub.com/info/download-data.action) this dataset contains mix of current and past loans:
>... files contain complete loan data for all loans issued through the time period stated, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter.


I will build predictive model of default according to **Reactive scoring** definition:
>Reactive scoring - the main aim of reactive scoring is to forecast the credit quality of loan applications submitted by customers. It attempts to predict the applicant’s probability of default if the application were accepted.

The default rate accounts for actually realized defaults over a given period, while PD is the predicted probability that a pool of obligors will default over the predefined future time horizon. This time horizon is– typically 12 months.

In this task target is a DEFAULT event defined as a `loan_status` which is taking on the following levels:   
* Charged off
* Default
* Does not meet the credit policy. Status: Charged Off
* Late (31-120 days)

The time horizon is not defined in task. 

Since the dataset is a mix of past and present loans I've decided to take a horizon of 36 months - it's the prevailing loan length time. 

### select  population for modeling


I choose only records with issue date less or equal to 2016-03:
* I estimate the "present date" on March/April of 2019 (its the max value of field `next_pymnt_d`), 
* in description is `The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter`
* most of loans (70%) has term is issued for 36 months,
* 2019.03 minus 36 months is 2016.03

In result 1 021 327 loans will be included (45% of whole dataset).


```python
print("Distribution of 'next_pymnt_d': (since 2018-08) \n")

DF.select(f.to_date("next_pymnt_d",'MMM-yyyy').alias('next_pymnt_d')) \
          .filter(f.col("next_pymnt_d") > f.to_date(f.lit('2018-06-01'))) \
          .groupBy("next_pymnt_d").count().orderBy('next_pymnt_d') \
          .show(200, False)

```

    Distribution of 'next_pymnt_d': (since 2018-08) 
    
    +------------+------+
    |next_pymnt_d|count |
    +------------+------+
    |2018-08-01  |1     |
    |2018-09-01  |2     |
    |2018-12-01  |3     |
    |2019-02-01  |406   |
    |2019-03-01  |953821|
    |2019-04-01  |78    |
    +------------+------+
    



```python
_tmp = DF.select(f.to_date("issue_d",'MMM-yyyy').alias('issue_d')) \
          .select(f.when(f.col("issue_d") > f.to_date(f.lit('2016-03-01')),"N").otherwise("Y").alias("Include")) \
          .groupBy("Include").count().toPandas()
_tmp.assign(prc = lambda x:x['count']/(sum(_tmp['count'].values))) 
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
      <th>Include</th>
      <th>count</th>
      <th>prc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Y</td>
      <td>1021327</td>
      <td>0.451781</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N</td>
      <td>1239341</td>
      <td>0.548219</td>
    </tr>
  </tbody>
</table>
</div>




```python
import datetime
default_in_time_in =default_in_time[(default_in_time['issue_d']<=datetime.datetime.strptime('2016-03', '%Y-%m').date())]
default_in_time_out =default_in_time[(default_in_time['issue_d']>datetime.datetime.strptime('2016-03', '%Y-%m').date())]

```


```python
from matplotlib.ticker import FuncFormatter

sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(10,6))

sns.lineplot(data=default_in_time_in, x='issue_d', y='default_rate', color='navy', alpha=1, 
             linewidth=2.5, ax=ax1, label='included')
sns.lineplot(data=default_in_time_out, x='issue_d', y='default_rate', color='red', alpha=1, 
             linewidth=2.5, ax=ax1, label='excluded')
ax1.set(xlabel='issue date', ylabel='default rate')
plt.title('Loan data: percentage of clients in DEFAULT', y=1.05, fontsize = 16)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

```


![png](01_Target_docker_spark_files/01_Target_docker_spark_25_0.png)


## Variables

Since I've decided to build application scoring I choose only the variable that are avilible at the moment of credit application.    
They are listed in "browseNotes" tab of LCDataDictionary.xlsx


```python
import re
import unicodedata
def unic(string): return unicodedata.normalize("NFKD", string)
```


```python
LCDataDictionary = pd.read_excel('../01_data/LCDataDictionary.xlsx',
                             sheet_name='browseNotes')
```


```python
bn_or=[unic(i) for i in LCDataDictionary.BrowseNotesFile.tolist()[:-2]]
bn_md=[re.sub("[_ ]","",unic(i).lower()) for i in LCDataDictionary.BrowseNotesFile.tolist()[:-2]]
ds=DF.columns
```


```python
selected_variables = ['issue_d']+[i for i in ds if re.sub("[_ ]","",i).lower() in bn_md]+['mths_since_recent_inq','last_credit_pull_d','verification_status_joint','mo_sin_old_il_acct','loan_status']

```


```python
with codecs.open('../01_data/variables_browseNotes_select.txt','w') as f:
    for item in selected_variables:
        f.write("%s\n" % item)
    
```

manually maped 
```
creditPullD -> last_credit_pull_d
mthsSinceMostRecentInq -> mths_since_recent_inq
verified_status_joint -> verification_status_joint
mths_since_oldest_il_open -> mo_sin_old_il_acct
```

Not in dataset

```
acceptD
effective_int_rate
expD
ficoRangeHigh
ficoRangeLow
ils_exp_d
isIncV
listD
msa
mths_since_oldest_il_open
mthsSinceRecentLoanDelinq
reviewStatus
reviewStatusD
serviceFeeRate
sec_app_fico_range_low
sec_app_fico_range_high
```


```python
with codecs.open('../01_data/variables_browseNotes_select.txt','r') as f:
    variables = [i[:-1] for i in f.readlines()]

```


```python
DF.select(variables).limit(1).show(vertical=True)
```

    -RECORD 0-------------------------------------------------
     issue_d                             | Dec-2018           
     id                                  |                    
     member_id                           |                    
     loan_amnt                           | 2500               
     funded_amnt                         | 2500               
     term                                |  36 months         
     int_rate                            | 13.56              
     installment                         | 84.92              
     grade                               | C                  
     sub_grade                           | C1                 
     emp_title                           | Chef               
     emp_length                          | 10+ years          
     home_ownership                      | RENT               
     annual_inc                          | 55000              
     url                                 |                    
     desc                                |                    
     purpose                             | debt_consolidation 
     title                               | Debt consolidation 
     zip_code                            | 109xx              
     addr_state                          | NY                 
     dti                                 | 18.24              
     delinq_2yrs                         | 0                  
     earliest_cr_line                    | Apr-2001           
     inq_last_6mths                      | 1                  
     mths_since_last_delinq              | 0                  
     mths_since_last_record              | 45                 
     open_acc                            | 9                  
     pub_rec                             | 1                  
     revol_bal                           | 4341               
     revol_util                          | 10.3               
     total_acc                           | 34                 
     initial_list_status                 | w                  
     collections_12_mths_ex_med          | 0                  
     mths_since_last_major_derog         | 0                  
     application_type                    | Individual         
     annual_inc_joint                    | 0                  
     dti_joint                           |                    
     acc_now_delinq                      | 0                  
     tot_coll_amt                        | 0                  
     tot_cur_bal                         | 16901              
     open_acc_6m                         | 2                  
     open_act_il                         | 2                  
     open_il_12m                         | 1                  
     open_il_24m                         | 2                  
     mths_since_rcnt_il                  | 2                  
     total_bal_il                        | 12560              
     il_util                             | 69                 
     open_rv_12m                         | 2                  
     open_rv_24m                         | 7                  
     max_bal_bc                          | 2137               
     all_util                            | 28                 
     total_rev_hi_lim                    | 42000              
     inq_fi                              | 1                  
     total_cu_tl                         | 11                 
     inq_last_12m                        | 2                  
     acc_open_past_24mths                | 9                  
     avg_cur_bal                         | 1878               
     bc_open_to_buy                      | 34360              
     bc_util                             | 5.9                
     chargeoff_within_12_mths            | 0                  
     delinq_amnt                         | 0                  
     mo_sin_old_rev_tl_op                | 212                
     mo_sin_rcnt_rev_tl_op               | 1                  
     mo_sin_rcnt_tl                      | 1                  
     mort_acc                            | 0                  
     mths_since_recent_bc                | 1                  
     mths_since_recent_revol_delinq      | 0                  
     num_accts_ever_120_pd               | 0                  
     num_actv_bc_tl                      | 2                  
     num_actv_rev_tl                     | 5                  
     num_bc_sats                         | 3                  
     num_bc_tl                           | 3                  
     num_il_tl                           | 16                 
     num_op_rev_tl                       | 7                  
     num_rev_accts                       | 18                 
     num_rev_tl_bal_gt_0                 | 5                  
     num_sats                            | 9                  
     num_tl_120dpd_2m                    | 0                  
     num_tl_30dpd                        | 0                  
     num_tl_90g_dpd_24m                  | 0                  
     num_tl_op_past_12m                  | 3                  
     pct_tl_nvr_dlq                      | 100                
     percent_bc_gt_75                    | 0                  
     pub_rec_bankruptcies                | 1                  
     tax_liens                           | 0                  
     tot_hi_cred_lim                     | 60124              
     total_bal_ex_mort                   | 16901              
     total_bc_limit                      | 36500              
     total_il_high_credit_limit          | 18124              
     revol_bal_joint                     | 0                  
     sec_app_earliest_cr_line            |                    
     sec_app_inq_last_6mths              | 0                  
     sec_app_mort_acc                    | 0                  
     sec_app_open_acc                    | 0                  
     sec_app_revol_util                  | 0                  
     sec_app_open_act_il                 | 0                  
     sec_app_num_rev_accts               | 0                  
     sec_app_chargeoff_within_12_mths    | 0                  
     sec_app_collections_12_mths_ex_med  | 0                  
     sec_app_mths_since_last_major_derog | 0                  
     disbursement_method                 | Cash               
     mths_since_recent_inq               | 2                  
     last_credit_pull_d                  | Feb-2019           
     verification_status_joint           |                    
     mo_sin_old_il_acct                  | 140                
    


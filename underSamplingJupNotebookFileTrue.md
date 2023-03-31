```python
from sklearn.model_selection import train_test_split,KFold
import pandas as pd
import numpy as np 
import sqlite3
import random
from imblearn.under_sampling import RandomUnderSampler

#Read data
con = sqlite3.connect('FPA_FOD_20170508.sqlite')
Fires=pd.read_sql_query('select * from "Fires"',con)
con.close()

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
      <th>OBJECTID</th>
      <th>FOD_ID</th>
      <th>FPA_ID</th>
      <th>SOURCE_SYSTEM_TYPE</th>
      <th>SOURCE_SYSTEM</th>
      <th>NWCG_REPORTING_AGENCY</th>
      <th>NWCG_REPORTING_UNIT_ID</th>
      <th>NWCG_REPORTING_UNIT_NAME</th>
      <th>SOURCE_REPORTING_UNIT</th>
      <th>SOURCE_REPORTING_UNIT_NAME</th>
      <th>...</th>
      <th>FIRE_SIZE_CLASS</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>OWNER_CODE</th>
      <th>OWNER_DESCR</th>
      <th>STATE</th>
      <th>COUNTY</th>
      <th>FIPS_CODE</th>
      <th>FIPS_NAME</th>
      <th>Shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>FS-1418826</td>
      <td>FED</td>
      <td>FS-FIRESTAT</td>
      <td>FS</td>
      <td>USCAPNF</td>
      <td>Plumas National Forest</td>
      <td>0511</td>
      <td>Plumas National Forest</td>
      <td>...</td>
      <td>A</td>
      <td>40.036944</td>
      <td>-121.005833</td>
      <td>5.0</td>
      <td>USFS</td>
      <td>CA</td>
      <td>63</td>
      <td>063</td>
      <td>Plumas</td>
      <td>b'\x00\x01\xad\x10\x00\x00\xe8d\xc2\x92_@^\xc0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>FS-1418827</td>
      <td>FED</td>
      <td>FS-FIRESTAT</td>
      <td>FS</td>
      <td>USCAENF</td>
      <td>Eldorado National Forest</td>
      <td>0503</td>
      <td>Eldorado National Forest</td>
      <td>...</td>
      <td>A</td>
      <td>38.933056</td>
      <td>-120.404444</td>
      <td>5.0</td>
      <td>USFS</td>
      <td>CA</td>
      <td>61</td>
      <td>061</td>
      <td>Placer</td>
      <td>b'\x00\x01\xad\x10\x00\x00T\xb6\xeej\xe2\x19^\...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>FS-1418835</td>
      <td>FED</td>
      <td>FS-FIRESTAT</td>
      <td>FS</td>
      <td>USCAENF</td>
      <td>Eldorado National Forest</td>
      <td>0503</td>
      <td>Eldorado National Forest</td>
      <td>...</td>
      <td>A</td>
      <td>38.984167</td>
      <td>-120.735556</td>
      <td>13.0</td>
      <td>STATE OR PRIVATE</td>
      <td>CA</td>
      <td>17</td>
      <td>017</td>
      <td>El Dorado</td>
      <td>b'\x00\x01\xad\x10\x00\x00\xd0\xa5\xa0W\x13/^\...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>FS-1418845</td>
      <td>FED</td>
      <td>FS-FIRESTAT</td>
      <td>FS</td>
      <td>USCAENF</td>
      <td>Eldorado National Forest</td>
      <td>0503</td>
      <td>Eldorado National Forest</td>
      <td>...</td>
      <td>A</td>
      <td>38.559167</td>
      <td>-119.913333</td>
      <td>5.0</td>
      <td>USFS</td>
      <td>CA</td>
      <td>3</td>
      <td>003</td>
      <td>Alpine</td>
      <td>b'\x00\x01\xad\x10\x00\x00\x94\xac\xa3\rt\xfa]...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>FS-1418847</td>
      <td>FED</td>
      <td>FS-FIRESTAT</td>
      <td>FS</td>
      <td>USCAENF</td>
      <td>Eldorado National Forest</td>
      <td>0503</td>
      <td>Eldorado National Forest</td>
      <td>...</td>
      <td>A</td>
      <td>38.559167</td>
      <td>-119.933056</td>
      <td>5.0</td>
      <td>USFS</td>
      <td>CA</td>
      <td>3</td>
      <td>003</td>
      <td>Alpine</td>
      <td>b'\x00\x01\xad\x10\x00\x00@\xe3\xaa.\xb7\xfb]\...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
#Keeping columns as mentioned on the word doc
fire1 = Fires[['NWCG_REPORTING_AGENCY','CONT_DATE','CONT_DOY','LONGITUDE',"LATITUDE",\
             'NWCG_REPORTING_UNIT_ID','SOURCE_REPORTING_UNIT','OWNER_CODE',\
              'SOURCE_SYSTEM_TYPE','DISCOVERY_DATE','DISCOVERY_DOY','FIRE_SIZE',\
             'STAT_CAUSE_DESCR']]
#Removed row with classes Miscellaneous and missing/undefined
fireNoMiss=fire1.loc[(fire1['STAT_CAUSE_DESCR'] !=  "Miscellaneous") & \
                     (fire1['STAT_CAUSE_DESCR'] != "Missing/Undefined")]
```


```python

xVal=fireNoMiss.loc[:,fireNoMiss.columns != 'STAT_CAUSE_DESCR']
yVal=fireNoMiss['STAT_CAUSE_DESCR']

#Training and test set split
xTrain,xTest,yTrain,yTest=train_test_split(xVal,yVal,\
                           test_size=0.1,random_state =441)

```


```python
#Perform undersampling
newSampStrat=RandomUnderSampler(sampling_strategy="not minority",random_state=441) #Goal balance all classes
newX,newY=newSampStrat.fit_resample(xTrain,yTrain) #perform the balancing newX and newY are balanced X and y
newY.value_counts()
```




    Arson             3423
    Campfire          3423
    Children          3423
    Debris Burning    3423
    Equipment Use     3423
    Fireworks         3423
    Lightning         3423
    Powerline         3423
    Railroad          3423
    Smoking           3423
    Structure         3423
    Name: STAT_CAUSE_DESCR, dtype: int64




```python

```




    138994




```python

```

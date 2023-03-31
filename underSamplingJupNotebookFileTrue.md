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
newY.value_counts()  #print result showing the nunmber of observation in each class
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





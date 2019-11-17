# Python 基础

### OnlineJudge

http://acm.zju.edu.cn/onlinejudge/showProblem.do?problemId=1

数据库连接



mongoDb

```python
import pandas as pd
import pymongo

# name of the CSV file to read from and SQLite database
r_filenameCSV = '../../Data/Chapter01/realEstate_trans.csv'

# read the data
csv_read = pd.read_csv(r_filenameCSV)

# transform sale_date to a datetime object
csv_read['sale_date'] = pd.to_datetime(csv_read['sale_date'])

# connect to the MongoDB database
client = pymongo.MongoClient()

# and select packt database
db = client['packt']

# then connect to real_estate collection
real_estate = db['real_estate']

# if there are any documents stored already -- remove them
if real_estate.count() > 0:
    real_estate.remove()

# and then insert the data
real_estate.insert(csv_read.to_dict(orient='records'))

# print out the top 10 documents 
# sold in ZIP codes 95841 and 95842
sales = real_estate.find({'zip': {'$in': [95841, 95842]}})
for sale in sales.sort('_id').limit(10):
    print(sale)
```



postgresql



```python
import pandas as pd
import sqlalchemy as sa

# name of the CSV file to read from
r_filenameCSV = '../../Data/Chapter01/realEstate_trans.csv'

# database credentials
usr  = 'drabast'
pswd = 'pAck7!B0ok'

# create the connection to the database
engine = sa.create_engine(
    'postgresql://{0}:{1}@localhost:5432/{0}' \
    .format(usr, pswd)
)

# read the data
csv_read = pd.read_csv(r_filenameCSV)

# transform sale_date to a datetime object
csv_read['sale_date'] = pd.to_datetime(csv_read['sale_date'])

# store the data in the database
csv_read.to_sql('real_estate', engine, if_exists='replace')

# print the top 10 rows from the database
query = 'SELECT * FROM real_estate LIMIT 10'
top10 = pd.read_sql_query(query, engine)
print(top10)
```



Sqlite

```python
import pandas as pd
import sqlalchemy as sa

# name of the CSV file to read from and SQLite database
r_filenameCSV = '../../Data/Chapter01/realEstate_trans.csv'
rw_filenameSQLite = '../../Data/Chapter01/realEstate_trans.db'

# create the connection to the database
engine = sa.create_engine(
    'sqlite:///{0}'.format(rw_filenameSQLite)
)

# read the data
csv_read = pd.read_csv(r_filenameCSV)

# transform sale_date to a datetime object
csv_read['sale_date'] = pd.to_datetime(csv_read['sale_date'])

# store the data in the database
csv_read.to_sql('real_estate', engine, if_exists='replace')

# print the top 10 rows from the database
query = 'SELECT * FROM real_estate LIMIT 10'
top10 = pd.read_sql_query(query, engine)
print(top10)
```




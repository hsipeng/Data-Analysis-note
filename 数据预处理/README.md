# 数据规整



## 数据科学家80%实践都花费在来这些清洗任务上?

> 好的数据分析师必定是一名数据清洗高手，要知道杂整个数据分析过程中，不论是实践还是功夫上，数据清洗都占到了80%

## 数据质量的准则

**完全合一**

1. **完**整性： 单条数据是否存在空值
2. **全**面性： 观察某一列的全部数值，通过尝试判定该列是否有问题
3. **合**法性：数据的类型、内容、大小的合法性。
4. 唯**一**性：数据是否存在重复记录

## 清洗数据，一一击破

### 1. 完整性

#### 问题1: 缺失值

三种方法：

删除：删除缺失值记录

均值：当前列的平均值

高频：当前列出现频率最高的数据



```python
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 高频
age_maxf = train_features['Age'].value_counts().index[0]
train_features['Age'].fillna(age_maxf, inplace=True)

```



#### 问题2: 空行

```python
df.dropna(how='all', inplace=True)
```



### 2. 全面性

将磅(lbs)转换为千克(kgs)

![image-20190903132616689](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190903132616689.png)

### 3. 合理性

### 问题：非ASCII 字符

```python
df['first_name'].replace({r'[^\x00-\x7F]+':''},regex=True, inplace=True)

```

### 4. 唯一性

#### 问题1: 一列有多个参数

```python
# 切分名字，删除源数据列
df[['first_name','last_name']] = df['name'].str.splict(expand=True)
df.drop('name', axis=1, inplace=True)
```



#### 问题2: 重复数据

```python
# 删除重复数据行
df.drop_duplicates(['first_name', 'last_name'],inplace=True)

```



![image-20190903133209323](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190903133209323.png)

11-数据科学家80%实践都花费在来这些清洗任务上？


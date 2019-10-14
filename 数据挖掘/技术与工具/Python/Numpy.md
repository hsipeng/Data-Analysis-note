# Python科学计算:用NumPy快速处理数据

## 使用NumPy 让你的 Python 的科学计算更高效

`Numpy`的数据结构存储在一个连续的内存块

使用矢量化指令计算

矩阵计算可以采用多线程的方式

> 避免采用隐式拷贝，而是采用就地操作的方式， 例如 想让 一个数值 x 是原来的两倍, 可以直接写成 `x*2`, 不要写成 y = `x*2`



NumPy 有两个重要的对象  

ndarray(N-dimensional array object)  解决多维数组的问题

ufunc (universal function object) 解决对数组进行处理的函数



## ndarray 对象

Ndarray 实际上是多维数组的含义。在 Numpy 数组中，维数称为秩(rank)， 一维数组的秩为1，二维数组的秩为2。在numpy 中，每一个线性数组称为一个轴(axes)， 其实秩就是描述轴的数量



### 创建数组

```python
import numpy as np

a = np.array([1,2,3])
print(a.shape)
print(a.dtype)
```



### 结构数组

想统计一个班级里面学生的姓名、年龄、以及语文、英语、数学成绩怎么办？用下标表示不同字段，不是很显性。



```python
import numpy as np

persontype = np.dtype({
  'names': ['name', 'age', 'chinese', 'math', 'english'],
  'formats': ['S32', 'i', 'i', 'f']
})

peoples = np.array([
  ('ZhangFei', 32, 75, 100, 90),
	('GuanYu', 24, 85, 96, 88.5),
	('ZhaoYun', 28, 85, 92, 96.5),
	('HuangZhong', 29, 65, 85, 100)],
	dtype=persontype)

ages = peoples[:]['age']
print(np.mean(ages))
```



## ufunc 运算

对数组中每个元素进行函数操作

### 连续数组创建

```python
x1 = np.arange(1,11,2)
x2 = np.linspace(1,9,5)
```

`arange` 和 `linspace`都是创建等差数组。

`orange`是类似内置函数`range()`, 通过 **初始值**,**终值**，**步长**来创建等差数列的一维数组，默认不包括终值的。

`linspace` 是 linear space 的缩写，嗲表线性等分向量的含义。 `linspace()` 通过指定 **初始值**、**终值**、**元素个数**来创建等差数列的一维数组，默认包括终值的。



### 算术运算

```python
x1 = np.arange(1,11,2)
x2 = np.linspace(1,9,5)

print(np.add(x1,x2)) # 加
print(np.subtract(x1,x2))# 减
print(np.multiply(x1,x2)) # 乘
print(np.divide(x1,x2)) # 除
print(np.power(x1,x2)) # 求n次方
print(np.remainder(x1,x2)) # 取余数
```



### 统计函数

#### 计算数组/矩阵中的最大值函数amax()、最小值函数amin()

```python
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(np.amin(a)) # 全部元素的最小值
print(np.amin(a,0))# axis =0 轴的最小值，把数据看成 [1,4,7],[2,5,8],[3,6,9]
# 结果 [1 2 3]
print(np.amin(a,1))# axis =1 轴的最小值，把数据看成 [1,2,3],[4,5,6],[7,8,9]
# 结果 [1 4 7]

print(np.amax(a))
print(np.amax(a,0))
print(np.amax(a,1))
```



### 统计最大值与最小值之差ptp()



```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(np.ptp(a)) # 全部元素最大值与最小值之差 9-1=8
print(np.ptp(a,0)) # axis =0 轴，把数据看成 [1,4,7],[2,5,8],[3,6,9]
# 结果 [6 6 6]
print(np.ptp(a)) #axis =1 轴，把数据看成 [1,2,3],[4,5,6],[7,8,9]
# 结果 [2 2 2]
```



### 统计数组的百分位数 percentile()

`percentile()`代表着第p个百分位数，这里的p 取值范围是 0-100， 如果p=0，就是求最小值，如果p=50，就是求平均值, p=100，就是求最大值。axis=0 和 axis=1 两个轴上的p%百分位数.

```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(np.percentile(a, 50))
print(np.percentile(a, 50, axis=0))
print(np.percentile(a, 50, axis=1))
```



### 统计数组的中位数 median()、平均数mean()

```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 求中位数
print(np.median(a))
print(np.median(a, axis=0))
print(np.median(a, axis=0))

# 求平均数
print(np.mean(a))
print(np.mean(a, axis=0))
print(np.mean(a, axis=0))
```



### 统计数组中的加权平均数average()

```python
a = np.array([1,2,3,4])
wts = np.array([1,2,3,4])

print(np.average(a)) # (1+2+3+4)/4=2.5
print(np.average(a, weights=wts)) # (1*1 + 2*2 + 3*3 + 4*4)/(1+2+3+4)=3.0
```



### 统计数组中的标准差std()、方差var()



```python
a = np.array([1,2,3,4])

print(np.std(a))# 标准差是方差的算术平方根
print(np.var(a)) # 方差是每个数值与平均值之差的平方求和的平均值
# mean((x - x.mean())**2)
```



## NumPy 排序

`sort`函数

`sort(a, axis=-1, kind='quicksort', order=None)`

参数

- axis 默认-1， 即沿着最后一个轴进行排序， None， 扁平化的方式作为一个向量进行排序。
- Kind 可以指定 quicksort, merge sort, heapsort 分别进行快速排序，合并排序，堆排序。
- order 指定按照某个字段进行排序。



```python
a = np.array([[4,3,2],[2,4,1]])

print(np.sort(a))
print(np.sort(a, axis=None))
print(np.sort(a, axis=0))
print(np.sort(a, axis=1))
```



![image-20190903100633529](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190903100633529.png)



![image-20190903104241790](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190903104241790.png)





```python
import numpy as np

persontype = np.dtype({
  'names': ['name', 'chinese', 'english', 'math', 'total'],
  'formats': ['S32', 'i', 'i', 'i', 'i']
})

peoples = np.array([
    ('ZhangFei', 66, 65, 30, 0),
    ('GuanYu', 95, 85, 98, 0),
    ('ZhaoYun', 93, 92, 96, 0),
    ('HuangZhong', 90, 88, 77, 0),
    ('DianWei', 80, 90, 90, 0)],
    dtype=persontype)

print(peoples)


print("平均成绩:")
chinese = peoples[:]['chinese'] # 语文
print("语文: {}".format(np.mean(chinese)))

english = peoples[:]['english'] # 英语
print("英语: {}".format(np.mean(english)))

math = peoples[:]['math'] # 数学
print("数学: {}".format(np.mean(math)))


print("最小成绩:")
print("语文: {}".format(np.amin(chinese)))
print("英语: {}".format(np.amin(english)))
print("数学: {}".format(np.amin(math)))

print("最大成绩:")
print("语文: {}".format(np.amax(chinese)))
print("英语: {}".format(np.amax(english)))
print("数学: {}".format(np.amax(math)))

print("方差:")
print("语文: {}".format(np.var(chinese)))
print("英语: {}".format(np.var(english)))
print("数学: {}".format(np.var(math)))

print("标准差:")
print("语文: {}".format(np.std(chinese)))
print("英语: {}".format(np.std(english)))
print("数学: {}".format(np.std(math)))

peoples[:]['total'] = peoples[:]['chinese'] + peoples[:]['english'] + peoples[:]['math']
peoples

print(np.sort(peoples, order='total'))
```


# KNN



KNN (K-Nearest Neighbor)

KNN 的工作原理

"近朱者赤，近墨者黑"

1. 计算待分类物体与其他物体之间的距离
2. 统计距离最近的K个邻居
3. 对于K个最近的邻居，他们属于哪个分类最多，待分类物体就属于哪一类



* Python



如何用 KNN 对手写数字进行识别分类

手写数据集 MNIST

```python
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# 读取数据
digits = load_digits()
data = digits.data
print(data.shape)
print(digits.images[0])
print(digits.target[0])
plt.gray()
plt.imshow(digits.images[0])
plt.show()
k
# 数据划分
train_X, test_X = train_test_split(data, test_size=0.25, random_state=33)
train_y, test_y = train_test_split(digits.target, test_size=0.25, random_state=33)
# 数据标准化
ss = preprocessing.StandardScaler()
train_ss_X = ss.fit_transform(train_X)
test_ss_X = ss.fit_transform(test_X)

# KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_X, train_y)
predict = knn.predict(test_ss_X)
print('KNN准确率：%.4lf' % accuracy_score(predict, test_y))

# SVM分类器
svm = SVC()
svm.fit(train_ss_X, train_y)
predict = svm.predict(test_ss_X)
print('SVM准确率：%.4lf' % accuracy_score(predict, test_y))

# 朴素贝叶斯
# 特征不能为负值，用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_X = mm.fit_transform(train_X)
test_mm_X = mm.fit_transform(test_X)
nb = MultinomialNB()
nb.fit(train_mm_X, train_y)
predict = nb.predict(test_mm_X)
print('朴素贝叶斯准确率：%.4lf' % accuracy_score(predict, test_y))

# 决策树
dtc = DecisionTreeClassifier()
dtc.fit(train_ss_X, train_y)
predict = dtc.predict(test_ss_X)
print('决策树准确率：%.4lf' % accuracy_score(predict, test_y))



```



* R

```R
# k - 最近邻 KNN

# 用于不知道给定的预测因子下响应变量条件分布的情况

# class 包 knn

data("iris")

n <- nrow(iris)
ntrain <- round(n*0.6)
set.seed(333)
tindex <- sample(n, ntrain)
train_iris <- iris[tindex,]
test_iris <- iris[-tindex,]
head(train_iris)


library(class)

# 观察训练集的观测点

plot(train_iris$Petal.Length, train_iris$Petal.Width,pch=c(train_iris$Species))
legend('topleft', legend = c("setosa", "versicolor", "verginica"), pch=c(1,2,3), bty='o')


# 一个距离函数来定义一个数据点到另一个数据点之间的距离，通常度量标准叫欧几里得距离

# knn 参数, 
# 1. 训练集的预测因子
# 2. 测试集的预测因子
# 3. 训练集的响应值
# 4. 参数 K

train_x <- train_iris[,-5]
train_y <- train_iris[,5]
test_x <- test_iris[,-5]
test_y <- test_iris[,5]
prediction <- knn(train_x, test_x, train_y, k=5)

table(prediction, test_iris$Species)

(19+18+22)/nrow(test_iris)


# 错误总数
sum(prediction != test_y)

# 观察数
length(test_y)

```


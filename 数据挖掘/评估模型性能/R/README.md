# 评估模型性能

```R
# 数据科学步骤

# 数据处理
# 特征选择
# EDA 和 统计分析
# 评估模型性能


# 问题

# 1. 过拟合
# 训练集误差很小，但是测试集误差很大

library(kernlab)

data(spam)
set.seed(333)
sampleIndex <- sample(dim(spam)[1], size = 10)
sampleSpam <- spam[sampleIndex,]

# 探索性数据分析
spamSymbol <- (sampleSpam$type=="spam") + 1
plot(sampleSpam$capitalAve, pch =spamSymbol)
legend('topright', legend=c("nonspam", "spam"), pch=c(1,2))

# 查看所有的capitalAve 的值
sampleSpam$capitalAve



## 手工算法对数据进行分类

alg1 <- function(x){
  pred <- rep(NA, length(x))
  pred[x>2.7] <- "spam"
  pred[x<2.4] <- "nonspam"
  # addition rules result in overfitting
  pred[x<=2.45& x >=2.4] <- "spam"
  pred[x<=2.7 & x > 2.45] <- "nonspam"
  
  return(pred)
}

## 对比真实标签和预测标签的混合矩阵
table(alg1(sampleSpam$capitalAve), sampleSpam$type)

# 没有过拟合的
alg2 <- function(x){
  pred <- rep(NA, length(x))
  pred[x>2.8] <- "spam"
  pred[x<=2.8] <- "nonspam"
  
  return(pred)
}

table(alg2(sampleSpam$capitalAve), sampleSpam$type)


## 放到整个垃圾邮件数据集

# alg1
table(alg1(spam$capitalAve), spam$type)
sum(alg1(spam$capitalAve)!= spam$type)

# alg2

table(alg2(spam$capitalAve), spam$type)
sum(alg2(spam$capitalAve)!= spam$type)

# alg1 明显比 alg2 错误率高

# 处理方法

# 1. 第一种尝试减少特征数目
# 2. 第二种时正则化, 保留所有特征，但是减小特征变量值的量级
# glmnet() 算法, 参数 alpha=0, 岭回归， alpha=1， 套索模型


# 2. 偏差和方差
# 3. 干扰因子
# 4. 数据泄漏
# 5. 测定回归性能

# 测定回归性能

library(car)

data(Prestige)
Prestige_noNA <- na.omit(Prestige)
n <- nrow(Prestige_noNA)
ntrain <- round(n*0.7)
set.seed(333)
tindex <- sample(n, ntrain)
prestige_train <- Prestige_noNA[tindex,]
prestige_test <- Prestige_noNA[-tindex,]


## 计算误差率 MSE 或者 RMSE 
# y_hat 预测量, y 实际响应变量
rmse <- function(y_hat, y){
  return(sqrt(mean((y_hat-y)^2)))
}

## 拟合一个线性模型

lm1 <- lm(prestige~., data=prestige_train)

rmse_train <- rmse(predict(lm1, newdata=prestige_train),prestige_train$prestige)
rmse_train
rmse_test <- rmse(predict(lm1, newdata=prestige_test),prestige_test$prestige)
rmse_test



# RMSE 无法立刻清楚的展示一般情况下模型表现的特征
# R^2 来度量使用的模型比仅仅使用平均值要好多少倍
rsquared <- function(y_hat, y){
  mu <- mean(y)
  res <- mean((y_hat - y)^2)/ mean((mu - y)^2)
  rsquared <- (1-res)*100
  return(rsquared)
}

y_hat <- lm1$fitted.value
y <- prestige_train$prestige

rsquared(y_hat, y)

# 测试集
y_hat <- predict(lm1, newdata=prestige_test)
y <- prestige_test$prestige

rsquared(y_hat, y)


# 6. 测定分类性能
# 测定分类性能

library(randomForest)
# 下载文件
#download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "wine.csv")
# curl http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -o wine.csv

df <- read.csv("wine.csv", sep = ";", header = TRUE)

df$quality <- factor(df$quality)

n <- nrow(df)
ntrain <- round(n*0.7)
set.seed(333)
tindex <- sample(n ,ntrain)
wine_train <- df[tindex,]
wine_test <- df[-tindex,]

# randomForest 训练拟合模型

rf <- randomForest(quality~., data = wine_train, ntree=20, nodesize=5, mtry=9)

table(wine_test$quality, predict(rf, wine_test))

sum(wine_test$quality != predict(rf, wine_test))/nrow(wine_test)

# 7. 交叉验证
# 8. 其他(获取更多训练观测数据，特征降维，添加新特征，添加多项式特征，对正则化参数进行微调)

```


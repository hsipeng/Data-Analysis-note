# R 线性回归



## 一元线性回归

```R
## 回归

# 一元线性回归

# 基于特征变量(预测因子)和响应变量(预测的变量) 有线性关系

library(MASS)

data(Boston)
names(Boston)

lml <- lm(medv~rm, data=Boston)
lml

summary(lml)


names(lml)

coef(lml)

# attach 将 Boston 数据集放入R 的搜索路径中

attach(Boston)
# rm medv 数值对
plot(rm, medv, pch=20, xlab="AVG. # Rooms", ylab="Median Value")
# 最佳拟合回归线
lines(rm, lml$fitted, lwd=3)
## 预测 基于平均数6,预测房屋价值中位数
## 方法 1
coef(lml)[1] + coef(lml)[2]*6
## 方法2

newdata <- data.frame(rm=6)
predict(lml, newdata)


## 诊断图

par(mfrow=c(2,2))
plot(lml)

## Cook 距离
# 拟合残差应该在0 线附近跳动，形成一个水平带，没有残差是突出的
par(mfrow=c(1,1))
plot(cooks.distance(lml))


## 残差图

par(mfrow=c(1,1))
plot(predict(lml), residuals(lml))


```



## 多元线性回归



```R
## 多元线性回归
## 基于多个特征变量(预测因子)和响应变量(预测的变量) 有线性关系
library(car)

data(Prestige)

summary(Prestige)


head(Prestige)


## 定义训练集，测试集， 通常 六四分

Prestige_noNa <- na.omit(Prestige)
n <- nrow(Prestige_noNa) # number for observatios
ntrain <- round(n*0.6)
set.seed(333) # set seed for reproducible results
tindex <- sample(n, ntrain)
trainPrestige <- Prestige_noNa[tindex,]
testPrestige <- Prestige_noNa[-tindex,]

# 探索性数据分析
plot(trainPrestige$prestige, trainPrestige$education) # Trend
plot(trainPrestige$prestige, trainPrestige$income) # no Trend
plot(trainPrestige$prestige, trainPrestige$women) # no trend


# lm 函数 线性回归  公式 prestige~.

lm2 <- lm(prestige~., data = trainPrestige)
summary(lm2)

# 诊断图
# 1.预测值和残差图
plot(lm2$fitted, lm2$residuals)
# 2.残差索引图
plot(lm2$residuals, pch=19) # 没有趋势

## testPrestige 预测响应变量 prestige 
predict2 <- predict(lm2, newdata = testPrestige)

cor(predict2, testPrestige$prestige)
#3. qqnrom 和 qqline 函数 诊断 
# 展示分位数和分位数，确认残差是否成正态分布

rs <- residuals(lm2)
qqnorm(rs)
qqline(rs)



# 4. 使用模型中未使用的变量生成一张图表

plot(testPrestige$prestige, predict2, pch=c(testPrestige$type))
legend('topleft', legend=c("bc", "prof", "wc"), pch=c(1,2,3), bty="o")

```



## 多项式回归



```R
# 多项式回归
# 特征变量(预测因子)和响应变量(预测的变量) 有非线性关系
library(MASS)

data("Boston")
names(Boston)
fit_d1 <- lm(nox~dis, data=Boston)
summary(fit_d1)


# 一次多项式回归曲线 -- 线性拟合
plot(Boston$dis, Boston$nox)
lines(Boston$dis, fit_d1$fitted, col=2, lwd=3)

# 二次多项式回归曲线 -- 二次拟合

fit_d2 <- lm(nox~poly(dis,2, raw=TRUE), data=Boston)
summary(fit_d2)

plot(Boston$dis, Boston$nox)
lines(sort(Boston$dis), fit_d2$fitted.values[order(Boston$dis)], col=2, lwd=3)


# 三次多项式回归曲线 -- 三次拟合

fit_d3 <- lm(nox~poly(dis,3, raw=TRUE), data=Boston)
summary(fit_d3)

plot(Boston$dis, Boston$nox)
lines(sort(Boston$dis), fit_d3$fitted.values[order(Boston$dis)], col=2, lwd=3)

# anova 比较三个模型的假设校验， 方差分析用于测试模型是否有效解释数据
anova(fit_d1,fit_d2, fit_d3)

```


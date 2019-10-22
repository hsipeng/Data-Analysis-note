# 回归

* 线性回归

  * OLS(最小二乘法)
  * CART
  * KNN
  * 随机森林
  * SVM
  * 神经网络

* 逻辑回归

* 其他

  * 岭回归
  * 多元逻辑回归

  



回归

回归分析试图解释一组变量对另外一个变量的结果的影响



### 线性回归

线性回归是一种用来对若干输入变量与一个连续结果变量之间关系建模的分析技术。


$$
y=\beta_0 + \beta_1x_1 + \beta_2x_2 + ...+\beta_{p-1}x_{p-1}+\varepsilon
$$


### 逻辑回归

结果变量是分类的。


$$
ln(\frac{p}{1-p}) = y=\beta_0 + \beta_1x_1 + \beta_2x_2 + ...+\beta_{p-1}x_{p-1}
$$
$ln(\frac{p}{1-p})$ 称为对数差异比(log odds ratio) 或是p 的对数成败比率曲线(logit)



极大似然估计(MLE) 可以用来估计模型参数



接收者操作特征曲线(ROC)

真阳性率(TPR)和假阳性率(FPR)的对比图

ROC 曲线下的区域面积(AUC)



* R

线性回归

```R
###############################################################
# This code covers the code presented in 
# Section 6.1 Linear Regression
###############################################################

###############################################################
# Section 6.1.2
###############################################################

# Example in R

income_input = as.data.frame(  read.csv("c:/data/income.csv")   )
income_input[1:10,]

summary(income_input)

library(lattice)

splom(~income_input[c(2:5)],  groups=NULL, data=income_input, 
      axis.line.tck = 0,
      axis.text.alpha = 0)

results <- lm(Income~Age + Education + Gender, income_input)
summary(results)

results2 <- lm(Income ~ Age + Education, income_input)
summary(results2)

###############################################################
# this code from the text is for illustrative purposes only
# the income_input variable does not contain the U.S. states
results3 <- lm(Income~Age + Education,
               + Alabama,
               + Alaska,
               + Arizona,
               .
               .
               .
               + WestVirginia,
               + Wisconsin,
               income_input)
###############################################################

# compute confidence intevals for the model parameters
confint(results2, level = .95)

# compute a confidence interval on the expected income of a person
Age <- 41
Education <- 12
new_pt <- data.frame(Age, Education)

conf_int_pt <- predict(results2, new_pt, level=.95, interval="confidence")
conf_int_pt 

# compute a prediction interval on the income of the same person
pred_int_pt <- predict(results2, new_pt, level=.95, interval="prediction")
pred_int_pt

###############################################################
# section 6.1.3 Diagnostics 
###############################################################

with(results2, {
  plot(fitted.values, residuals,ylim=c(-40,40) )
  points(c(min(fitted.values),max(fitted.values)), c(0,0), type = "l")})

hist(results2$residuals, main="")

qqnorm(results2$residuals, ylab="Residuals", main="")
qqline(results2$residuals) 



```



逻辑回归



```R
###############################################################
# This code covers the code presented in 
# Section 6.2 Logistic Regression
###############################################################

###############################################################
# Section 6.2.3 Diagnostics
###############################################################

churn_input = as.data.frame(  read.csv("c:/data/churn.csv")   )
head(churn_input)

sum(churn_input$Churned)

Churn_logistic1 <- glm (Churned~Age + Married + Cust_years + Churned_contacts, 
                        data=churn_input, family=binomial(link="logit"))
summary(Churn_logistic1)

Churn_logistic2 <- glm (Churned~Age + Married +  Churned_contacts,
                        data=churn_input, family=binomial(link="logit"))
summary(Churn_logistic2)

Churn_logistic3 <- glm (Churned~Age + Churned_contacts, 
                        data=churn_input, family=binomial(link="logit"))
summary(Churn_logistic3)

# Deviance and the Log Likelihood Ratio Test

# Using the residual deviances from Churn_logistics2 and Churn_logistic3
# determine the signficance of the computed test statistic
summary(Churn_logistic2)
pchisq(.9 , 1, lower=FALSE) 

# Receiver Operating Characteristic (ROC) Curve

install.packages("ROCR")     #install, if necessary
library(ROCR)

pred = predict(Churn_logistic3, type="response")
predObj = prediction(pred, churn_input$Churned )

rocObj = performance(predObj, measure="tpr", x.measure="fpr")
aucObj = performance(predObj, measure="auc")  

plot(rocObj, main = paste("Area under the curve:", round(aucObj@y.values[[1]] ,4))) 

# extract the alpha(threshold), FPR, and TPR values from rocObj
alpha <- round(as.numeric(unlist(rocObj@alpha.values)),4)
fpr <- round(as.numeric(unlist(rocObj@x.values)),4)
tpr <- round(as.numeric(unlist(rocObj@y.values)),4)

# adjust margins and plot TPR and FPR
par(mar = c(5,5,2,5))
plot(alpha,tpr, xlab="Threshold", xlim=c(0,1), ylab="True positive rate", type="l")
par(new="True")
plot(alpha,fpr, xlab="", ylab="", axes=F, xlim=c(0,1), type="l" )
axis(side=4)
mtext(side=4, line=3, "False positive rate")
text(0.18,0.18,"FPR")
text(0.58,0.58,"TPR")

i <- which(round(alpha,2) == .5)
paste("Threshold=" , (alpha[i]) , " TPR=" , tpr[i] , " FPR=" , fpr[i])

i <- which(round(alpha,2) == .15)
paste("Threshold=" , (alpha[i]) , " TPR=" , tpr[i] , " FPR=" , fpr[i])




```


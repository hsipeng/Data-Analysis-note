# R 数据预处理/特征工程

* 修正变量

```r
setwd("~/Documents/Workspace/R")

# 修正变量名
df <- data.frame("Address 1"=character(0), direction=character(0),street=character(0),CrossStreet=character(0),intersection=character(0), Location.1=character(0))

names(df)

names(df) <- tolower(names(df))# convert to all lower case
names(df)

# 根据句点把字符串拆分成子列表
splitnames <- strsplit(names(df), "\\.")

class(splitnames)

splitnames[[6]][1]

# 从每一个元素中都提取子列表的第一个元素
firstelement <- function(x){x[1]}

# sapply 函数
names(df) <- sapply(splitnames, firstelement)

names(df)
```



* 创建变量

```r
airquality$Ozone[1:10]
# 计算范围
ozonRanges <- cut(airquality$Ozone, seq(0, 200, by=25))

ozonRanges[1:10]
class(ozonRanges)
table(ozonRanges, useNA = 'ifany')
airquality$ozoneRanges <- ozoneRanges
head(airquality)

```



* 处理缺失值

```r
 library(e1071)
 iris_missing_data <- iris
 View(iris_missing_data)
 iris_missing_data[5,1] <- NA
 iris_missing_data[7,3] <- NA
 iris_missing_data[10,4] <- NA
 iris_missing_data[1:10,-5]
 irrs_repaired <- impute(iris_missing_data[, 1:4], what="mean")
 irrs_repaired <- data.frame(irrs_repaired)
 irrs_repaired[1:10, -5]

 df <- iris_missing_data
 nrow(df)
# 1. 去除缺失值
iris_trimmed <- df[complete.cases(df[,1:4]),]
# iris_trimmed <- na.omit(df) 同理
nrow(iris_trimmed)

# 2. 去除缺失值
df.has.na <- apply(df, 1, function(x){any(is.na(x))})
 sum(df.has.na)
iris_trimmed <- df[!df.has.na,]

```



* 数值离散

```R
data("iris")
buckets <- 10
maxSepLen <- max(iris$Sepal.Length)
minSepLen <- min(iris$Sepal.Length)
cutPoints <- seq(minSepLen, maxSepLen, by=(maxSepLen - minSepLen)/buckets)
cutPoints
cutSepLen <- cut(iris$Sepal.Length, breaks = cutPoints, include.lowest = TRUE)
newiris <- data.frame(contSepLen=iris$Sepal.Length, discSepLen=cutSepLen)
head(newiris)
```



* 日期处理

```r
 library(lubridate)
 data("lakers")
 data("lakers")
 df <- lakers
 str(df$date)
 playdate <- df$date[1]
 playtime <- df$time[1]
 playdatetime <- paste(playdate, playtime)
 playdatetime <- parse_date_time(playdatetime, "%y-%m-%d %H.%M")
 playdatetime
 class(playdatetime)
 
 # ymd 时间转换函数
 df$date <- ymd(df$date)
 str(df$date)
 class(df$date)
 
 
 df$PlayDateTime <- parse_date_time(paste(df$date, df$time), "%y-%m-%d %H.%M")
 str(df$PlayDateTime)
```



* 特征缩放

```r
head(iris)

scaleiris <- scale(iris[, 1:4])

head(scaleiris)

```



* 降维

```R
cor(iris[,-5])

iris_pca <- prcomp(iris[, -5], scale=T)

summary(iris_pca)

plot(iris_pca)

iris_pca$rotation

predict(iris_pca)[1:2,]

biplot(iris_pca)

```


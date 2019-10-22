# 文本分析/自然语言处理

* RFIDF(词频-逆文档频率)

  > 词频 TF
  > $$
  > TF_1(t,d) = \sum_{i=1}^nf(t,t_i)
  > $$
  > 其中 
  > $$
  > f(t,t^{'}) = \begin{cases} 1& \text{如果 t'} \\ 0& \text{其他情况} \end{cases}
  > $$
  > 文档频率
  > $$
  > DF(t) = \sum_{i=1}^{N}f'(t,d_i)
  > $$
  > 其中
  > $$
  > f'(t,d') = \left\{
  > \begin{aligned}
  > 1& \text{, 如果t}\in\text{d'}\\
  > 0& \text{, 其他情况}\\
  > \end{aligned}
  > \right .
  > $$
  > 逆文档频率
  > $$
  > IDF_1(t) = log{\frac{N}{DF(t)}}
  > $$
  > TF_IDF
  > $$
  > TFIDF(t,d) = TF(t,d) \times IDF(t)
  > $$
  > 

  

  

* 通过主题来分类

  > 狄利克雷分配(LDA)

* 情感分析





### 文本分析步骤

* 句法分析

  > POS(词性标注)、命名实体识别、词形还原、或词干提取等文本预处理技术

* 搜索和检索

* 文本挖掘



### 文本分析基本流程

* 收集原始文本
* 表示文本
* 使用TFIDF来计算文本中每个词的有用性
* 使用主题建模按照主题进行文档分类
* 情感分析
* 获取更好的洞察力



​                                                                                                                    



* R

 主题分析

```R
#######################################################
# section 9.6 Categorizing Documents by Topics
#######################################################

require("ggplot2")
require("reshape2")
require("lda")

# load documents and vocabulary
data(cora.documents)
data(cora.vocab)

theme_set(theme_bw())

# Number of topic clusters to display
K <- 10

# Number of documents to display
N <- 9

result <- lda.collapsed.gibbs.sampler(cora.documents,
                                      K,  ## Num clusters
                                      cora.vocab,
                                      25,  ## Num iterations
                                      0.1,
                                      0.1,
                                      compute.log.likelihood=TRUE) 

# Get the top words in the cluster
top.words <- top.topic.words(result$topics, 5, by.score=TRUE)

# build topic proportions
topic.props <- t(result$document_sums) / colSums(result$document_sums)

document.samples <- sample(1:dim(topic.props)[1], N)
topic.props <- topic.props[document.samples,]

topic.props[is.na(topic.props)] <-  1 / K

colnames(topic.props) <- apply(top.words, 2, paste, collapse=" ")

topic.props.df <- melt(cbind(data.frame(topic.props),
                             document=factor(1:N)),
                       variable.name="topic",
                       id.vars = "document")  

qplot(topic, value*100, fill=topic, stat="identity", 
      ylab="proportion (%)", data=topic.props.df, 
      geom="histogram") + 
  theme(axis.text.x = element_text(angle=0, hjust=1, size=12)) + 
  coord_flip() +
  facet_wrap(~ document, ncol=3)

```



* Python

   情感分析

```python
#######################################################
# section 9.6 Categorizing Documents by Topics
#######################################################

require("ggplot2")
require("reshape2")
require("lda")

# load documents and vocabulary
data(cora.documents)
data(cora.vocab)

theme_set(theme_bw())

# Number of topic clusters to display
K <- 10

# Number of documents to display
N <- 9

result <- lda.collapsed.gibbs.sampler(cora.documents,
                                      K,  ## Num clusters
                                      cora.vocab,
                                      25,  ## Num iterations
                                      0.1,
                                      0.1,
                                      compute.log.likelihood=TRUE) 

# Get the top words in the cluster
top.words <- top.topic.words(result$topics, 5, by.score=TRUE)

# build topic proportions
topic.props <- t(result$document_sums) / colSums(result$document_sums)

document.samples <- sample(1:dim(topic.props)[1], N)
topic.props <- topic.props[document.samples,]

topic.props[is.na(topic.props)] <-  1 / K

colnames(topic.props) <- apply(top.words, 2, paste, collapse=" ")

topic.props.df <- melt(cbind(data.frame(topic.props),
                             document=factor(1:N)),
                       variable.name="topic",
                       id.vars = "document")  

qplot(topic, value*100, fill=topic, stat="identity", 
      ylab="proportion (%)", data=topic.props.df, 
      geom="histogram") + 
  theme(axis.text.x = element_text(angle=0, hjust=1, size=12)) + 
  coord_flip() +
  facet_wrap(~ document, ncol=3)

```


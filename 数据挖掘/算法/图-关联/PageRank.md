# PageRank



## 搞懂Google的PageRank算法



### PageRank 的简化模型



![image-20190909115802322](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190909115802322.png)



一个页面影响力 = 所有入链集合的页面的加权影响力之和

面临两个问题:

1. 等级泄漏(Rank Leak): 一个网页没有出链，就像黑洞一样吸收了其他的网页影响力而不释放， PR 值 为 0。
2. 等级沉没(Rank Sink)：如果一个网页只有出链，没有入链，计算迭代的过程中， PR 值 为 0



### PageRank 的随机浏览模型



![image-20190909145813577](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190909145813577.png)



### PageRank 的社交影响力

### PageRank 给我们带来的启发

### 总结

![image-20190909150200534](/Users/lirawx/Documents/Notes/Learning/数据分析实战45/images/image-20190909150200534.png)




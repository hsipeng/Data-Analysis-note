# K-Means

K-Means 是一种非监督学习，解决的是聚类问题。

K 代表 K 类， Means 代表的是中心

> 可以理解这个算法的本质是确定 K 类的中心点，当你找到了这些中心点，也就完成了聚类



思考三个问题:

1. 如何确定 K 类的中心点
2. 如何将其他点划分到 K 类中
3. 如何区分 K-Means 与 KNN



K-Means 的工作原理

1. 选取 K 个点作为初始的类中心点，这些点一般都是从数据集中随机抽取的。
2. 将每个点分配到最近的类中心点，这样就形成了 K 个类，然后重新计算每个类的中心点
3. 重复第二步，知道类不发生变化，或者也可以设置最大叠戴次数，这样即使类中心点发生变化，但是只要达到最大迭代次数就会结束。





* Python

将微信开屏封面进行分割

```python
import numpy as np
import PIL.Image as image
from sklearn import preprocessing
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.image as mpimg

def load_data(filePath):
    # 读文件
    f = open(filePath, 'rb')
    data = []
    # 得到图像像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到x，y的三个通道值
            R, G, B = img.getpixel((x, y))
            data.append([R, G, B])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height
  
  # 加载图像，得到规范化的结果 img，以及图像尺寸
img, width, height = load_data('/Users/lirawx/Documents/Notes/Learning/数据分析实战45/code/kmeans-master/weixin.jpg')

# 用K-Means对图像进行2聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸矩阵
label = label.reshape([width, height])
# 创建个新图像，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new('L', (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置灰度，类别0灰度为255，类别1灰度为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
pic_mark.save('weixin_mark.jpg', 'JPEG')

# 分割成16个部分
kmeans = KMeans(n_clusters=16)
kmeans.fit(img)
label = kmeans.predict(img)
label = label.reshape([width, height])
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2) # 1,2维调换
images = image.fromarray(label_color)
images.save('weixin_mark_color.jpg')

# 创建个新图像，用来保存图像聚类压缩后的结果
img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        R = kmeans.cluster_centers_[label[x, y], 0]
        G = kmeans.cluster_centers_[label[x, y], 1]
        B = kmeans.cluster_centers_[label[x, y], 2]
        img.putpixel((x, y), (int(R*256)-1, int(G*256)-1, int(B*256)-1))
img.save('weixin_new.jpg')


```



* R

```R
# 数据
# Student, English, Math, Science
# 1,99,96,97
# 2,99,96,97
# 3,98,97,97
# 4,95,100,95
# 5,95,96,96
# 6,96,97,96
# 7,100,96,97
# 8,95,98,98
# 9,98,96,96
# 10,99,99,95

# 聚类
grade_input = as.data.frame(read.csv("./grades_km_input.csv"))

kmdata_orig = as.matrix(grade_input[,c("Student", "English", "Math", "Science")])

kmdata <- kmdata_orig[,2:4]

kmdata[1:10,]
# wss 启发式方案
wss <- numeric(15)

for (k in 1:15) wss[k] <- sum(kmeans(kmdata, centers = k, nstart = 25)$withinss)

plot(1:15, wss, type="b", xlab = "Number of Clusters", ylab = "Within Sum of Squares")

# 选定k 值为 3
km <- kmeans(kmdata, 3, nstart = 25)
km

# ggplot 画图
library("ggplot2")
library("grid")
library("gridExtra")
df = as.data.frame(kmdata_orig[,2:4])
df$cluster = factor(km$cluster)
centers = as.data.frame(km$centers)

g1 = ggplot(data=df, aes(x=English, y=Math, color=cluster)) + 
  geom_point() + theme(legend.position="right")+
  geom_point(data=centers, aes(x=English, y=Math, color=as.factor(c(1,2,3))),size=10, alpha=.3,show.legend=FALSE)

g2 = ggplot(data=df, aes(x=English, y=Science, color=cluster)) + 
  geom_point() +
  geom_point(data=centers, aes(x=English, y=Science, color=as.factor(c(1,2,3))),size=10, alpha=.3,show.legend=FALSE)


g3 = ggplot(data=df, aes(x=Math, y=Science, color=cluster)) + 
  geom_point() +
  geom_point(data=centers, aes(x=Math, y=Science, color=as.factor(c(1,2,3))),size=10, alpha=.3,show.legend=FALSE)

temp = ggplot_gtable(ggplot_build(g1))
# gridExtra 包显示多张图
grid.arrange(arrangeGrob(g1 + theme(legend.position="none"),
                         g2 + theme(legend.position="none"),
                         g3 + theme(legend.position="none"),
                         top="High School Student Cluster Analysis",
                         ncol=1))



```


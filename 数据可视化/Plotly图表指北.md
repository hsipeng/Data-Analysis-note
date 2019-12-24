# Plotly + Cufflinks 简介

The Plotly Python library is an open-source version of the [Plotly](https://plot.ly/) visualization software made by Plotly

[Cufflinks is a wrapper ](https://github.com/santosjorge/cufflinks)around the plotly library specifically for plotting with Pandas dataframes. 





导入依赖包

```python
# plotly standard imports
import plotly.graph_objs as go
import plotly.plotly as py

# Cufflinks wrapper on plotly
import cufflinks

# Data science imports
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 30
```



离线设置

```python
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
```



**单变量分布：直方图和箱线图**

```python
df['claps'].iplot(kind='hist', xTitle='claps', yTitle='count', title='Claps Distribution')
```



绘制叠加的直方图



```python
df[['time_started', 'time_published']].iplot(
    kind='hist',
    linecolor='black',
    bins=24,
    histnorm='percent',
    bargap=0.1,
    opacity=0.8,
    barmode='group',
    xTitle='Time of Day',
    yTitle='(%) of Articles',
  title='Time Started and Time Published')
```



双轴直方图

```python
df2 = df[['views', 'read_time',
          'published_date']].set_index('published_date').resample('M').mean()

df2.iplot(
    kind='bar',
    xTitle='Date',
    secondary_y='read_time',
    secondary_y_title='Average Read Time',
    yTitle='Average Views',
    title='Monthly Averages')
```



箱线图

```python
df[['claps', 'fans']].iplot(secondary_y='fans', secondary_y_title='Fans',
    kind='box', yTitle='Claps', title='Box Plot of Claps and Fans')
```



**散点图**



时间折线图

```python
tds[['claps', 'fans']].iplot(
    mode='lines+markers',
    opacity=0.8,
    size=8,
    symbol=1,
    xTitle='Date',
    yTitle='Fans and Claps',
    title='Fans and Claps over Time')
```



散点图

```python
tds.iplot(
    x='read_time',
    y='read_ratio',
    xTitle='Read Time',
    yTitle='Reading Percent',
    text='title',
    mode='markers',
    title='Reading Percent vs Reading Time')
```



散点图+拟合曲线

```python
tds.sort_values('read_time').iplot(
    x='read_time',
    y='read_ratio',
    xTitle='Read Time',
    yTitle='Reading Percent',
    text='title',
    mode='markers+lines',
    bestfit=True, bestfit_colors=['blue'],
    title='Reading Percent vs Reading Time')
```



散点图+分类

```python
df.iplot(
    x='read_time',
    y='read_ratio',
    categories='publication',
    xTitle='Read Time',
    yTitle='Reading Percent',
    title='Reading Percent vs Read Ratio by Publication')
```



多变量

```python
df.iplot(
    x='word_count',
    y='views',
    categories='publication',
    mode='markers',
    text='title',
    size=8,
    layout=dict(
        xaxis=dict(title='Word Count'),
        yaxis=dict(title='Views'),
        title='Views vs Word Count by Publication'))
```



**进阶图表**



**散点矩阵**

```python
colorscales = [
    'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues',
    'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth',
    'Electric', 'Viridis', 'Cividis'
]
import plotly.figure_factory as ff

figure = ff.create_scatterplotmatrix(
    df[['claps', 'publication', 'views', 'read_ratio', 'word_count']],
    height=1000,
    width=1000,
    text=df['title'],
    diag='histogram',
    index='publication')
iplot(figure)
```



**相关性热力图**



```python
colorscales = [
    'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues',
    'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth',
    'Electric', 'Viridis', 'Cividis'
]

corrs = df.corr()

figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    colorscale='Earth',
    annotation_text=corrs.round(2).values,
    showscale=True, reversescale=True)

figure.layout.margin = dict(l=200, t=200)
figure.layout.height = 800
figure.layout.width = 1000

iplot(figure)
```



饼图

```python
df.groupby(
    'publication', as_index=False)['reads'].count().iplot(
        kind='pie', labels='publication', values='reads', title='Percentage of Reads by Publication')
```

参考



- [data-analysis-plotly](https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/plotly/Plotly%20Whirlwind%20Introduction.ipynb)
- [github - code](https://github.com/WillKoehrsen/Data-Analysis)


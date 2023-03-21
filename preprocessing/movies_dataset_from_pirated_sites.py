import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math

#中文乱码处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#数据摘要
def data_summary(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    for i in range(data.shape[1]): 
        datas = [str(j) for j in data[list(data)[i]]]
        datas = pd.DataFrame(datas)
        print(data.columns[i] + ':')
        print(datas.value_counts()[0:5])  #给出每个可能取值的频数,取频数最高的前5显示

    #获取数据库中数字类型数据,非数字类型数据无法得到五数概况
    data_int = data.drop(data.select_dtypes(include='object'), axis=1)

    print(data_int.describe())  #五数概况

    col_null = data.isnull().sum(axis=0)    #缺失值个数
    print(col_null)

#绘制柱状图
def draw_bar(name, data, before):
    data = data.values
    list = Counter(data).most_common()
    x_list = []
    y_list = []
    s = len(list)
    if len(list) > 15:
        s = 15
    for i in range(s):
        x_list.append(list[i][0])
        y_list.append(list[i][1])
    p2 = plt.bar(x=range(len(x_list)), height=y_list, tick_label=x_list)
    if before == -1:
        plt.title('没有操作前的' + name)
    elif before == 0:
        plt.title(name)
    else:
        plt.title('处理缺失值之后的' + name)        
    plt.xlabel(name)
    plt.ylabel('频数')
    plt.xticks(rotation=90, fontsize=10)
    plt.bar_label(p2)
    plt.show()

#数据可视化
def data_visual(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    draw_bar('appropriate_for', data['appropriate_for'], 0)
    draw_bar('director', data['director'], 0)
    draw_bar('industry', data['industry'], 0)
    draw_bar('language', data['language'], 0)
    draw_bar('posted_date', data['posted_date'], 0)
    draw_bar('release_date', data['release_date'], 0)
    draw_bar('run_time', data['run_time'], 0)
    draw_bar('title', data['title'], 0)
    draw_bar('writer', data['writer'], 0)

    #获取数据库中数字类型数据,非数字类型数据无法绘制直方图盒盒图
    data_int = data.drop(data.select_dtypes(include='object'), axis=1)

    #绘制直方图
    for i in range(data_int.shape[1]):

        plt.hist(x=data_int[list(data_int)[i]], bins=50, edgecolor='black')
        plt.xlabel(list(data_int)[i])
        plt.ylabel('频数')
        plt.title(list(data_int)[i] + '直方图')
        plt.show()


    #绘制盒图
    for i in range(data_int.shape[1]):
        data_int[list(data_int)[i]].plot.box(title = list(data_int)[i] + '盒图')
        plt.grid(linestyle="--")
        plt.show()

#将缺失部分剔除
def filldata_drop(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    #绘制删去缺失值之前的柱状图
    draw_bar('IMDb-rating', data['IMDb-rating'], -1)
    draw_bar('appropriate_for', data['appropriate_for'], -1)
    draw_bar('director', data['director'], -1)
    draw_bar('language', data['language'], -1)
    draw_bar('writer', data['writer'], -1)

    print('删去缺失值之前总数据量：'+ str(data.shape[0]))
    data_drop = data.dropna()   #删去有缺失值的行
    print('删去缺失值之后总数据量：'+ str(data_drop.shape[0]))

    #绘制删去缺失值之后的柱状图
    draw_bar('IMDb-rating', data_drop['IMDb-rating'], 1)
    draw_bar('appropriate_for', data_drop['appropriate_for'], 1)
    draw_bar('director', data_drop['director'], 1)
    draw_bar('language', data_drop['language'], 1)
    draw_bar('writer', data_drop['writer'], 1)

#用最高频率值来填补缺失值
def filldata_fre(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    print(data.describe())  #五数概况

    #绘制修改缺失值之前的柱状图
    draw_bar('IMDb-rating', data['IMDb-rating'], -1)
    draw_bar('appropriate_for', data['appropriate_for'], -1)
    draw_bar('director', data['director'], -1)
    draw_bar('language', data['language'], -1)
    draw_bar('writer', data['writer'], -1)

    #mode函数默认删除缺失值, 故无需判断缺失值是否频数最大
    data.fillna(data.mode().iloc[0], inplace=True)

    print(data.describe())  #五数概况

    #绘制修改缺失值之后的柱状图
    draw_bar('IMDb-rating', data['IMDb-rating'], 1)
    draw_bar('appropriate_for', data['appropriate_for'], 1)
    draw_bar('director', data['director'], 1)
    draw_bar('language', data['language'], 1)
    draw_bar('writer', data['writer'], 1)


#通过属性的相关关系来填补缺失值，通过调查可以发现该数据集中，downloads和views的皮尔森系数很高，但是该数据集中这两个属性基本无缺失数据。IMDb-rating和downloads的皮尔森系数得到的相关性并不强，但是下载数量越大，评分越高，因此还是可以进行简单的填充。
def filldata_corr(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    #print(data.corr(method='pearson'))

    #用众数填充downloads属性
    datas = data[['IMDb-rating', 'downloads']].copy()
    datas['downloads'].fillna(datas['downloads'].mode().iloc[0], inplace=True)
    
    print(datas['IMDb-rating'].describe())  #五数概况
    #绘制修改缺失值之前的柱状图
    draw_bar('IMDb-rating', datas['IMDb-rating'], -1)


    #将无需填充的IMDb-rating和downloads按从小到大排序，方便后面给缺失属性按照downloads的排名赋值
    data_rate = datas['IMDb-rating'].copy()
    data_rate = data_rate.dropna()
    data_rate = data_rate.sort_values(ascending=True).reset_index(drop=True)
    data_download = datas['downloads'].copy()
    data_download = data_download.sort_values(ascending=True).reset_index(drop=True)

    #按照顺序，取IMDb-rating缺失值处downloads的值所处的百分比对应位置的IMDb-rating填充
    for i in range(len(datas)):
        if pd.isna(datas.loc[i, 'IMDb-rating']):
            #获取缺失值对应downloads的值所处的位置
            l1 = list(data_download).index(datas.loc[i, 'downloads'])
            #获取百分比处对应的IMDb-rating索引
            t = round(len(data_rate) * l1 / len(data_download))
            datas.loc[i, 'IMDb-rating'] = data_rate[t]

    #绘制修改缺失值之后的柱状图
    print(datas['IMDb-rating'].describe())  #五数概况
    #绘制修改缺失值之前的柱状图
    draw_bar('IMDb-rating', datas['IMDb-rating'], 1)


#通过数据对象之间的相似性来填补缺失值。首先将数据按照industry进行分组，对于有缺失值的数据对象，直接用分组后属性的众数进行填充，如果众数仍然为空，则用所有数据对应的属性的众数进行填充
def filldata_simi(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=0, thousands=',')

    #用众数填充industry属性
    datas = data[['industry', 'language']].copy()
    datas['industry'].fillna(datas['industry'].mode().iloc[0], inplace=True)

    print(datas.describe())
    draw_bar('language', datas['language'], -1)

    #把数据按照industry分组，对于有缺失的language属性用本组众数填充，若本组众数仍然为空，则用整个数据集的language众数填充
    #获取整个数据集的language属性对应众数
    datas_mode = datas['language'].mode()

    #获取分组后数据的众数,若本组众数仍然为空，则用整个数据集的language众数填充
    data_mode  = datas.groupby('industry').agg(lambda x: x.value_counts().index[0]).reset_index()
    for i in range(len(data_mode)):
        if pd.isna(data_mode.loc[i, 'language']):
            data_mode.loc[i, 'language'] = datas_mode

    #用分组数据替换缺失值
    for i in range(len(datas)):
        if pd.isna(datas.loc[i, 'language']):
            for j in range(len(data_mode)):
                if data_mode.loc[j, 'industry'] == datas.loc[i, 'industry']:
                    datas.loc[i, 'language'] = data_mode.loc[j, 'language']
                    break

    print(datas.describe())
    draw_bar('language', datas['language'], 1)


file_path = '../data/Movies_Dataset_from_Pirated_Sites/movies_dataset.csv'

#data_summary(file_path)      #数据摘要
#data_visual(file_path)       #数据可视化
#filldata_drop(file_path)     #将缺失部分剔除
#filldata_fre(file_path)      #用最高频率值来填补缺失值
#filldata_corr(file_path)     #通过属性的相关关系来填补缺失值
filldata_simi(file_path)     #通过数据对象之间的相似性来填补缺失值
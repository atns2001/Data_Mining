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
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

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

#数据可视化
def data_visual(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

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

#绘制柱状图
def draw_bar(name, data, before):
    data = data.values
    list = Counter(data).most_common()
    x_list = []
    y_list = []
    for i in range(20):
        x_list.append(list[i][0])
        y_list.append(list[i][1])
    p2 = plt.bar(x=range(len(x_list)), height=y_list, tick_label=x_list)
    if before == -1:
        plt.title('没有操作前的' + name)
    else:
        plt.title('处理缺失值之后的' + name)
    plt.xlabel(name)
    plt.ylabel('频数')
    plt.bar_label(p2)
    plt.show()

#将缺失部分剔除
def filldata_drop(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

    #绘制删去缺失值之前的柱状图
    draw_bar('name', data['name'], -1)
    draw_bar('stars_count', data['stars_count'], -1)
    draw_bar('forks_count', data['forks_count'], -1)
    draw_bar('watchers', data['watchers'], -1)
    draw_bar('primary_language', data['primary_language'], -1)
    draw_bar('languages_used', data['languages_used'], -1)
    draw_bar('commit_count', data['commit_count'], -1)
    draw_bar('created_at', data['created_at'], -1)
    draw_bar('licence', data['licence'], -1)

    print('删去缺失值之前总数据量：'+ str(data.shape[0]))
    data_drop = data.dropna()   #删去有缺失值的行
    print('删去缺失值之后总数据量：'+ str(data_drop.shape[0]))

    #绘制删去缺失值之后的柱状图
    draw_bar('name', data_drop['name'], 1)
    draw_bar('stars_count', data_drop['stars_count'], 1)
    draw_bar('forks_count', data_drop['forks_count'], 1)
    draw_bar('watchers', data_drop['watchers'], 1)
    draw_bar('primary_language', data_drop['primary_language'], 1)
    draw_bar('languages_used', data_drop['languages_used'], 1)
    draw_bar('commit_count', data_drop['commit_count'], 1)
    draw_bar('created_at', data_drop['created_at'], 1)
    draw_bar('licence', data_drop['licence'], 1)

#用最高频率值来填补缺失值
def filldata_fre(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

    print(data.describe())  #五数概况

    #绘制修改缺失值之前的柱状图
    draw_bar('name', data['name'], -1)
    draw_bar('primary_language', data['primary_language'], -1)
    draw_bar('languages_used', data['languages_used'], -1)
    draw_bar('commit_count', data['commit_count'], -1)
    draw_bar('licence', data['licence'], -1)

    #mode函数默认删除缺失值, 故无需判断缺失值是否频数最大
    data.fillna(data.mode().iloc[0], inplace=True)

    print(data.describe())  #五数概况

    #绘制修改缺失值之后的柱状图
    draw_bar('name', data['name'], 1)
    draw_bar('primary_language', data['primary_language'], 1)
    draw_bar('languages_used', data['languages_used'], 1)
    draw_bar('commit_count', data['commit_count'], 1)
    draw_bar('licence', data['licence'], 1)


#通过属性的相关关系来填补缺失值，通过调查可以发现该数据集中的数字类型数据无缺失，故无需使用皮尔森相关系数填补。但是通过观察数据集可以发现primary_language和languages_used这两个属性的相关性很高，在无缺失的部分中languages_used的第一个语言基本就是primary_language，故可以先使用众数填充languages_used后，再使用languages_used的第一个语言来填充对应的primary_language缺失值
def filldata_corr(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

    #print(data.corr())

    #获取数据库中数字类型数据,非数字类型数据难以比较皮尔森相关系数
    data_lan_used = data['languages_used'].copy()
    data_pri_lan = data['primary_language'].copy()

    #比较languages_used的第一个语言和primary_language的相似性
    # i = 0
    # j = 0
    # for m in range(data_lan_used.shape[0]):
    #     if (not pd.isna(data_lan_used[m])) and (not pd.isna(data_pri_lan[m])):
    #         list = data_lan_used[m].split('\'')
    #         j = j + 1
    #         if(list[1] == data_pri_lan[m]):
    #             i = i + 1
    # print(i/j)

    print(data_lan_used.describe())  #五数概况
    print(data_pri_lan.describe())  #五数概况

    #绘制修改缺失值之后的柱状图
    draw_bar('languages_used', data_lan_used, -1)
    draw_bar('primary_language', data_pri_lan, -1)

    #使用众数填充languages_used后
    data_lan_used.fillna(data_lan_used.mode().iloc[0], inplace=True)
    #使用languages_used的第一个语言来填充对应的primary_language缺失值
    for m in range(data_lan_used.shape[0]):
        if pd.isna(data_pri_lan[m]):
            temp = data_lan_used[m].split('\'')[1]
            data_pri_lan.iloc[m] = temp

    print(data_lan_used.describe())  #五数概况
    print(data_pri_lan.describe())  #五数概况

    #绘制修改缺失值之后的柱状图
    draw_bar('languages_used', data_lan_used, 1)
    draw_bar('primary_language', data_pri_lan, 1)

#通过数据对象之间的相似性来填补缺失值。首先将数据按照name进行分组，对于有缺失值的数据对象，直接用分组后属性的中位数进行填充，如果中位数仍然为空，则用所有数据对应的属性的中位数进行填充
def filldata_simi(file_path):
    data = pd.read_csv(file_path, engine='python', encoding='utf-8' ,header=0, index_col=False)

    #用众数填充name属性
    datas = data[['name', 'commit_count']].copy()
    datas['name'].fillna(datas['name'].mode().iloc[0], inplace=True)

    print(datas.describe())
    draw_bar('commit_count', datas['commit_count'], -1)

    #把数据按照name分组，对于有缺失的commit_count属性用本组平均数填充，若本组平均数仍然为空，则用整个数据集的commit_count平均数填充
    #获取整个数据集的commit_count属性对应平均数
    datas_mean = datas['commit_count'].mean()

    #获取分组后数据的平均数
    datas['med'] = datas.groupby('name')['commit_count'].transform('mean')
    for i in range(len(datas)):
        if pd.isna(datas.loc[i, 'med']):
            datas.loc[i, 'med'] = datas_mean

    #用分组数据替换缺失值
    for i in range(len(datas)):
        if pd.isna(datas.loc[i, 'commit_count']):
            datas.loc[i, 'commit_count'] = datas.loc[i, 'med']

    print(datas.describe())
    draw_bar('commit_count', datas['commit_count'], 1)


file_path = '../data/GitHub_Dataset/repository_data.csv'

data_summary(file_path)      #数据摘要
data_visual(file_path)       #数据可视化
filldata_drop(file_path)     #将缺失部分剔除
filldata_fre(file_path)      #用最高频率值来填补缺失值
filldata_corr(file_path)     #通过属性的相关关系来填补缺失值
filldata_simi(file_path)     #通过数据对象之间的相似性来填补缺失值
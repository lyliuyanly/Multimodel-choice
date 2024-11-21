# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:55:41 2024

@author: ly
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from factor_analyzer import FactorAnalyzer


# 加载数据
inputLocation ="../Google Mobility/"
# data_mobility = pd.read_csv(inputLocation+'Global_Mobility_Report.csv').iloc[:,:]
# data_mobility_us=data_mobility[data_mobility['country_region_code'] == 'US']

columns=['country_region_code','country_region','sub_region_1','sub_region_2',
'metro_area','iso_3166_2_code','census_fips_code','place_id','date',	
'retail_and_recreation_percent_change_from_baseline',
'grocery_and_pharmacy_percent_change_from_baseline',
'parks_percent_change_from_baseline',
'transit_stations_percent_change_from_baseline',
'workplaces_percent_change_from_baseline	',
'residential_percent_change_from_baseline']

data_cases_us = pd.read_csv(inputLocation+'/covid-19-data-master-cases/us.csv').iloc[:,:]
data_cases_us['cases_new'] = data_cases_us['cases'].diff().fillna(0)
data_cases_us['deaths_new'] = data_cases_us['deaths'].diff().fillna(0)
# 保持第一行的值不变
data_cases_us.loc[0, 'cases_new'] = data_cases_us.loc[0, 'cases']
data_cases_us.loc[0, 'deaths_new'] = data_cases_us.loc[0, 'deaths']



# 加载数据
inputLocation2 ="../NHTS2022/"
# 定义一个函数来减1并处理0变成7的情况
def adjust_day(day):
    return 7 if day == 1 else day - 1


data_nhts = pd.read_csv(inputLocation2+'tripperson_path_data2022_dummy.csv')#.iloc[:9068,:]
data_nhts = pd.read_csv(inputLocation2+'tripperson_data2022_dummy_map0.csv')
data_nhts = pd.read_csv(inputLocation2+"tripperson_data2022_map.csv")
data_nhts = pd.read_csv(inputLocation2+"tripperson_data2022_dummy_map_with_risk_alt5_dummy_merge.csv")


# 假设data_nhts已经被定义并包含DAY列
def data_cases(data_nhts):
    # 应用函数到DAY列,转换之前1是星期7，2是星期1，7是星期6；转换后1是星期1
    data_nhts['DAY'] = data_nhts['TRAVDAY']
    data_nhts['DAY'] = data_nhts['DAY'].apply(adjust_day)
    
    
    # data_nhts_day1=data_nhts[data_nhts['DAY']==1]#   
    # # # 然后进行筛选
    # # data_nhts_day1_work = data_nhts_day1[data_nhts_day1['WHYTO'].str.contains('3', na=False)]

    
    # 将 'TDAYDATE' 列转换为字符串类型
    data_nhts['TDAYDATE'] = data_nhts['TDAYDATE'].astype(str)
    # 从TDAYDATE提取年、月、日和星期几的特征
    data_nhts['year'] = data_nhts['TDAYDATE'].str[:4].astype(int)
    data_nhts['month'] = data_nhts['TDAYDATE'].str[4:].astype(int)
    data_nhts['weekday'] = data_nhts['DAY']
    
    
    
    # 确保date列是日期类型
    data_cases_us['date'] = pd.to_datetime(data_cases_us['date'])
    # 从data_cases_us提取年、月、日和星期几的特征
    data_cases_us['year'] = data_cases_us['date'].dt.year
    data_cases_us['month'] = data_cases_us['date'].dt.month
    data_cases_us['day'] = data_cases_us['date'].dt.day
    data_cases_us['weekday'] = data_cases_us['date'].dt.weekday+1  # 将星期几转换为0-6，其中0表示星期一
    
    
    # 创建两个空列表，用于存储计算结果
    cases_new_list = []
    deaths_new_list = []
    
    #对data_nhts的每一行，取其['year']['month']和['weekday']，在data_cases_us里面对应相同['year']['month']和['weekday']的数据；多少病例和死亡人数
    # 循环遍历 data_nhts 中的每一行
    for index, row in data_nhts.iterrows():
        # 提取当前行的年份、月份和星期几
        year = row['year']
        month = row['month']
        weekday = row['weekday']
        
        # 在 data_cases_us 中找到与当前行匹配的数据
        matching_row = data_cases_us[(data_cases_us['year'] == year) & (data_cases_us['month'] == month) & (data_cases_us['weekday'] == weekday)]
        # print(index,matching_row)
        cases_new=matching_row['cases_new'].sum()
        deaths_new=matching_row['deaths_new'].sum()
    
        # 将计算结果添加到列表中
        cases_new_list.append(cases_new)
        deaths_new_list.append(deaths_new)
    
    # 将列表中的结果添加为新列到 data_nhts 中
    data_nhts['cases_new'] = cases_new_list
    data_nhts['deaths_new'] = deaths_new_list
    
    # 输出结果
    print(data_nhts)
    data_nhts.to_csv(inputLocation2 + 'tripperson_data2022_map_adjustday.csv', encoding="utf-8",index=False)

# data_cases(data_nhts)
# data_nhts_cases=pd.read_csv(inputLocation2+'tripperson_data2022_map_adjustday.csv')
 
# Define parameters for each transportation mode
transportation_parameters = pd.read_excel(inputLocation2+'transportation parameters2.xlsx', sheet_name='Sheet1')

def data_risk_calculate(data_nhts_cases, transportation_parameters):
    # 创建一个空的列表来存储感知风险
    perceived_risks = []

    # 处理数据框中的每一行
    for index, row in data_nhts_cases.iterrows():
        transport_mode = row["TRPTRANS_NEW"]
        travel_time = row["TRVLCMIN"]/60#hour
        
        # 检查交通方式是否在参数表中
        if transport_mode in transportation_parameters['TRPTRANS_NEW'].values:
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]
            print( transport_params )
            print(transport_mode)
            #第一种提取
            I = transportation_parameters.loc[transport_mode-1,'Ii']
            print(I)
            # 第二种提取各参数
            weight=transport_params['weight']
            I = transport_params['Ii']
            print(I)
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Vi = transport_params['Vi']
            Qi = transport_params['Qi']
            ni = transport_params['ni']
            alphai = transport_params['Alphai']

            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            perceived_risks.append(Pi)
        else:
            perceived_risks.append(None)  # 如果交通方式不在参数表中，标记为 None
    
    # 将感知风险添加到原始数据框中
    data_nhts_cases["Perceived_Risk"] = perceived_risks
    
    # 保存更新后的数据框到新的 CSV 文件
    data_nhts_cases.to_csv(inputLocation2 + 'tripperson_data2022_dummy_map_with_risk.csv', index=False)

    print("感知风险计算完成，并保存到 'tripperson_data2022_dummy_map_with_risk.csv' 文件中。")

def data_risk_calculate2(data_nhts_cases, transportation_parameters):
    # 遍历数据框中的每一行
    for index, row in data_nhts_cases.iterrows():
        travel_time = row["TRVLCMIN"] / 60  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {}

        # 针对1到9的交通方式计算感知风险
        for transport_mode in range(1, 10):
            # print(transport_mode)
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]

            # 提取各参数
            I = transport_params['Ii']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']

            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            perceived_risks[f"Perceived_Risk_{transport_mode}"] = Pi  # 添加到字典中
        # print(perceived_risks)
        # 将所有感知风险值添加到原始数据框的对应行
        for key, value in perceived_risks.items():
            data_nhts_cases.loc[index, key] = value

    # 保存更新后的数据框到新的 CSV 文件
    data_nhts_cases.to_csv(inputLocation2 + 'tripperson_data2022_dummy_map_with_risk_alt.csv', index=False)

    print("感知风险计算完成，并保存到 'tripperson_data2022_dummy_map_with_risk_alt.csv' 文件中。")

# 执行感知风险计算
# data_risk_calculate(data_nhts_cases, transportation_parameters)

# 执行感知风险计算,9个交通方式，9列感知风险
# data_risk_calculate2(data_nhts_cases, transportation_parameters)


    

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def data_risk_calculate3(data_nhts_cases, transportation_parameters, output_file):
    # 创建一个新的数据框来存储所有的时间和对应的风险
    data_time_risk = pd.DataFrame()
    attributes_pr =  ['Perceived_Risk_1', 'Perceived_Risk_2', 'Perceived_Risk_3',
                 'Perceived_Risk_4', 'Perceived_Risk_5', 'Perceived_Risk_6',
                 'Perceived_Risk_7', 'Perceived_Risk_8', 'Perceived_Risk_9']
    # 定义交通方式的具体名称
    transport_names = [
        "小汽车", "面包车", "SUV", "皮卡", "公交",
        "地铁", "打车", "骑行", "步行"
    ]

    # 遍历120分钟内，每1分钟为一个步长
    for tt in range(1, 121, 1):
        travel_time = tt /2  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {"Travel_Time": tt}  # 包括旅行时间
        
        # 针对1到9的交通方式计算感知风险
        for transport_mode in range(1, 10):
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]
        
            # 提取各参数
            I = transport_params['Ii']
            # Ri =transport_params['Ri']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']
            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            # 按顺序添加到字典中
            perceived_risks[f"Perceived_Risk_{transport_names[transport_mode - 1]}"] = Pi  
        
        # 将风险字典转换为数据框的行并附加到结果数据框
        data_time_risk = data_time_risk.append(perceived_risks, ignore_index=True)
    
    # 保存更新后的数据框到新的 CSV 文件
    data_time_risk.to_csv(inputLocation2 + output_file, index=False)
    print(f"感知风险计算完成，并保存到 '{output_file}' 文件中。")

    # 绘制每种交通方式的感知风险随时间的变化曲线
    plt.figure(figsize=(12, 8))
    # 设置字体，确保支持中文
    font_path = 'C:\Windows\Fonts\SimHei.ttf'  # 黑体字体路径，需要确保这个字体文件在你的工作目录或路径正确
    font_prop = FontProperties(fname=font_path)
    for transport_name in transport_names:
        plt.plot(
            data_time_risk['Travel_Time'],
            data_time_risk[f'Perceived_Risk_{transport_name}'],
            label=transport_name,
            linewidth=2  # 设置线条宽度
        )
    
    plt.xlabel('旅行时间 (分钟)', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.ylabel('感知风险', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.title('不同交通方式的感知风险随时间的变化', fontproperties=font_prop, fontsize=20)  # 设置字体大小
    plt.legend(loc='upper left', prop=font_prop, fontsize=16)  # 设置图例字体大小
    plt.grid(True)
    plt.savefig('perceived_risk_plot.png')
    plt.show()

# 调用函数
# data_risk_calculate3(data_nhts_cases, transportation_parameters, 'time_risk.csv')


def data_risk_calculate3_2(data_nhts_cases, transportation_parameters, output_file):
    # 创建一个新的数据框来存储所有的时间和对应的风险
    data_time_risk = pd.DataFrame()
    attributes_pr = ['Perceived_Risk_1',  'Perceived_Risk_3',
                     'Perceived_Risk_5',  'Perceived_Risk_8']
    # 定义交通方式的具体名称
    transport_names = [ "Car", "SUV", "Transit",  "Slow"]

    # 遍历120分钟内，每1分钟为一个步长
    for tt in range(1, 121, 1):
        travel_time = tt /2  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {"Travel_Time": tt}  # 包括旅行时间
        iii=0
        # 针对1到9的交通方式计算感知风险
        for transport_mode in [1,3,5,8]:
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]
        
            # 提取各参数
            I = transport_params['Ii']
            # Ri =transport_params['Ri']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']
            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            # 按顺序添加到字典中
            perceived_risks[f"Perceived_Risk_{transport_names[iii]}"] = Pi  
            iii=iii+1
        
        # 将风险字典转换为数据框的行并附加到结果数据框
        data_time_risk = data_time_risk.append(perceived_risks, ignore_index=True)
    
    # 保存更新后的数据框到新的 CSV 文件
    data_time_risk.to_csv(inputLocation2 + output_file, index=False)
    print(f"感知风险计算完成，并保存到 '{output_file}' 文件中。")

    # 绘制每种交通方式的感知风险随时间的变化曲线
    plt.figure(figsize=(12, 8))
    # 设置字体，确保支持中文
    font_path = 'C:\Windows\Fonts\SimHei.ttf'  # 黑体字体路径，需要确保这个字体文件在你的工作目录或路径正确
    font_prop = FontProperties(fname=font_path)
    for transport_name in transport_names:
        plt.plot(
            data_time_risk['Travel_Time'],
            data_time_risk[f'Perceived_Risk_{transport_name}'],
            label=transport_name,
            linewidth=2  # 设置线条宽度
        )
    
    plt.xlabel('旅行时间 (分钟)', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.ylabel('感知风险', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.title('不同交通方式的感知风险随时间的变化', fontproperties=font_prop, fontsize=20)  # 设置字体大小
    plt.legend(loc='upper left', prop=font_prop, fontsize=16)  # 设置图例字体大小
    plt.grid(True)
    plt.savefig('perceived_risk_plot.png')
    plt.show()

# 调用函数
# data_risk_calculate3_2(data_nhts_cases, transportation_parameters, 'time_risk2.csv')


def data_risk_calculate3_3(data_nhts_cases, transportation_parameters, output_file):
    # 创建一个新的数据框来存储所有的时间和对应的风险
    data_time_risk = pd.DataFrame()
    attributes_pr = ['Perceived_Risk_1',  'Perceived_Risk_5','Perceived_Risk_6','Perceived_Risk_7',
                     'Perceived_Risk_8']
    # 定义交通方式的具体名称
    transport_names = [ "Car", "Ground PT", "Rail PT","Ride Sharing" ,"Slow"]

    # 遍历120分钟内，每1分钟为一个步长
    for tt in range(1, 121, 1):
        travel_time = tt /2  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {"Travel_Time": tt}  # 包括旅行时间
        iii=0
        # 针对1到9的交通方式计算感知风险
        for transport_mode in [1,5,6,7,8]:
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]
        
            # 提取各参数
            I = transport_params['Ii']
            # Ri =transport_params['Ri']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']
            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            # 按顺序添加到字典中
            perceived_risks[f"Perceived_Risk_{transport_names[iii]}"] = Pi  
            iii=iii+1
            
        
        # 将风险字典转换为数据框的行并附加到结果数据框
        data_time_risk = data_time_risk.append(perceived_risks, ignore_index=True)
    
    # 保存更新后的数据框到新的 CSV 文件
    data_time_risk.to_csv(inputLocation2 + output_file, index=False)
    print(f"感知风险计算完成，并保存到 '{output_file}' 文件中。")

    # 绘制每种交通方式的感知风险随时间的变化曲线
    plt.figure(figsize=(12, 8))
    # 设置字体，确保支持中文
    font_path = 'C:\Windows\Fonts\SimHei.ttf'  # 黑体字体路径，需要确保这个字体文件在你的工作目录或路径正确
    font_prop = FontProperties(fname=font_path)
    for transport_name in transport_names:
        plt.plot(
            data_time_risk['Travel_Time'],
            data_time_risk[f'Perceived_Risk_{transport_name}'],
            label=transport_name,
            linewidth=2  # 设置线条宽度
        )
    
    plt.xlabel('旅行时间 (分钟)', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.ylabel('感知风险', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.title('不同交通方式的感知风险随时间的变化', fontproperties=font_prop, fontsize=20)  # 设置字体大小
    plt.legend(loc='upper left', prop=font_prop, fontsize=16)  # 设置图例字体大小
    plt.grid(True)
    plt.savefig('perceived_risk_plot.png')
    plt.show()

# 调用函数
# data_risk_calculate3_3(data_nhts_cases, transportation_parameters, 'time_risk3.csv')

def data_risk_calculate3_4(data_nhts_cases, transportation_parameters, output_file):
    # 创建一个新的数据框来存储所有的时间和对应的风险
    data_time_risk = pd.DataFrame()
    attributes_pr = ['Perceived_Risk_1',  'Perceived_Risk_5','Perceived_Risk_6', 'Perceived_Risk_8']
    # 定义交通方式的具体名称
    transport_names = [ "Car", "Ground PT", "Rail PT","Slow"]

    # 遍历120分钟内，每1分钟为一个步长
    for tt in range(1, 121, 1):
        travel_time = tt /2  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {"Travel_Time": tt}  # 包括旅行时间
        iii=0
        # 针对1到9的交通方式计算感知风险
        for transport_mode in [1,5,6,8]:
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]
        
            # 提取各参数
            I = transport_params['Ii']
            # Ri =transport_params['Ri']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']
            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            # 按顺序添加到字典中
            perceived_risks[f"Perceived_Risk_{transport_names[iii]}"] = Pi  
            iii=iii+1
            
        
        # 将风险字典转换为数据框的行并附加到结果数据框
        data_time_risk = data_time_risk.append(perceived_risks, ignore_index=True)
    
    # 保存更新后的数据框到新的 CSV 文件
    data_time_risk.to_csv(inputLocation2 + output_file, index=False)
    print(f"感知风险计算完成，并保存到 '{output_file}' 文件中。")

    # 绘制每种交通方式的感知风险随时间的变化曲线
    plt.figure(figsize=(12, 8))
    # 设置字体，确保支持中文
    font_path = 'C:\Windows\Fonts\SimHei.ttf'  # 黑体字体路径，需要确保这个字体文件在你的工作目录或路径正确
    font_prop = FontProperties(fname=font_path)
    for transport_name in transport_names:
        plt.plot(
            data_time_risk['Travel_Time'],
            data_time_risk[f'Perceived_Risk_{transport_name}'],
            label=transport_name,
            linewidth=2  # 设置线条宽度
        )
    
    plt.xlabel('旅行时间 (分钟)', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.ylabel('交通易感度', fontproperties=font_prop, fontsize=16)  # 设置字体大小
    plt.title('不同交通方式的交通易感度随时间的变化', fontproperties=font_prop, fontsize=20)  # 设置字体大小
    plt.legend(loc='upper left', prop=font_prop, fontsize=16)  # 设置图例字体大小
    plt.grid(True)
    plt.savefig('perceived_risk_plot.png')
    plt.show()

# 调用函数
# data_risk_calculate3_4(data_nhts_cases, transportation_parameters, 'time_risk4.csv')



def data_risk_calculate4(data_nhts_cases, transportation_parameters):
    # 遍历数据框中的每一行
    for index, row in data_nhts_cases.iterrows():
        travel_time = row["TRVLCMIN"] /60  # 将旅行时间转换为小时
        
        # 初始化一个字典来存储所有交通方式的感知风险
        perceived_risks = {}

        # 针对1到9的交通方式计算感知风险
        for transport_mode in range(1, 10):
            # print(transport_mode)
            # 提取对应交通方式的参数
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]

            # 提取各参数
            I = transport_params['Ii']
            # Ri =transport_params['Ri']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']
        
            # 计算风险的指数项
            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)
            # 计算感知风险 Pi
            Pi = 1 - np.exp(exponent_term)
            perceived_risks[f"Perceived_Risk_{transport_mode}"] = Pi  # 添加到字典中
        # print(perceived_risks)
        # 将所有感知风险值添加到原始数据框的对应行
        for key, value in perceived_risks.items():
            data_nhts_cases.loc[index, key] = value

    # 保存更新后的数据框到新的 CSV 文件
    data_nhts_cases.to_csv(inputLocation2 + 'tripperson_data2022_dummy_map_with_risk_alt7_dummy_merge.csv', index=False)

    print("感知风险计算完成，并保存到 'tripperson_data2022_dummy_map_with_risk_alt5_dummy_merge.csv' 文件中。")


# 执行感知风险计算,9个交通方式，9列感知风险
data_risk_calculate4(data_nhts, transportation_parameters)

    
    
    
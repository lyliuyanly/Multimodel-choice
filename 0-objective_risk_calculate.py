# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:19:39 2025

@author: HUAWEI
"""
import pandas as pd
import numpy as np

def data_risk_calculate(data_nhts_cases, transportation_parameters,time_divide):
    perceived_risks = []
    attributes_pr = ['Perceived_Risk_1', 'Perceived_Risk_2', 'Perceived_Risk_3',
                      'Perceived_Risk_4', 'Perceived_Risk_5', 'Perceived_Risk_6',
                      'Perceived_Risk_7', 'Perceived_Risk_8', 'Perceived_Risk_9']

    data_nhts_cases= data_nhts_cases.drop(columns= attributes_pr)

    for index, row in data_nhts_cases.iterrows():
        travel_time = row["TRVLCMIN"] / time_divide  # 将旅行时间转换为小时
        
        perceived_risks = {}

        for transport_mode in range(1, 10):
            # print(transport_mode)
            transport_params = transportation_parameters[transportation_parameters['TRPTRANS_NEW'] == transport_mode].iloc[0]

            I = transport_params['Ii']
            Nai = transport_params['Nai']
            Nbi = transport_params['Nbi']
            q = transport_params['q']
            p = transport_params['p']
            thetai = transport_params['Thetai']
            Qi = transport_params['Qi']
            alphai = transport_params['Alphai']

            exponent_term = - (I * (Nai / Nbi) * q * p * thetai * travel_time) / (Qi * alphai)

            Pi = 1 - np.exp(exponent_term)
            perceived_risks[f"Perceived_Risk_{transport_mode}"] =Pi#travel_time#*transport_params['TRPTRANS_NEW']#Pi  # 添加到字典中
        # print(perceived_risks)
        for key, value in perceived_risks.items():
            data_nhts_cases.loc[index, key] = value
    return data_nhts_cases
import pandas as pd
import numpy as np

# hk=pd.read_csv('hk.csv')
# HuToHKAndShenToHK=pd.read_csv('HuToHKAndShenToHK.csv')
#分离沪港通和深港通的数据
# HuToHK=HuToHKAndShenToHK[HuToHKAndShenToHK.MarketLinkCode=="HKEXtoSSE"]
# ShenToHK=HuToHKAndShenToHK[HuToHKAndShenToHK.MarketLinkCode=="HKEXtoSZSE"]
# HuToHK.to_csv('HuToHK.csv',encoding='utf_8_sig')
# ShenToHK.to_csv('ShenToHK.csv',encoding='utf_8_sig')
#分离沪市和深市的数据
# Hu=HuAndShen[(HuAndShen.Markettype==1) | (HuAndShen.Markettype==2) | (HuAndShen.Markettype==32)]
# Shen=HuAndShen[(HuAndShen.Markettype==4) | (HuAndShen.Markettype==8) | (HuAndShen.Markettype==16)]
# Hu.to_csv("Hu.csv",encoding='utf_8_sig')
# Shen.to_csv("Shen.csv",encoding='utf_8_sig')
# HUdf3.to_csv("HuToHK.csv",encoding='utf_8_sig')
# Shendf3.to_csv("ShenToHK.csv",encoding='utf_8_sig')
# hk.to_csv("HK.csv",encoding='utf_8_sig')

# li=[]
# for row_index, row in .iterrows():
#     print(row_index, row, sep='\n')

# NameSet={}
# for row_index, row in HuToHK.iterrows():
#     NameSet.add(row["Symbol"])
#分离沪港通和深港通的信息数据
# HuToHKAndShenToHKInfo=pd.read_csv('HuToHKAndShenToHKInfo.csv')
# HuToHKInfo=HuToHKAndShenToHKInfo[(HuToHKAndShenToHKInfo.MarketLinkCode=='HKEXtoSSE') & (HuToHKAndShenToHKInfo.WhetherAandH=='Y')]
# ShenToHKInfo=HuToHKAndShenToHKInfo[(HuToHKAndShenToHKInfo.MarketLinkCode=='HKEXtoSZSE') & (HuToHKAndShenToHKInfo.WhetherAandH=='Y')]
# HuToHKInfo.to_csv('HuToHKInfo.csv',encoding='utf_8_sig')
# ShenToHKInfo.to_csv('ShenToHKInfo.csv',encoding='utf_8_sig')
#获得名称的集合
# HuToHKInfo=pd.read_csv('HuToHKInfo.csv')
# NameSet={}
# for row_index, row in HuToHKInfo.iterrows():
#     NameSet.add(row["Symbol"])

# HuToHK=pd.read_csv('HuToHK.csv',encoding='utf_8_sig')
# HuToHKInfo=pd.read_csv('HuToHKInfo.csv',encoding='utf_8_sig')
# ShenToHK=pd.read_csv('ShenToHK.csv',encoding='utf_8_sig')
# ShenToHKInfo=pd.read_csv('ShenToHKInfo.csv',encoding='utf_8_sig')
# HKInfo=pd.read_csv('HKInfo.csv',encoding='utf_8_sig')
# AInfo=pd.read_csv('AInfo.csv',encoding='utf_8_sig')
#将同时在两地的沪港通数据筛出
# li=[]
# for row_index, row in HuToHKInfo.iterrows():
#     a=row["Symbol"]
#     temp=AInfo[AInfo.Stkcd==a]
#     li.append(temp)
# df=pd.concat(li)
# df.to_csv('AInfoBothInHuToHK.csv',encoding='utf_8_sig')
# del df
#将同时在两地的A股数据筛出
# AInfoHuToHK=pd.read_csv('AInfoHuToHK.csv',encoding='utf_8_sig')
# AInfoShenToHK=pd.read_csv('AInfoShenToHK.csv',encoding='utf_8_sig')
# AStock=pd.read_csv('AStock.csv')
# li=[]
# for row_index, row in AInfoHuToHK.iterrows():
#     a=row["Stkcd"]
#     temp=AStock[AStock.Stkcd==a]
#     li.append(temp)
# df=pd.concat(li)
# df.to_csv('ABothInHuToHK.csv',encoding='utf_8_sig')
# del df
# li=[]
# for row_index, row in AInfoShenToHK.iterrows():
#     a=row["Stkcd"]
#     temp=AStock[AStock.Stkcd==a]
#     li.append(temp)
# df=pd.concat(li)
# df.to_csv('ABothInShenToHK.csv',encoding='utf_8_sig')
# del df
#将同时在两地的港股数据筛出
# HKInfoHuToHK=pd.read_csv('HKInfoHuToHK.csv',encoding='utf_8_sig')
# HKInfoShenToHK=pd.read_csv('HKInfoShenToHK.csv',encoding='utf_8_sig')
# HKStock=pd.read_csv('HKStock.csv')
# li=[]
# for row_index, row in HKInfoHuToHK.iterrows():
#     a=row["Symbol"]
#     temp=HKStock[HKStock.Symbol==a]
#     li.append(temp)
# df=pd.concat(li)
# df.to_csv('HKBothInHuToHK.csv',encoding='utf_8_sig')
# del df
# li=[]
# for row_index, row in HKInfoShenToHK.iterrows():
#     a=row["Symbol"]
#     temp=HKStock[HKStock.Symbol==a]
#     li.append(temp)
# df=pd.concat(li)
# df.to_csv('HKBothInShenToHK.csv',encoding='utf_8_sig')
# del df

#
#%%
# -*- encoding:utf-8 -*-
# 导库层
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from decimal import Decimal, ROUND_HALF_UP
import math
import os
import pathlib
# 忽略警告
import warnings

warnings.filterwarnings("ignore")
# 数据截止至2023年8月25日，交易费用截止至2023年9月20日
# 读取数据层
AHInfo = pd.read_csv(pathlib.Path('data', 'A+HInfo.csv'), encoding='gbk')
ARHab = pd.read_csv(pathlib.Path('data', 'ARHab.csv'), encoding='gbk')
HAInfo = pd.read_csv(pathlib.Path('data', 'H+AInfo.csv'), encoding='utf_8_sig')
HRHabByRMB = pd.read_csv(pathlib.Path('data', 'HRHabByRMB.csv'), encoding='gbk')
HKDCNY = pd.read_csv(pathlib.Path('data', 'HKDCNY.EX.csv'), encoding='gbk')
AHInfoSH = AHInfo.query('证券代码.str.contains("SH")', engine='python').copy()
AHInfoSZ = AHInfo.query('证券代码.str.contains("SZ")', engine='python').copy()
# 重新设置索引
ARHab = ARHab.set_index(['date'])
HRHabByRMB = HRHabByRMB.set_index(['date'])
HKDCNY = HKDCNY.set_index(['时间'])
HKDCNY = HKDCNY.drop(columns=['代码', '简称'])
# 设置索引格式
ARHab.index = pd.to_datetime(ARHab.index, format='%Y-%m-%d')
HRHabByRMB.index = pd.to_datetime(HRHabByRMB.index, format='%Y-%m-%d')
HKDCNY.index = pd.to_datetime(HKDCNY.index, format='%Y-%m-%d')
# 全局数据清洗
ARHab = ARHab.astype(str)
for index, column in ARHab.items():  # 去除所有的','
    ARHab.loc[:, index] = ARHab.loc[:, index].str.replace(",", "")
ARHab = pd.DataFrame(ARHab, dtype=float)
HRHabByRMB = HRHabByRMB.astype(str)
for index, column in HRHabByRMB.items():  # 去除所有的','
    HRHabByRMB.loc[:, index] = HRHabByRMB.loc[:, index].str.replace(",", "")
HRHabByRMB = pd.DataFrame(HRHabByRMB, dtype=float)
HKDCNY = HKDCNY.astype(str)
for index, column in HKDCNY.items():  # 去除所有的','
    HKDCNY.loc[:, index] = HKDCNY.loc[:, index].str.replace(",", "")
HKDCNY = pd.DataFrame(HKDCNY, dtype=float)
# 生成股票代码
SHStockCode = []
for index, row in AHInfoSH.iterrows():
    SHStockCode.append(row['证券代码'])
SZStockCode = []
for index, row in AHInfoSZ.iterrows():
    SZStockCode.append(row['证券代码'])
SHStockCode = np.array(SHStockCode)
SZStockCode = np.array(SZStockCode)


# 函数层
# 数据选择函数
# 输入A股证券代码，查询A股和H股的对应序列
def DataSelect(AID):
    # AID:A股代码
    HID = AHInfo.loc[AHInfo["证券代码"] == AID]["同公司港股代码"].copy()
    HID = HID.iloc[0]
    data = pd.merge(ARHab[AID].to_frame(), HRHabByRMB[HID].to_frame(), left_index=True, right_index=True,
                    how='outer').copy()
    return data


# 去除空值行
def DataClean(data):
    # data:为要处理的dataframe
    data = data.dropna()
    return data


# 创建文件
def mkdir(path):
    # 输入文件路径
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


# print(HKDCNY)


# 根据时间对数据进行切片
def cutDataByTime(data, StartTime, EndTime, tf, tt):
    '''data:传入的结果数据,StartTime:起始时间,EndTime:结束时间,tt:交易期长度'''
    dataCopy = data.copy()
    dataCopy['name'] = pd.to_datetime(dataCopy['name'])
    StartTime = datetime.datetime.strptime(StartTime, "%Y-%m-%d")
    EndTime = datetime.datetime.strptime(EndTime, "%Y-%m-%d")
    EndTime = EndTime - relativedelta(months=tt) - relativedelta(months=tf)
    dataCopy = dataCopy[(dataCopy['name'] >= StartTime) & (dataCopy['name'] <= EndTime)]
    return dataCopy


# 根据时间切片计算汇总结果
def outputSummaryByTime(IDSet, IsSH, kli):
    '''IDSet:id集合,IsSH:是否沪股，startTime:起始时间,endTime:结束时间'''
    All = 1000000
    CD = 0.05
    mkdir(pathlib.Path('..', 'SummaryByTime'))
    path1 = None
    path2 = None
    path3 = None
    earlyTime = '2002-04-02'
    endTime = '2023-08-25'
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d")
    earlyTime = datetime.datetime.strptime(earlyTime, "%Y-%m-%d")
    preStartTime = ''
    preEndTime = ''
    postStartTime = ''
    postEndTime = ''
    if IsSH == 1:
        path1 = pathlib.Path('..', 'SummaryByTime', 'SH')
        datapath1 = pathlib.Path('..', 'result', 'SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
        # 设置沪港通前后的时间范围
        preStartTime = '2006-02-01'
        preEndTime = '2014-05-01'
        postStartTime = '2015-05-01'
        postEndTime = '2023-08-01'
    else:
        path1 = pathlib.Path('..', 'SummaryByTime', 'SZ')
        datapath1 = pathlib.Path('..', 'result', 'SZ')
        interInterworkingTime = '2016-12-05'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
        # 设置深港通前后的时间范围
        preStartTime = '2010-04-01'
        preEndTime = '2016-06-01'
        postStartTime = '2017-06-01'
        postEndTime = '2023-08-01'
    # kli = np.array([0, 0.25, 0.5, 0.75, 1])
    # [0,0.25,0.5,0.75,1]
    # [0.05,0.10,0.15,0.2]
    # [0.01,0.02,0.03,0.04]
    # [0.002,0.004,0.006,0.008]
    # [0.0004,0.0008,0.0012,0.0016]
    # [0.0001,0.0002,0.003,0.004]
    # [0.000025,0.00005,0.000075]
    # [0.000005,0.00001,0.000015,0.00002]
    # [0.000001,0.000002,0.000003,0.000004]
    # 0.0001,0.005,0.015,0.02
    mkdir(path1)
    AllpreResultsDf = pd.DataFrame(
        columns=['K', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate', 'MonthlyExcessYieldRate',
                 'annualYieldRate(non-compounding)', 'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                 'originalYieldExcessYieldRate',
                 'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)', 'SharpeRate(original)'])
    AllpostResultsDf = pd.DataFrame(
        columns=['K', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate', 'MonthlyExcessYieldRate',
                 'annualYieldRate(non-compounding)', 'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                 'originalYieldExcessYieldRate',
                 'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)', 'SharpeRate(original)'])
    for k in kli:
        print('k=', k)
        path2 = path1.joinpath('k' + str(k))
        datapath2 = datapath1.joinpath('k' + str(k))
        mkdir(path2)
        KpreResultsDf = pd.DataFrame(
            columns=['tf', 'tt', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate',
                     'MonthlyExcessYieldRate', 'annualYieldRate(non-compounding)',
                     'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                     'originalYieldExcessYieldRate',
                     'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)', 'SharpeRate(original)'])
        KpostResultsDf = pd.DataFrame(
            columns=['tf', 'tt', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate',
                     'MonthlyExcessYieldRate', 'annualYieldRate(non-compounding)',
                     'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                     'originalYieldExcessYieldRate',
                     'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)', 'SharpeRate(original)'])
        for tf in range(1, 25):
            for tt in range(1, 13):
                # for tf in range(1, 3):
                #     for tt in range(1, 3):
                # for tf in range(10, 11):
                #     for tt in range(12, 13):
                print('tf=', tf, 'tt=', tt)
                path3 = path2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                datapath3 = datapath2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                mkdir(path3)
                path4 = path3.joinpath('pre')
                datapath4 = datapath3.joinpath('pre')
                mkdir(path4)
                path5 = path3.joinpath('post')
                datapath5 = datapath3.joinpath('post')
                mkdir(path5)
                preResultsDf = pd.DataFrame(
                    columns=['AID', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate',
                             'MonthlyExcessYieldRate', 'annualYieldRate(non-compounding)',
                             'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                             'originalYieldExcessYieldRate',
                             'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)',
                             'SharpeRate(original)'])
                postResultsDf = pd.DataFrame(
                    columns=['AID', 'annualYieldRate', 'annualExcessYieldRate', 'MonthlyYieldRate',
                             'MonthlyExcessYieldRate', 'annualYieldRate(non-compounding)',
                             'annualExcessYieldRate(non-compounding)', 'originalYieldRate',
                             'originalYieldExcessYieldRate',
                             'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'SharpeRate(annual)',
                             'SharpeRate(original)'])
                for AID in IDSet:
                    preDataPath = datapath4.joinpath(
                        AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Result' + '.csv')
                    postDataPath = datapath5.joinpath(
                        AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Result' + '.csv')
                    if preDataPath.exists():
                        preData = pd.read_csv(preDataPath)
                        preData = preData.iloc[:, 1:]
                        preData = cutDataByTime(preData, preStartTime, preEndTime, tf, tt)
                        if preData.size > 0:
                            preData['annualYieldRate'] = (1 + (preData['yieldRate']) / tt) ** (12 / tt) - 1
                            preData['annualExcessYieldRate'] = (1 + (preData['excessYieldRate']) / tt) ** (12 / tt) - 1
                            resultRow = {'AID': AID,
                                         'annualYieldRate': preData['annualYieldRate'].mean(),
                                         'annualExcessYieldRate': preData['annualExcessYieldRate'].mean(),
                                         'MonthlyYieldRate': preData['yieldRate'].mean() / tt,
                                         'MonthlyExcessYieldRate': preData['excessYieldRate'].mean() / tt,
                                         'annualYieldRate(non-compounding)': (preData['yieldRate'].mean() / tt) * 12,
                                         'annualExcessYieldRate(non-compounding)': (preData[
                                                                                        'excessYieldRate'].mean() / tt) * 12,
                                         'originalYieldRate': preData['yieldRate'].mean(),
                                         'originalYieldExcessYieldRate': preData['excessYieldRate'].mean(),
                                         'openingTimes': preData['openingTimes'].mean(),
                                         'closingTimes': preData['closingTimes'].mean(),
                                         'ATF': preData['ATF'].mean(),
                                         'HTF': preData['HTF'].mean(),
                                         'TF': preData['TF'].mean(),
                                         'SharpeRate(annual)': preData['annualYieldRate'].mean() / preData[
                                             'annualExcessYieldRate'].std(),
                                         'SharpeRate(original)': preData['yieldRate'].mean() / preData[
                                             'excessYieldRate'].std()
                                         }
                            preResultsDf = pd.concat([preResultsDf, pd.DataFrame(resultRow, index=[0])],
                                                     ignore_index=True)
                    if postDataPath.exists():
                        postData = pd.read_csv(postDataPath)
                        postData = postData.iloc[:, 1:]
                        postData = cutDataByTime(postData, postStartTime, postEndTime, tf, tt)
                        if postData.size > 0:
                            postData['annualYieldRate'] = (1 + (postData['yieldRate']) / tt) ** (12 / tt) - 1
                            postData['annualExcessYieldRate'] = (1 + (postData['excessYieldRate']) / tt) ** (
                                    12 / tt) - 1
                            resultRow = {'AID': AID,
                                         'annualYieldRate': postData['annualYieldRate'].mean(),
                                         'annualExcessYieldRate': postData['annualExcessYieldRate'].mean(),
                                         'MonthlyYieldRate': postData['yieldRate'].mean() / tt,
                                         'MonthlyExcessYieldRate': postData['excessYieldRate'].mean() / tt,
                                         'annualYieldRate(non-compounding)': (postData['yieldRate'].mean() / tt) * 12,
                                         'annualExcessYieldRate(non-compounding)': (postData[
                                                                                        'excessYieldRate'].mean() / tt) * 12,
                                         'originalYieldRate': postData['yieldRate'].mean(),
                                         'originalYieldExcessYieldRate': postData['excessYieldRate'].mean(),
                                         'openingTimes': postData['openingTimes'].mean(),
                                         'closingTimes': postData['closingTimes'].mean(),
                                         'ATF': postData['ATF'].mean(),
                                         'HTF': postData['HTF'].mean(),
                                         'TF': postData['TF'].mean(),
                                         'SharpeRate(annual)': postData['annualYieldRate'].mean() / postData[
                                             'annualExcessYieldRate'].std(),
                                         'SharpeRate(original)': postData['yieldRate'].mean() / postData[
                                             'excessYieldRate'].std()
                                         }
                            postResultsDf = pd.concat([postResultsDf, pd.DataFrame(resultRow, index=[0])],
                                                      ignore_index=True)
                KPreresultRow = {'tf': tf,
                                 'tt': tt,
                                 'annualYieldRate': preResultsDf['annualYieldRate'].mean(),
                                 'annualExcessYieldRate': preResultsDf['annualExcessYieldRate'].mean(),
                                 'MonthlyYieldRate': preResultsDf['MonthlyYieldRate'].mean(),
                                 'MonthlyExcessYieldRate': preResultsDf['MonthlyExcessYieldRate'].mean(),
                                 'annualYieldRate(non-compounding)': preResultsDf[
                                     'annualYieldRate(non-compounding)'].mean(),
                                 'annualExcessYieldRate(non-compounding)': preResultsDf[
                                     'annualExcessYieldRate(non-compounding)'].mean(),
                                 'originalYieldRate': preResultsDf['originalYieldRate'].mean(),
                                 'originalYieldExcessYieldRate': preResultsDf['originalYieldExcessYieldRate'].mean(),
                                 'openingTimes': preResultsDf['openingTimes'].mean(),
                                 'closingTimes': preResultsDf['closingTimes'].mean(),
                                 'ATF': preResultsDf['ATF'].mean(),
                                 'HTF': preResultsDf['HTF'].mean(),
                                 'TF': preResultsDf['TF'].mean(),
                                 'SharpeRate(annual)': preResultsDf['annualYieldRate'].mean() / preResultsDf[
                                     'annualExcessYieldRate'].std(),
                                 'SharpeRate(original)': preResultsDf['originalYieldRate'].mean() / preResultsDf[
                                     'originalYieldExcessYieldRate'].std()
                                 }
                KpreResultsDf = pd.concat([KpreResultsDf, pd.DataFrame(KPreresultRow, index=[0])],
                                          ignore_index=True)
                KPostresultRow = {'tf': tf,
                                  'tt': tt,
                                  'annualYieldRate': postResultsDf['annualYieldRate'].mean(),
                                  'annualExcessYieldRate': postResultsDf['annualExcessYieldRate'].mean(),
                                  'MonthlyYieldRate': postResultsDf['MonthlyYieldRate'].mean(),
                                  'MonthlyExcessYieldRate': postResultsDf['MonthlyExcessYieldRate'].mean(),
                                  'annualYieldRate(non-compounding)': postResultsDf[
                                      'annualYieldRate(non-compounding)'].mean(),
                                  'annualExcessYieldRate(non-compounding)': postResultsDf[
                                      'annualExcessYieldRate(non-compounding)'].mean(),
                                  'originalYieldRate': postResultsDf['originalYieldRate'].mean(),
                                  'originalYieldExcessYieldRate': postResultsDf['originalYieldExcessYieldRate'].mean(),
                                  'openingTimes': postResultsDf['openingTimes'].mean(),
                                  'closingTimes': postResultsDf['closingTimes'].mean(),
                                  'ATF': postResultsDf['ATF'].mean(),
                                  'HTF': postResultsDf['HTF'].mean(),
                                  'TF': postResultsDf['TF'].mean(),
                                  'SharpeRate(annual)': postResultsDf['annualYieldRate'].mean() / postResultsDf[
                                      'annualExcessYieldRate'].std(),
                                  'SharpeRate(original)': postResultsDf['originalYieldRate'].mean() / postResultsDf[
                                      'originalYieldExcessYieldRate'].std()
                                  }
                KpostResultsDf = pd.concat([KpostResultsDf, pd.DataFrame(KPostresultRow, index=[0])],
                                           ignore_index=True)
                if IsSH:
                    preOtherDataOutPath = path3.joinpath(
                        'SH' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PreSummary.csv')
                    postOtherDataOutPath = path3.joinpath(
                        'SH' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PostSummary.csv')
                else:
                    preOtherDataOutPath = path3.joinpath(
                        'SZ' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PreSummary.csv')
                    postOtherDataOutPath = path3.joinpath(
                        'SZ' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PostSummary.csv')
                preResultsDf.to_csv(preOtherDataOutPath)
                postResultsDf.to_csv(postOtherDataOutPath)
                print('tf' + str(tf) + 'tt' + str(tt) + ' end')
        AllPreresultRow = {'K': k,
                           'annualYieldRate': KpreResultsDf['annualYieldRate'].mean(),
                           'annualExcessYieldRate': KpreResultsDf['annualExcessYieldRate'].mean(),
                           'MonthlyYieldRate': KpreResultsDf['MonthlyYieldRate'].mean(),
                           'MonthlyExcessYieldRate': KpreResultsDf['MonthlyExcessYieldRate'].mean(),
                           'annualYieldRate(non-compounding)': KpreResultsDf[
                               'annualYieldRate(non-compounding)'].mean(),
                           'annualExcessYieldRate(non-compounding)': KpreResultsDf[
                               'annualExcessYieldRate(non-compounding)'].mean(),
                           'originalYieldRate': KpreResultsDf['originalYieldRate'].mean(),
                           'originalYieldExcessYieldRate': KpreResultsDf['originalYieldExcessYieldRate'].mean(),
                           'openingTimes': KpreResultsDf['openingTimes'].mean(),
                           'closingTimes': KpreResultsDf['closingTimes'].mean(),
                           'ATF': KpreResultsDf['ATF'].mean(),
                           'HTF': KpreResultsDf['HTF'].mean(),
                           'TF': KpreResultsDf['TF'].mean(),
                           'SharpeRate(annual)': KpreResultsDf['annualYieldRate'].mean() / KpreResultsDf[
                               'annualExcessYieldRate'].std(),
                           'SharpeRate(original)': KpreResultsDf['originalYieldRate'].mean() / KpreResultsDf[
                               'originalYieldExcessYieldRate'].std()
                           }
        AllpreResultsDf = pd.concat([AllpreResultsDf, pd.DataFrame(AllPreresultRow, index=[0])],
                                    ignore_index=True)
        AllPostresultRow = {'K': k,
                            'annualYieldRate': KpostResultsDf['annualYieldRate'].mean(),
                            'annualExcessYieldRate': KpostResultsDf['annualExcessYieldRate'].mean(),
                            'MonthlyYieldRate': KpostResultsDf['MonthlyYieldRate'].mean(),
                            'MonthlyExcessYieldRate': KpostResultsDf['MonthlyExcessYieldRate'].mean(),
                            'annualYieldRate(non-compounding)': KpostResultsDf[
                                'annualYieldRate(non-compounding)'].mean(),
                            'annualExcessYieldRate(non-compounding)': KpostResultsDf[
                                'annualExcessYieldRate(non-compounding)'].mean(),
                            'originalYieldRate': KpostResultsDf['originalYieldRate'].mean(),
                            'originalYieldExcessYieldRate': KpostResultsDf['originalYieldExcessYieldRate'].mean(),
                            'openingTimes': KpostResultsDf['openingTimes'].mean(),
                            'closingTimes': KpostResultsDf['closingTimes'].mean(),
                            'ATF': KpostResultsDf['ATF'].mean(),
                            'HTF': KpostResultsDf['HTF'].mean(),
                            'TF': KpostResultsDf['TF'].mean(),
                            'SharpeRate(annual)': KpostResultsDf['annualYieldRate'].mean() / KpostResultsDf[
                                'annualExcessYieldRate'].std(),
                            'SharpeRate(original)': KpostResultsDf['originalYieldRate'].mean() / KpostResultsDf[
                                'originalYieldExcessYieldRate'].std()
                            }
        AllpostResultsDf = pd.concat([AllpostResultsDf, pd.DataFrame(AllPostresultRow, index=[0])],
                                     ignore_index=True)
        if IsSH:
            KpreOtherDataOutPath = path2.joinpath('SH' + 'k' + str(k) + 'PreSummary.csv')
            KpostOtherDataOutPath = path2.joinpath('SH' + 'k' + str(k) + 'PostSummary.csv')
        else:
            KpreOtherDataOutPath = path2.joinpath('SZ' + 'k' + str(k) + 'PreSummary.csv')
            KpostOtherDataOutPath = path2.joinpath('SZ' + 'k' + str(k) + 'PostSummary.csv')
        KpreResultsDf.to_csv(KpreOtherDataOutPath)
        KpostResultsDf.to_csv(KpostOtherDataOutPath)
        print('k=', k, ' end')
    if IsSH:
        AllpreOtherDataOutPath = path1.joinpath('SH' + 'PreSummary.csv')
        AllpostOtherDataOutPath = path1.joinpath('SH' + 'PostSummary.csv')
    else:
        AllpreOtherDataOutPath = path1.joinpath('SZ' + 'PreSummary.csv')
        AllpostOtherDataOutPath = path1.joinpath('SZ' + 'PostSummary.csv')
    AllpreResultsDf.to_csv(AllpreOtherDataOutPath)
    AllpostResultsDf.to_csv(AllpostOtherDataOutPath)
    print("*" * 100)
    print("程序结束")
    if IsSH:
        print('SH:pre:(', preStartTime, ',', preEndTime, ') post:(', postStartTime, ',', postEndTime, ')')
    else:
        print('SZ:pre:(', preStartTime, ',', preEndTime, ') post:(', postStartTime, ',', postEndTime, ')')

#%%


# 根据时间切片计算汇总结果
def StatisticKByAID(IDSet, IsSH, kli):
    '''IDSet:id集合,IsSH:是否沪股，startTime:起始时间,endTime:结束时间'''
    mkdir(pathlib.Path('..', 'SummaryKY'))
    path1 = None
    path2 = None
    path3 = None
    earlyTime = '2002-04-02'
    endTime = '2023-08-25'
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d")
    earlyTime = datetime.datetime.strptime(earlyTime, "%Y-%m-%d")
    preStartTime = ''
    preEndTime = ''
    postStartTime = ''
    postEndTime = ''
    if IsSH == 1:
        path1 = pathlib.Path('..', 'SummaryKY', 'SH')
        datapath1 = pathlib.Path('..', 'SummaryByTime', 'SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
        # 设置沪港通前后的时间范围
        preStartTime = '2006-02-01'
        preEndTime = '2014-05-01'
        postStartTime = '2015-05-01'
        postEndTime = '2023-08-01'
    else:
        path1 = pathlib.Path('..', 'SummaryKY', 'SZ')
        datapath1 = pathlib.Path('..', 'SummaryByTime', 'SZ')
        interInterworkingTime = '2016-12-05'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
        # 设置深港通前后的时间范围
        preStartTime = '2010-04-01'
        preEndTime = '2016-06-01'
        postStartTime = '2017-06-01'
        postEndTime = '2023-08-01'
    # kli = np.array([0, 0.25, 0.5, 0.75, 1])
    # [0,0.25,0.5,0.75,1]
    # [0.05,0.10,0.15,0.2]
    # [0.01,0.02,0.03,0.04]
    # [0.002,0.004,0.006,0.008]
    # [0.0004,0.0008,0.0012,0.0016]
    # [0.0001,0.0002,0.003,0.004]
    # [0.000025,0.00005,0.000075]
    # [0.000005,0.00001,0.000015,0.00002]
    # [0.000001,0.000002,0.000003,0.000004]
    # 0.0001,0.005,0.015,0.02
    mkdir(path1)
    preResultsDf = pd.DataFrame(
        columns=['AID', 'MaxK', 'MaxYiled'])
    postResultsDf = pd.DataFrame(
        columns=['AID', 'MaxK', 'MaxYiled'])
    for AID in IDSet:
        PreMaxK = -1
        PreMaxYiled = -100
        PreAvgYield=-101
        PostMaxK = -1
        PostMaxYiled = -100
        PostAvgYield=-101
        PrewrongAID=None
        PostwrongAID=None
        print("AID=", AID)
        for k in kli:
            print('k=', k)
            datapath2 = datapath1.joinpath('k' + str(k))
            Pren = 0
            PreYield = 0
            Postn = 0
            PostYield = 0
            for tf in range(1, 25):
                for tt in range(1, 13):
            # for tf in range(1, 3):
            #     for tt in range(1, 3):
                    # for tf in range(10, 11):
                    #     for tt in range(12, 13):
                    print('tf=', tf, 'tt=', tt)
                    datapath3 = datapath2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                    if IsSH:
                        preDataPath = datapath3.joinpath(
                            'SH' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PreSummary' + '.csv')
                        postDataPath = datapath3.joinpath(
                            'SH' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PostSummary' + '.csv')
                    else:
                        preDataPath = datapath3.joinpath(
                            'SZ' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PreSummary' + '.csv')
                        postDataPath = datapath3.joinpath(
                            'SZ' + 'k' + str(k) + 'tf' + str(tf) + 'tt' + str(tt) + 'PostSummary' + '.csv')
                    if preDataPath.exists():
                        preData = pd.read_csv(preDataPath)
                        preData = preData.iloc[:, 1:].copy()
                        existing_aids = set(preData['AID'])
                        if AID in existing_aids:
                            PreYield = PreYield + preData.loc[
                                preData['AID'] == AID, 'annualYieldRate(non-compounding)'].values[0]
                            Pren = Pren + 1
                    if postDataPath.exists():
                        postData = pd.read_csv(postDataPath)
                        postData = postData.iloc[:, 1:].copy()
                        existing_aids = set(postData['AID'])
                        if AID in existing_aids:
                            PostYield = PostYield + postData.loc[
                                postData['AID'] == AID, 'annualYieldRate(non-compounding)'].values[0]
                            Postn = Postn + 1
                    print('tf' + str(tf) + 'tt' + str(tt) + ' end')
            print('k=', k, ' end')
            if Pren!=0:
                PreAvgYield=PreYield/Pren
                if PreAvgYield>PreMaxYiled:
                    PreMaxYiled=PreAvgYield
                    PreMaxK=k
            else:
                PrewrongAID=AID
            if Postn!=0:
                PostAvgYield=PostYield/Postn
                if PostAvgYield>PostMaxYiled:
                    PostMaxYiled=PostAvgYield
                    PostMaxK=k
            else:
                PostwrongAID=AID
        if PrewrongAID!=AID:
            PreresultRow={'AID':AID,'MaxK':PreMaxK,'MaxYiled':PreAvgYield}
            preResultsDf = pd.concat([preResultsDf, pd.DataFrame(PreresultRow, index=[0])],ignore_index=True)
        if PostwrongAID!=AID:
            PostresultRow={'AID':AID,'MaxK':PostMaxK,'MaxYiled':PostAvgYield}
            postResultsDf = pd.concat([postResultsDf, pd.DataFrame(PostresultRow, index=[0])],ignore_index=True)
        print("AID=", AID,'end')
    if IsSH:
        preResultsDfOutPath = path1.joinpath('SH' + 'PreSummary.csv')
        postResultsDfOutPath = path1.joinpath('SH' + 'PostSummary.csv')
    else:
        preResultsDfOutPath = path1.joinpath('SZ' + 'PreSummary.csv')
        postResultsDfOutPath = path1.joinpath('SZ' + 'PostSummary.csv')
    preResultsDf.to_csv(preResultsDfOutPath)
    postResultsDf.to_csv(postResultsDfOutPath)
    print("*" * 100)
    print("程序结束")
    if IsSH:
        print('SH:pre:(', preStartTime, ',', preEndTime, ') post:(', postStartTime, ',', postEndTime, ')')
    else:
        print('SZ:pre:(', preStartTime, ',', preEndTime, ') post:(', postStartTime, ',', postEndTime, ')')


SHkliTrade = np.array([0.025, 0.075, 0.0125, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
SZkliTrade = np.array([0.01, 0.02, 0.03, 0.04, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4, 1.45])
SHkliCaculate = np.array(
    [0, 0.025, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2,
     1.25, 1.5, 1.75, 2])
SZkliCaculate = np.array(
    [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4,
     1.45, 1.5, 1.75, 2])
# [0,0.25,0.5,0.75,1]
# [0.05,0.1,0.15,0.2]
# [0.01,0.02,0.03,0.04]
# [0.002,0.004,0.006,0.008]
# [0.0004,0.0008,0.0012,0.0016]
# [0.0001,0.0002,0.003,0.004]
# [0.000025,0.00005,0.000075]
# [0.000005,0.00001,0.000015,0.00002]
# [0.000001,0.000002,0.000003,0.000004]
# 0.0001,0.005,0.015,0.02
# outputSummaryByTime(SHStockCode, 1, SHkliCaculate)
# outputSummaryByTime(SZStockCode, 0, SZkliCaculate)
# StatisticKByAID(SZStockCode, 0, np.array([0, 0.25, 0.5, 0.75, 1]))
StatisticKByAID(SHStockCode, 1, SHkliCaculate)
StatisticKByAID(SZStockCode, 0, SZkliCaculate)

# -*- encoding:utf-8 -*-
#导库层
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
#数据截止至2023年8月25日，交易费用截止至2023年9月20日
#读取数据层
AHInfo = pd.read_csv(pathlib.Path('data','A+HInfo.csv'), encoding='gbk')
ARHab = pd.read_csv(pathlib.Path('data','ARHab.csv'), encoding='gbk')
HAInfo = pd.read_csv(pathlib.Path('data','H+AInfo.csv'), encoding='utf_8_sig')
HRHabByRMB = pd.read_csv(pathlib.Path('data','HRHabByRMB.csv'), encoding='gbk')
HKDCNY = pd.read_csv(pathlib.Path('data','HKDCNY.EX.csv'), encoding='gbk')
AHInfoSH = AHInfo.query('证券代码.str.contains("SH")', engine='python').copy()
AHInfoSZ = AHInfo.query('证券代码.str.contains("SZ")', engine='python').copy()
#重新设置索引
ARHab = ARHab.set_index(['date'])
HRHabByRMB = HRHabByRMB.set_index(['date'])
HKDCNY = HKDCNY.set_index(['时间'])
HKDCNY = HKDCNY.drop(columns=['代码', '简称'])
#设置索引格式
ARHab.index = pd.to_datetime(ARHab.index, format='%Y-%m-%d')
HRHabByRMB.index = pd.to_datetime(HRHabByRMB.index, format='%Y-%m-%d')
HKDCNY.index = pd.to_datetime(HKDCNY.index, format='%Y-%m-%d')
#全局数据清洗
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
#生成股票代码
SHStockCode = []
for index, row in AHInfoSH.iterrows():
    SHStockCode.append(row['证券代码'])
SZStockCode = []
for index, row in AHInfoSZ.iterrows():
    SZStockCode.append(row['证券代码'])


#函数层
#数据选择函数
#输入A股证券代码，查询A股和H股的对应序列
def DataSelect(AID):
    #AID:A股代码
    HID = AHInfo.loc[AHInfo["证券代码"] == AID]["同公司港股代码"].copy()
    HID = HID.iloc[0]
    data = pd.merge(ARHab[AID].to_frame(), HRHabByRMB[HID].to_frame(), left_index=True, right_index=True, how='outer').copy()
    return data


#去除空值行
def DataClean(data):
    #data:为要处理的dataframe
    data = data.dropna()
    return data


#线性回归，返回截距与系数
def OLS(x, y):
    #x:解释变量，y:解释变量
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    OLSresults = model.fit()
    if len(OLSresults.params) == 1 and (not 'const' in OLSresults.params.index):
        constant = 0
        slope = OLSresults.params[0]
    elif len(OLSresults.params) == 1 and 'const' in OLSresults.params.index:
        constant = OLSresults.params[0]
        slope = 0
    elif len(OLSresults.params) == 0:
        constant = 0
        slope = 0
    else:
        constant = OLSresults.params[0]
        slope = OLSresults.params[1]
    return [constant, slope]


#判断是否能成为配对，返回是否为配对，1为是，0为否;截距;系数;残差的标准差
def IfPair(data, T0, tf, CD):
    #data为要检验的数据，start为起始时间，精确到日，length为时间跨度，单位为月份，CD为置信度
    data = pd.DataFrame(data, dtype=float)
    T0 = datetime.datetime.strptime(T0, "%Y-%m-%d")
    end = T0 + relativedelta(months=tf)
    temp = data.loc[T0:end].copy()
    temp.iloc[:, 0] = np.log(temp.iloc[:, 0])
    temp.iloc[:, 1] = np.log(temp.iloc[:, 1])
    #进行线性回归
    x = temp.iloc[:, 0].copy()
    y = temp.iloc[:, 1].copy()
    parameter = OLS(x, y)
    re = y - x * parameter[1] - parameter[0]
    std = re.std()
    #ADF检验
    try:
        ADFresultC = adfuller(re, regression='c')
        if ADFresultC[1] < CD:
            return [1, parameter[0], parameter[1], std]
        ADFresultCT = adfuller(re, regression='ct')
        if ADFresultCT[1] < CD:
            return [1, parameter[0], parameter[1], std]
        ADFresultCTT = adfuller(re, regression='ctt')
        if ADFresultCTT[1] < CD:
            return [1, parameter[0], parameter[1], std]
        ADFresultCNC = adfuller(re, regression='n')
        if ADFresultCNC[1] < CD:
            return [1, parameter[0], parameter[1], std]
    except:
        return [0, parameter[0], parameter[1], std]
    return [0, parameter[0], parameter[1], std]


#计算A股交易费用
def CalculateATF(IsS, SNV):
    # IsS:是否是卖方,SNV：股票净值
    # A股佣金比例（ACR）：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.00005；过户费(TransferF)：0.00001；交易所规费(EF)：0.0000687；
    ACR = 0.0003
    SD = 0.00005
    TransferF = 0.00001
    EF = 0.0000687
    if IsS == True:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > 5 else 5
        SDF = Decimal(SNV * SD).quantize(Decimal("0.00"))
        TransferFF = Decimal(SNV * TransferF).quantize(Decimal("0.00"))
        EFF = Decimal(SNV * EF).quantize(Decimal("0.00"))
        ATF = ACRF + SDF + TransferFF + EFF
        return ATF
    else:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > 5 else 5
        TransferFF = Decimal(SNV * TransferF).quantize(Decimal("0.00"))
        EFF = Decimal(SNV * EF).quantize(Decimal("0.00"))
        ATF = ACRF + TransferFF + EFF
        return ATF


#计算H股交易费用
def CalculateHTF(IsS, SNV, ExchangeRate):
    #SNV：股票净值
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.0013，不足一元亦作一元计;
    # 转手纸印花税(SDOTP):5港币，卖方负责缴付；过户费用(TransferF):2.50港币，由买方支付
    ACR = 0.0025
    SFCTL = 0.000027
    FRCTL = 0.0000015
    TF = 0.0000565
    SD = 0.0013
    SDOTP = 5
    TransferF = 2.50
    if IsS == True:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > Decimal(
            100 * ExchangeRate).quantize(Decimal("0.00")) else Decimal(100 * ExchangeRate).quantize(Decimal("0.00"))
        SFCTLF = Decimal(SNV * SFCTL).quantize(Decimal("0.00"))
        FRCTLF = Decimal(SNV * FRCTL).quantize(Decimal("0.00"))
        TFF = Decimal(SNV * TF).quantize(Decimal("0.00"))
        SDF = math.ceil(SNV * SD)
        SDOTPF = Decimal(SDOTP * ExchangeRate).quantize(Decimal("0.00"))
        ATF = ACRF + SFCTLF + FRCTLF + TFF + SDF + SDOTPF
        return ATF
    else:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > Decimal(
            100 * ExchangeRate).quantize(Decimal("0.00")) else Decimal(100 * ExchangeRate).quantize(Decimal("0.00"))
        SFCTLF = Decimal(SNV * SFCTL).quantize(Decimal("0.00"))
        FRCTLF = Decimal(SNV * FRCTL).quantize(Decimal("0.00"))
        TFF = Decimal(SNV * TF).quantize(Decimal("0.00"))
        SDF = math.ceil(SNV * SD)
        TransferFF = Decimal(TransferF * ExchangeRate).quantize(Decimal("0.00"))
        ATF = ACRF + SFCTLF + FRCTLF + TFF + SDF + TransferFF
        return ATF


#分配初始金额函数,返回值为数组:A股数量;港股数量
def AllotmentAmount(All, r, ExchangeRate, PA, PH):
    # All:初始资金;r:比例系数，为在判断是否为配对的函数中的比例系数;DateTime:分配初始金额的日期;date:股票的价格数据
    #A股佣金比例（ACR）：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.00005；过户费(TransferF)：0.00001；交易所规费(EF)：0.0000687；
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.0013，不足一元亦作一元计;
    # 转手纸印花税(SDOTP):5港币，卖方负责缴付；过户费用(TransferF):2.50港币，由买方支付
    #对斜率进行修正
    if r < 0:
        r = r * -1
    T = 2.5 * ExchangeRate  #过户费用
    rA = 0.0003787  #A股交易费用的比例部分
    rH = 0.003885  #H股交易费用的比例部分
    rA1 = 0.0000787  #A股去除佣金部分的比例系数
    ACRA = 0.0003  #A股佣金比例（ACR）
    ACRFA = 5  #A股佣金固定数值
    rH1 = 0.001385  #H股去除佣金部分的比例系数
    ACRFH = 100 * ExchangeRate  #H股佣金固定数值
    ACRH = 0.0025  #H股佣金比例（ACR）
    nH = (All - T) / (PA * r + PA * r * rA + PH + PH * rH)
    nA = r * nH
    # print(nA, nH, r)
    if (PA * nA * ACRA) > 5 and (PH * nH * ACRH) > (100 * ExchangeRate):
        nA = math.floor(nA)
        nH = math.floor(nH)
        return [nA, nH]
    nH = (All - T - ACRFA) / (PA * r + PA * r * rA1 + PH + PH * rH)
    nA = r * nH
    if (PA * nA * ACRA) <= 5 and (PH * nH * ACRH) > (100 * ExchangeRate):
        nA = math.floor(nA)
        nH = math.floor(nH)
        return [nA, nH]
    nH = (All - T - ACRFH) / (PA * r + PA * r * rA + PH + PH * rH1)
    nA = r * nH
    if (PA * nA * ACRA) > 5 and (PH * nH * ACRH) <= (100 * ExchangeRate):
        nA = math.floor(nA)
        nH = math.floor(nH)
        return [nA, nH]
    nH = (All - T - ACRFA - ACRFH) / (PA * r + PA * r * rA1 + PH + PH * rH1)
    nA = r * nH
    if (PA * nA * ACRA) <= 5 and (PH * nH * ACRH) <= (100 * ExchangeRate):
        if nA > 0:
            nA = math.floor(nA)
            nH = math.floor(nH)
            return [nA, nH]
        else:
            return [0, 0]


#购买A股函数，返回这笔资金能购买的A股股票数量
def NumberOfACanBePurchased(All, price):
    #All:资金;DateTime:日期;price:价格
    #A股佣金比例（ACR）：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.00005；过户费(TransferF)：0.00001；交易所规费(EF)：0.0000687；
    rA = 0.0003787  #A股交易费用的比例部分
    rA1 = 0.0000787  #A股去除佣金部分的比例系数
    ACRA = 0.0003  #A股佣金比例（ACR）
    ACRFA = 5  #A股佣金固定数值
    nA = All / ((1 + rA) * price)
    if price * nA * ACRA > 5:
        nA = math.floor(nA)
        return nA
    nA = (All - ACRFA) / ((1 + rA1) * price)
    if price * nA * ACRA <= 5:
        if nA < 0:
            return 0
        nA = math.floor(nA)
        return nA


#购买H股函数，返回这笔资金能购买的H股股票数量
def NumberOfHCanBePurchased(All, price, ExchangeRate):
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.0013，不足一元亦作一元计;
    # 转手纸印花税(SDOTP):5港币，卖方负责缴付；过户费用(TransferF):2.50港币，由买方支付
    T = 2.5 * ExchangeRate  #过户费用
    rH = 0.003885  #H股交易费用的比例部分
    rH1 = 0.001385  #H股去除佣金部分的比例系数
    ACRFH = 100 * ExchangeRate  #H股佣金固定数值
    ACRH = 0.0025  #H股佣金比例（ACR）
    nH = (All - T) / ((1 + rH) * price)
    if price * nH * ACRH > 100 * ExchangeRate:
        nH = math.floor(nH)
        return nH
    nH = (All - T - ACRFH) / ((1 + rH1) * price)
    if price * nH * ACRH <= 100 * ExchangeRate:
        if nH < 0:
            return 0
        nH = math.floor(nH)
        return nH


#交易函数，将模拟交易过程,返回为收益率,开仓次数,平仓次数
def trade(AID, All, data, CD, T0, tf, tt, dc, do, ds):
    # print("*" * 100)
    # print("开始交易", "证券代码:", AID, "开始日期:", T0, "形成期长度:", tf, "交易期长度:", tt)
    # 是否为配对，1为是，0为否;截距;系数;残差的标准差
    # print(data)
    dataCopy = data.copy()
    par = IfPair(dataCopy, T0, tf, CD)
    if par[0] == 0:
        # print("不构成配对，交易结束")
        templi = []
        while len(templi) < tt * 30 + 4:
            templi.append(0)
        return templi
    #dc平仓阈值系数,do为开仓阈值系数，ds为止损阈值系数
    dok = par[3] * do
    dck = par[3] * dc
    dsk = par[3] * ds
    T0 = datetime.datetime.strptime(T0, "%Y-%m-%d")
    T1 = T0 + relativedelta(months=tf)
    T2 = T0 + relativedelta(months=tf + tt)
    TradingPeriodData = data.loc[T1:T2].copy()
    if TradingPeriodData.size == 0:
        templi = []
        while len(templi) < tt * 30 + 4:
            templi.append(0)
        return templi
    openingTimes = 0
    closingTimes = 0
    # A股数量;港股数量
    # ExchangeRate0 = HKDCNY['开盘价(元)'][TradingPeriodData.iloc[0].name]
    ExchangeRate0 = getExchangeRate(TradingPeriodData.iloc[0].name)
    PA0 = TradingPeriodData.iloc[:, 0][TradingPeriodData.iloc[0].name]
    PH0 = TradingPeriodData.iloc[:, 1][TradingPeriodData.iloc[0].name]
    # print(PA0)
    PA0 = float(PA0)
    PH0 = float(PH0)
    par1 = AllotmentAmount(All, par[2], ExchangeRate0, PA0, PH0)
    nA = par1[0]
    nAStatic = nA
    nAStart = par1[0]
    nH = par1[1]
    nHStart = par1[1]
    nHStatic = nH
    cash = Decimal(All).quantize(Decimal("0.00")) - Decimal(nA * PA0).quantize(Decimal("0.00")) - CalculateATF(0,
                                                                                                               nA * PA0) - Decimal(
        nH * PH0).quantize(Decimal("0.00")) - CalculateHTF(0, nH * PH0, ExchangeRate0)
    rePre = None
    IsOpen = False
    redf = []  #用来记录每次模拟交易的残差值
    TradingPeriodData = pd.DataFrame(TradingPeriodData, dtype=float)
    TradingPeriodDataCopy = TradingPeriodData.copy()
    for index, row in TradingPeriodDataCopy.iterrows():

        re = math.log(row[1]) - math.log(row[0]) * par[2] - par[1]
        redf.append(re)
        reAbs = abs(re)
        # ExchangeRate = HKDCNY['开盘价(元)'][index]
        ExchangeRate = getExchangeRate(index)
        # print(re, re * 100 / par[3], dck, dok, dsk)
        if reAbs > dsk:
            if nA != 0:
                nACash = Decimal(nA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nA * row[0])
                nA = 0
                cash = cash + nACash
            if nH != 0:
                nHCash = Decimal(nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nH * row[1], ExchangeRate)
                nH = 0
                cash = cash + nHCash
            nAStaticCash = Decimal(nAStatic * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nAStatic * row[0])
            nHStaticCash = Decimal(nHStatic * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nHStatic * row[1],
                                                                                               ExchangeRate)
            staticCash = nAStaticCash + nHStaticCash
            staticYieldRate = float((staticCash - All) / All)
            yieldRate = float((cash - All) / All)
            excessYieldRate = yieldRate - staticYieldRate
            # print("超出阈值，停止交易")
            while len(redf) < tt * 30:
                redf.append(0)
            redf.append(yieldRate)
            redf.append(openingTimes)
            redf.append(closingTimes)
            redf.append(excessYieldRate)
            return redf
        if rePre is None:
            if reAbs > dok:
                if re > 0:
                    nHCash = Decimal(nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nH * row[1], ExchangeRate)
                    nH = 0
                    cash = cash + nHCash
                    nAP = NumberOfACanBePurchased(float(cash), row[0])
                    cash = cash - Decimal(nAP * row[0]).quantize(Decimal("0.00")) - CalculateATF(0, nAP * row[0])
                    nA = nA + nAP
                if re < 0:
                    nACash = Decimal(nA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nA * row[0])
                    nA = 0
                    cash = cash + nACash
                    nHP = NumberOfHCanBePurchased(float(cash), row[1], ExchangeRate)
                    cash = cash - Decimal(nHP * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, nHP * row[1],
                                                                                                 ExchangeRate)
                    nH = nH + nHP
                # print("开仓")
                IsOpen = True
                openingTimes = openingTimes + 1
            # if reAbs<=dck:
            #     if nA>nAStart:
            #         deltaA=nA-nAStart
            #         nACash=Decimal(deltaA*row[0]).quantize(Decimal("0.00"))-CalculateATF(1,deltaA*row[0])
            #         cash=cash+nACash
            #         nA=nAStart
            #     if nH>nHStart:
            #         deltaH=nH-nHStart
            #         nHCash=Decimal(deltaH*row[1]).quantize(Decimal("0.00"))-CalculateHTF(1,deltaH*row[1],ExchangeRate)
            #         cash=cash+nHCash
            #         nH=nHStart
            #     if nA<nAStart:
            #         deltaA=nAStart-nA
            #         nA=nAStart
            #         nACash=-1*Decimal(deltaA*row[0]).quantize(Decimal("0.00"))-CalculateATF(0,deltaA*row[0])
            #         cash=cash+nACash
            #     if nH<nHStart:
            #         deltaH=nHStart-nH
            #         nH=nHStart
            #         nHCash=-1*Decimal(deltaH*row[1]).quantize(Decimal("0.00"))-CalculateHTF(0,deltaH*row[1],ExchangeRate)
            #         cash=cash+nHCash
            #     print("平仓")
            #     closingTimes=closingTimes+1
        else:
            if (reAbs > dok and (not IsOpen)) or (reAbs > dok and IsOpen and re * rePre < 0):
                if re > 0:
                    nHCash = Decimal(nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nH * row[1], ExchangeRate)
                    nH = 0
                    cash = cash + nHCash
                    nAP = NumberOfACanBePurchased(float(cash), row[0])
                    cash = cash - Decimal(nAP * row[0]).quantize(Decimal("0.00")) - CalculateATF(0, nAP * row[0])
                    nA = nA + nAP
                if re < 0:
                    nACash = Decimal(nA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nA * row[0])
                    nA = 0
                    cash = cash + nACash
                    nHP = NumberOfHCanBePurchased(float(cash), row[1], ExchangeRate)
                    cash = cash - Decimal(nHP * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, nHP * row[1],
                                                                                                 ExchangeRate)
                    nH = nH + nHP
                # print("开仓")
                IsOpen = True
                openingTimes = openingTimes + 1
            if abs(re / par[3]) <= 0.001 and reAbs < dok:
                if nA > nAStart:
                    deltaA = nA - nAStart
                    nACash = Decimal(deltaA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, deltaA * row[0])
                    cash = cash + nACash
                    nA = nAStart
                if nH > nHStart:
                    deltaH = nH - nHStart
                    nHCash = Decimal(deltaH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, deltaH * row[1],
                                                                                               ExchangeRate)
                    cash = cash + nHCash
                    nH = nHStart
                if nA < nAStart:
                    deltaA = nAStart - nA
                    nA = nAStart
                    nACash = -1 * Decimal(deltaA * row[0]).quantize(Decimal("0.00")) - CalculateATF(0, deltaA * row[0])
                    cash = cash + nACash
                if nH < nHStart:
                    deltaH = nHStart - nH
                    nH = nHStart
                    nHCash = -1 * Decimal(deltaH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, deltaH * row[1],
                                                                                                    ExchangeRate)
                    cash = cash + nHCash
                # print("平仓")
                IsOpen = False
                closingTimes = closingTimes + 1
        rePre = re
    # ExchangeRateLast = HKDCNY['开盘价(元)'][TradingPeriodData.tail(1).index][0]
    ExchangeRateLast = getExchangeRate(TradingPeriodData.tail(1).index[0])
    PALast = TradingPeriodData.iloc[:, 0][TradingPeriodData.tail(1).index][0]
    PHLast = TradingPeriodData.iloc[:, 1][TradingPeriodData.tail(1).index][0]
    if nA != 0:
        nACash = Decimal(nA * PALast).quantize(Decimal("0.00")) - CalculateATF(1, nA * PALast)
        nA = 0
        cash = cash + nACash
    if nH != 0:
        nHCash = Decimal(nH * PHLast).quantize(Decimal("0.00")) - CalculateHTF(1, nH * PHLast, ExchangeRateLast)
        nH = 0
        cash = cash + nHCash
    nAStaticCash = Decimal(nAStatic * PALast).quantize(Decimal("0.00")) - CalculateATF(1, nAStatic * PALast)
    nHStaticCash = Decimal(nHStatic * PHLast).quantize(Decimal("0.00")) - CalculateHTF(1, nHStatic * PHLast,
                                                                                       ExchangeRateLast)
    staticCash = nAStaticCash + nHStaticCash
    staticYieldRate = float((staticCash - All) / All)
    yieldRate = float((cash - All) / All)
    excessYieldRate = yieldRate - staticYieldRate
    # print("结束交易")
    while len(redf) < tt * 30:
        redf.append(0)
    redf.append(yieldRate)
    redf.append(openingTimes)
    redf.append(closingTimes)
    redf.append(excessYieldRate)
    return redf


#创建文件
def mkdir(path):
    #输入文件路径
    folder = os.path.exists(path)

    if not folder:  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def getExchangeRate(time):
    # print(time)
    while not (time in HKDCNY.index):
        time = time - relativedelta(days=1)
    ExchangeRate = HKDCNY.loc[:, '开盘价(元)'][time]
    return ExchangeRate


# print(HKDCNY)

#%%

def pariTradeMain(IDSet, IsSH):
    All = 1000000
    CD = 0.05
    mkdir(pathlib.Path('..','result'))
    path1 = None
    path2 = None
    path3 = None
    earlyTime = '2002-04-02'
    endTime = '2023-08-25'
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d")
    earlyTime = datetime.datetime.strptime(earlyTime, "%Y-%m-%d")
    if IsSH == 1:
        path1 = pathlib.Path('..','result','SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    else:
        path1 = pathlib.Path('..','result','SZ')
        interInterworkingTime = '2016-12-05'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    # kli = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5]
    kli = [0.00001,0.001,0.01,0.0125]
    mkdir(path1)
    for k in kli:
        print('k=',k)
        path2 = path1.joinpath('k' + str(k))
        mkdir(path2)
        for tf in range(1, 25):
            for tt in range(1, 13):
        # for tf in range(1, 3):
        #     for tt in range(1, 3):
                # for tf in range(10, 11):
                #     for tt in range(12, 13):
                print('tf=',tf,'tt=',tt)
                path3 = path2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                mkdir(path3)
                path4 = path3.joinpath('pre')
                mkdir(path4)
                path5 = path3.joinpath('post')
                mkdir(path5)
                preOtherDic = {}
                preNameLi = []
                preExcessYieldRateLi = []
                preYieldRateLi = []
                preOpeningTimesLi = []
                preClosingTimesLi = []
                postOtherDic = {}
                postNameLi = []
                postYieldRateLi = []
                postExcessYieldRateLi = []
                postOpeningTimesLi = []
                postClosingTimesLi = []
                preSharpeRateLi = []
                postSharpeRateLi = []
                for AID in IDSet:
                    data = DataSelect(AID)
                    data = DataClean(data)
                    T0Li = []
                    preReDIC = {}
                    postReDIC = {}
                    T0set = set()
                    for i in range(0, data.shape[0]):
                        T0 = data.iloc[i].name
                        T0 = T0.replace(day=1)
                        if not T0.strftime("%Y-%m-%d") in T0set:
                            T0set.add(T0.strftime("%Y-%m-%d"))
                            T0Li.append(T0)
                    preTempYieldRateLi = []
                    preTempOpeningTimesLi = []
                    preTempClosingTimesLi = []
                    preTempExcessYieldRateLi = []
                    postTempYieldRateLi = []
                    postTempOpeningTimesLi = []
                    postTempClosingTimesLi = []
                    postTempExcessYieldRateLi = []
                    try:
                        for T0 in T0Li:
                            if T0 < earlyTime or (T0 + relativedelta(
                                    months=tf + tt) > interInterworkingTime and T0 < interInterworkingTime) or (
                                    T0 > interInterworkingTime and (T0 + relativedelta(months=tf + tt) > endTime)):
                                continue
                            # [yieldRate, openingTimes, closingTimes,redf]
                            reliName = str(T0) + 'tf' + str(tf) + 'tt' + str(tt)
                            T0Copy = T0.strftime("%Y-%m-%d")
                            dataCopy = data.copy()
                            tradeData = trade(AID, All, dataCopy, CD, T0Copy, tf, tt, 0, k, 1.96)
                            if T0 < interInterworkingTime:
                                preReDIC.update({reliName: tradeData})
                            else:
                                postReDIC.update({reliName: tradeData})
                            if T0 < interInterworkingTime:
                                if tradeData[-4] != 0 or tradeData[-3] != 0 or tradeData[-2] != 0 or tradeData[-1] != 0:
                                    preTempYieldRateLi.append(tradeData[-4])
                                    preTempOpeningTimesLi.append(tradeData[-3])
                                    preTempClosingTimesLi.append(tradeData[-2])
                                    preTempExcessYieldRateLi.append(tradeData[-1])
                            else:
                                if tradeData[-4] != 0 or tradeData[-3] != 0 or tradeData[-2] != 0 or tradeData[-1] != 0:
                                    postTempYieldRateLi.append(tradeData[-4])
                                    postTempOpeningTimesLi.append(tradeData[-3])
                                    postTempClosingTimesLi.append(tradeData[-2])
                                    postTempExcessYieldRateLi.append(tradeData[-1])
                    except:
                        print(AID)
                        continue
                    if not len(preTempYieldRateLi) == 0:
                        preNameLi.append(AID)
                        preYieldRateLi.append(np.mean(preTempYieldRateLi))
                        preExcessYieldRateLi.append(np.mean(preTempExcessYieldRateLi))
                        preOpeningTimesLi.append(np.mean(preTempOpeningTimesLi))
                        preClosingTimesLi.append(np.mean(preTempClosingTimesLi))
                        preSharpeRateLi.append(np.mean(preTempExcessYieldRateLi) / np.std(preTempYieldRateLi))
                    if not len(postTempYieldRateLi) == 0:
                        postNameLi.append(AID)
                        postYieldRateLi.append(np.mean(postTempYieldRateLi))
                        postOpeningTimesLi.append(np.mean(postTempOpeningTimesLi))
                        postClosingTimesLi.append(np.mean(postTempClosingTimesLi))
                        postExcessYieldRateLi.append(np.mean(postTempExcessYieldRateLi))
                        postSharpeRateLi.append(np.mean(postTempExcessYieldRateLi) / np.std(postTempYieldRateLi))
                    preReDataOut = pd.DataFrame.from_dict(preReDIC)
                    postReDataOut = pd.DataFrame.from_dict(postReDIC)
                    preReDataOut.to_csv(path4.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + '.csv'))
                    postReDataOut.to_csv(path5.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + '.csv'))
                print('tf' + str(tf) + 'tt' + str(tt) + ' end')
        print('k=', k, ' end')
    print("*" * 100)
    print("程序结束")

# print(HKDCNY)

def outputSummaryAll(IDSet, IsSH):
    All = 1000000
    CD = 0.05
    mkdir(pathlib.Path('..','Summary'))
    path1 = None
    path2 = None
    path3 = None
    earlyTime = '2002-04-02'
    endTime = '2023-08-25'
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d")
    earlyTime = datetime.datetime.strptime(earlyTime, "%Y-%m-%d")
    if IsSH == 1:
        path1 = pathlib.Path('..','Summary','SH')
        datapath1=pathlib.Path('..','result','SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    else:
        path1 = pathlib.Path('..','Summary','SZ')
        datapath1=pathlib.Path('..','result','SZ')
        interInterworkingTime = '2016-12-05'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    kli = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5]
    # kli = [0.025]
    mkdir(path1)
    AllpreOtherDic = {}
    AllpreNameLi = []
    AllpreYieldRateLi = []
    AllpreOpeningTimesLi = []
    AllpreClosingTimesLi = []
    AllpostOtherDic = {}
    AllpostNameLi = []
    AllpostYieldRateLi = []
    AllpostOpeningTimesLi = []
    AllpostClosingTimesLi = []
    AllpreExcessYieldRateLi = []
    AllpostExcessYieldRateLi = []
    AllpreSharpeRateLi = []
    AllpostSharpeRateLi = []
    for k in kli:
        print('k=',k)
        path2 = path1.joinpath('k' + str(k))
        datapath2=datapath1.joinpath('k' + str(k))
        mkdir(path2)
        KpreOtherDic = {}
        KpreNameLi = []
        KpreYieldRateLi = []
        KpreOpeningTimesLi = []
        KpreClosingTimesLi = []
        KpostOtherDic = {}
        KpostNameLi = []
        KpostYieldRateLi = []
        KpostOpeningTimesLi = []
        KpostClosingTimesLi = []
        KpreExcessYieldRateLi = []
        KpostExcessYieldRateLi = []
        KpreSharpeRateLi = []
        KpostSharpeRateLi = []
        for tf in range(1, 25):
            for tt in range(1, 13):
        # for tf in range(1, 3):
        #     for tt in range(1, 3):
                # for tf in range(10, 11):
                #     for tt in range(12, 13):
                print('tf=',tf,'tt=',tt)
                path3 = path2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                datapath3=datapath2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                mkdir(path3)
                path4 = path3.joinpath('pre')
                datapath4=datapath3.joinpath('pre')
                mkdir(path4)
                path5 = path3.joinpath('post')
                datapath5=datapath3.joinpath('post')
                mkdir(path5)
                preOtherDic = {}
                preNameLi = []
                preExcessYieldRateLi = []
                preYieldRateLi = []
                preOpeningTimesLi = []
                preClosingTimesLi = []
                postOtherDic = {}
                postNameLi = []
                postYieldRateLi = []
                postExcessYieldRateLi = []
                postOpeningTimesLi = []
                postClosingTimesLi = []
                preSharpeRateLi = []
                postSharpeRateLi = []
                for AID in IDSet:
                    preDataPath = datapath4.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + '.csv')
                    postDataPath = datapath5.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + '.csv')
                    if preDataPath.exists():
                        preData=pd.read_csv(preDataPath)
                        preData = preData.iloc[:, 1:]
                        if preData.size>0:
                            preNameLi.append(AID)
                            preYieldRateLi.append(preData.iloc[-4].mean())
                            preExcessYieldRateLi.append(preData.iloc[-1].mean())
                            preOpeningTimesLi.append(preData.iloc[-3].mean())
                            preClosingTimesLi.append(preData.iloc[-2].mean())
                            preSharpeRateLi.append(preData.iloc[-1].mean()/ preData.iloc[-4].std())
                    if postDataPath.exists():
                        postData=pd.read_csv(postDataPath)
                        postData = postData.iloc[:, 1:]
                        if postData.size>0:
                            postNameLi.append(AID)
                            postYieldRateLi.append(postData.iloc[-4].mean())
                            postExcessYieldRateLi.append(postData.iloc[-1].mean())
                            postOpeningTimesLi.append(postData.iloc[-3].mean())
                            postClosingTimesLi.append(postData.iloc[-2].mean())
                            postSharpeRateLi.append(postData.iloc[-1].mean()/ postData.iloc[-4].std())
                KpreNameLi.append('tf' + str(tf) + 'tt' + str(tt))
                KpreYieldRateLi.append(np.mean(preYieldRateLi))
                KpreOpeningTimesLi.append(np.mean(preOpeningTimesLi))
                KpreClosingTimesLi.append(np.mean(preClosingTimesLi))
                KpreExcessYieldRateLi.append(np.mean(preExcessYieldRateLi))
                KpreSharpeRateLi.append(np.mean(preExcessYieldRateLi) / np.std(preYieldRateLi))
                KpostNameLi.append('tf' + str(tf) + 'tt' + str(tt))
                KpostYieldRateLi.append(np.mean(postYieldRateLi))
                KpostOpeningTimesLi.append(np.mean(postOpeningTimesLi))
                KpostClosingTimesLi.append(np.mean(postClosingTimesLi))
                KpostExcessYieldRateLi.append(np.mean(postExcessYieldRateLi))
                KpostSharpeRateLi.append(np.mean(postExcessYieldRateLi) / np.std(postYieldRateLi))
                preOtherDic.update({'AID': preNameLi, 'YieldRate': preYieldRateLi, 'OpeningTimes': preOpeningTimesLi,
                                    'ClosingTimes': preClosingTimesLi, 'ExcessYieldRate': preExcessYieldRateLi,
                                    'SharpeRate': preSharpeRateLi})
                postOtherDic.update(
                    {'AID': postNameLi, 'YieldRate': postYieldRateLi, 'OpeningTimes': postOpeningTimesLi,
                     'ClosingTimes': postClosingTimesLi, 'ExcessYieldRate': postExcessYieldRateLi,
                     'SharpeRate': postSharpeRateLi})
                preOtherDataOut = pd.DataFrame.from_dict(preOtherDic)
                postOtherDataOut = pd.DataFrame.from_dict(postOtherDic)
                preOtherDataOutPath=path3.joinpath('tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k)+'PreSummary.csv')
                postOtherDataOutPath=path3.joinpath('tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k)+'PostSummary.csv')
                preOtherDataOut.to_csv(preOtherDataOutPath)
                postOtherDataOut.to_csv(postOtherDataOutPath)
                print('tf' + str(tf) + 'tt' + str(tt) + ' end')
        KpreOtherDic.update({'TimePair': KpreNameLi, 'YieldRate': KpreYieldRateLi, 'OpeningTimes': KpreOpeningTimesLi,
                             'ClosingTimes': KpreClosingTimesLi, 'ExcessYieldRate': KpreExcessYieldRateLi,
                             'SharpeRate': KpreSharpeRateLi})
        KpostOtherDic.update(
            {'TimePair': KpostNameLi, 'YieldRate': KpostYieldRateLi, 'OpeningTimes': KpostOpeningTimesLi,
             'ClosingTimes': KpostClosingTimesLi, 'ExcessYieldRate': KpostExcessYieldRateLi,
             'SharpeRate': KpostSharpeRateLi})
        KpreOtherDataOut = pd.DataFrame.from_dict(KpreOtherDic)
        KpostOtherDataOut = pd.DataFrame.from_dict(KpostOtherDic)
        KpreOtherDataOutPath=path2.joinpath('k' + str(k)+'PreSummary.csv')
        KpostOtherDataOutPath=path2.joinpath('k' + str(k)+'PostSummary.csv')
        KpreOtherDataOut.to_csv(KpreOtherDataOutPath)
        KpostOtherDataOut.to_csv(KpostOtherDataOutPath)
        AllpreNameLi.append('k' + str(k))
        AllpreYieldRateLi.append(np.mean(KpreYieldRateLi))
        AllpreOpeningTimesLi.append(np.mean(KpreOpeningTimesLi))
        AllpreClosingTimesLi.append(np.mean(KpreClosingTimesLi))
        AllpreExcessYieldRateLi.append(np.mean(KpreExcessYieldRateLi))
        AllpreSharpeRateLi.append(np.mean(KpreExcessYieldRateLi) / np.std(KpreYieldRateLi))
        AllpostNameLi.append('k' + str(k))
        AllpostYieldRateLi.append(np.mean(KpostYieldRateLi))
        AllpostOpeningTimesLi.append(np.mean(KpostOpeningTimesLi))
        AllpostClosingTimesLi.append(np.mean(KpostClosingTimesLi))
        AllpostExcessYieldRateLi.append(np.mean(KpostExcessYieldRateLi))
        AllpostSharpeRateLi.append(np.mean(KpostExcessYieldRateLi) / np.std(KpostYieldRateLi))
        print('k=', k, ' end')
    AllpreOtherDic.update({'KValue': AllpreNameLi, 'YieldRate': AllpreYieldRateLi, 'OpeningTimes': AllpreOpeningTimesLi,
                           'ClosingTimes': AllpreClosingTimesLi, 'ExcessYieldRate': AllpreExcessYieldRateLi,
                           'SharpeRate': AllpreSharpeRateLi})
    AllpostOtherDic.update(
        {'KValue': AllpostNameLi, 'YieldRate': AllpostYieldRateLi, 'OpeningTimes': AllpostOpeningTimesLi,
         'ClosingTimes': AllpostClosingTimesLi, 'ExcessYieldRate': AllpostExcessYieldRateLi,
         'SharpeRate': AllpostSharpeRateLi})
    AllpreOtherDataOut = pd.DataFrame.from_dict(AllpreOtherDic)
    AllpostOtherDataOut = pd.DataFrame.from_dict(AllpostOtherDic)
    if IsSH:
        AllpreOtherDataOutPath=path1.joinpath('SH'+'PreSummary.csv')
        AllpostOtherDataOutPath=path1.joinpath('SH'+'PostSummary.csv')
    else:
        AllpreOtherDataOutPath=path1.joinpath('SZ'+'PreSummary.csv')
        AllpostOtherDataOutPath=path1.joinpath('SZ'+'PostSummary.csv')
    AllpreOtherDataOut.to_csv(AllpreOtherDataOutPath)
    AllpostOtherDataOut.to_csv(AllpostOtherDataOutPath)
    print("*" * 100)
    print("程序结束")


outputSummaryAll(SHStockCode, 1)
outputSummaryAll(SZStockCode, 0)
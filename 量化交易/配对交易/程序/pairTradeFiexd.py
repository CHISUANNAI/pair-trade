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


# 线性回归，返回截距与系数
def OLS(x, y):
    # x:解释变量，y:解释变量
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


# 判断是否能成为配对，返回是否为配对，1为是，0为否;截距;系数;残差的标准差
def IfPair(data, T0, tf, CD):
    # data为要检验的数据，start为起始时间，精确到日，length为时间跨度，单位为月份，CD为置信度
    data = pd.DataFrame(data, dtype=float)
    T0 = datetime.datetime.strptime(T0, "%Y-%m-%d")
    end = T0 + relativedelta(months=tf)
    temp = data.loc[T0:end].copy()
    temp.iloc[:, 0] = np.log(temp.iloc[:, 0])
    temp.iloc[:, 1] = np.log(temp.iloc[:, 1])
    # 进行线性回归
    x = temp.iloc[:, 0].copy()
    y = temp.iloc[:, 1].copy()
    parameter = OLS(x, y)
    re = y - x * parameter[1] - parameter[0]
    std = re.std()
    # ADF检验
    try:
        ADFresult = adfuller(re)
        if ADFresult[1] < CD:
            return [1, parameter[0], parameter[1], std]
        # ADFresultCT = adfuller(re, regression='ct')
        # if ADFresultCT[1] < CD:
        #     return [1, parameter[0], parameter[1], std]
        # ADFresultCTT = adfuller(re, regression='ctt')
        # if ADFresultCTT[1] < CD:
        #     return [1, parameter[0], parameter[1], std]
        # ADFresultCNC = adfuller(re, regression='n')
        # if ADFresultCNC[1] < CD:
        #     return [1, parameter[0], parameter[1], std]
    except:
        return [0, parameter[0], parameter[1], std]
    return [0, parameter[0], parameter[1], std]


# 计算A股交易费用
def CalculateATF(IsS, SNV):
    # IsS:是否是卖方,SNV：股票净值
    # A股佣金比例（ACR）买卖双向收取：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.0005；过户费(TransferF)买卖双向收取,深股不收取：0.00001；
    # 交易经手费(THF)：由交易所收取，买卖双向收取,万分之0.341(0.00341%),证管费(SCF)：由证监会收取，买卖双向收取。万分之0.2（0.002%）
    ACR = 0.0003
    SD = 0.0005
    TransferF = 0.00001
    THF = 0.0000341
    SCF = 0.00002
    if IsS == True:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > 5 else 5
        SDF = Decimal(SNV * SD).quantize(Decimal("0.00"))
        TransferFF = Decimal(SNV * TransferF).quantize(Decimal("0.00"))
        THFF = Decimal(SNV * THF).quantize(Decimal("0.00"))
        SCFF = Decimal(SNV * SCF).quantize(Decimal("0.00"))
        ATF = ACRF + SDF + TransferFF + THFF + SCFF
        return ATF
    else:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > 5 else 5
        TransferFF = Decimal(SNV * TransferF).quantize(Decimal("0.00"))
        THFF = Decimal(SNV * THF).quantize(Decimal("0.00"))
        SCFF = Decimal(SNV * SCF).quantize(Decimal("0.00"))
        ATF = ACRF + TransferFF + THFF + SCFF
        return ATF


# 计算H股交易费用
def CalculateHTF(IsS, SNV, ExchangeRate):
    # SNV：股票净值
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.001，不足一元亦作一元计;
    # 卖方负责缴付；过户费用(TransferF):2.50港币，由买方支付
    ACR = 0.0025
    SFCTL = 0.000027
    FRCTL = 0.0000015
    TF = 0.0000565
    SD = 0.001
    TransferF = 2.50
    if IsS == True:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > Decimal(
            100 * ExchangeRate).quantize(Decimal("0.00")) else Decimal(100 * ExchangeRate).quantize(Decimal("0.00"))
        SFCTLF = Decimal(SNV * SFCTL).quantize(Decimal("0.00"))
        FRCTLF = Decimal(SNV * FRCTL).quantize(Decimal("0.00"))
        TFF = Decimal(SNV * TF).quantize(Decimal("0.00"))
        SDF = math.ceil(SNV * SD)
        HTF = ACRF + SFCTLF + FRCTLF + TFF + SDF
        return HTF
    else:
        ACRF = Decimal(SNV * ACR).quantize(Decimal("0.00")) if Decimal(SNV * ACR).quantize(Decimal("0.00")) > Decimal(
            100 * ExchangeRate).quantize(Decimal("0.00")) else Decimal(100 * ExchangeRate).quantize(Decimal("0.00"))
        SFCTLF = Decimal(SNV * SFCTL).quantize(Decimal("0.00"))
        FRCTLF = Decimal(SNV * FRCTL).quantize(Decimal("0.00"))
        TFF = Decimal(SNV * TF).quantize(Decimal("0.00"))
        SDF = math.ceil(SNV * SD)
        TransferFF = Decimal(TransferF * ExchangeRate).quantize(Decimal("0.00"))
        HTF = ACRF + SFCTLF + FRCTLF + TFF + SDF + TransferFF
        return HTF


# 分配初始金额函数,返回值为数组:A股数量;港股数量
def AllotmentAmount(All, r, ExchangeRate, PA, PH):
    # All:初始资金;r:比例系数，为在判断是否为配对的函数中的比例系数;DateTime:分配初始金额的日期;date:股票的价格数据
    # A股佣金比例（ACR）买卖双向收取：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.0005；过户费(TransferF)买卖双向收取：0.00001；
    # 交易经手费(THF)：由交易所收取，买卖双向收取,万分之0.341(0.0000341),证管费(SCF)：由证监会收取，买卖双向收取。万分之0.2（0.00002）
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.001，不足一元亦作一元计;
    # 过户费用(TransferF):2.50港币，由买方支付
    # 对斜率进行修正
    if r < 0:
        r = r * -1
    T = 2.5 * ExchangeRate  # 过户费用
    rA = 0.0003641  # A股交易费用的比例部分
    rH = 0.003585  # H股交易费用的比例部分
    rA1 = 0.0000641  # A股去除佣金部分的比例系数
    ACRA = 0.0003  # A股佣金比例（ACR）
    ACRFA = 5  # A股佣金固定数值
    rH1 = 0.001085  # H股去除佣金部分的比例系数
    ACRFH = 100 * ExchangeRate  # H股佣金固定数值
    ACRH = 0.0025  # H股佣金比例（ACR）
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


# 购买A股函数，返回这笔资金能购买的A股股票数量
def NumberOfACanBePurchased(All, price):
    # All:资金;DateTime:日期;price:价格
    # A股佣金比例（ACR）：0.0003,不满5元按5元收取;印花税(SD)：向卖方收取0.00005；过户费(TransferF)：0.00001；交易所规费(EF)：0.0000687；
    rA = 0.0003641  # A股交易费用的比例部分
    rA1 = 0.0000641  # A股去除佣金部分的比例系数
    ACRA = 0.0003  # A股佣金比例（ACR）
    ACRFA = 5  # A股佣金固定数值
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


# 购买H股函数，返回这笔资金能购买的H股股票数量
def NumberOfHCanBePurchased(All, price, ExchangeRate):
    # 港股佣金比例(ACR)：0.0025，最低收费100港币；证监会交易征费(SFCTL)：0.000027;财务汇报局交易征费(FRCTL)：0.0000015;交易费(TF):0.0000565;H股印花税(SD)：0.0013，不足一元亦作一元计;
    # 转手纸印花税(SDOTP):5港币，卖方负责缴付；过户费用(TransferF):2.50港币，由买方支付
    T = 2.5 * ExchangeRate  # 过户费用
    rH = 0.003585  # H股交易费用的比例部分
    rH1 = 0.001085  # H股去除佣金部分的比例系数
    ACRFH = 100 * ExchangeRate  # H股佣金固定数值
    ACRH = 0.0025  # H股佣金比例（ACR）
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


# 获取汇率
def getExchangeRate(time):
    # print(time)
    while not (time in HKDCNY.index):
        time = time - relativedelta(days=1)
    ExchangeRate = HKDCNY.loc[:, '开盘价(元)'][time]
    return ExchangeRate


# 交易函数，将模拟交易过程,返回为收益率,开仓次数,平仓次数
def trade(AID, All, data, CD, T0, tf, tt, dc, do, ds):
    # print("*" * 100)
    # print("开始交易", "证券代码:", AID, "开始日期:", T0, "形成期长度:", tf, "交易期长度:", tt)
    # 是否为配对，1为是，0为否;截距;系数;残差的标准差
    # print(data)
    name = T0
    dataCopy = data.copy()
    par = IfPair(dataCopy, T0, tf, CD)
    if par[0] == 0:
        # print("不构成配对，交易结束")
        # templi = []
        # while len(templi) < tt * 30:
        #     templi.append(0)
        return [None, None]
    # dc平仓阈值系数,do为开仓阈值系数，ds为止损阈值系数
    dok = par[3] * do
    dck = par[3] * dc
    dsk = par[3] * ds
    T0 = datetime.datetime.strptime(T0, "%Y-%m-%d")
    T1 = T0 + relativedelta(months=tf)
    T2 = T0 + relativedelta(months=tf + tt)
    TradingPeriodData = data.loc[T1:T2].copy()
    if TradingPeriodData.size == 0:
        # templi = []
        # while len(templi) < tt * 30 + 4:
        #     templi.append(0)
        return [None, None]
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
    ATF = CalculateATF(0, nA * PA0)
    HTF = CalculateHTF(0, nH * PH0, ExchangeRate0)
    TF = ATF + HTF
    rePre = None
    nALoan = 0
    nHLoan = 0
    ASecurityDeposit = 0
    HSecurityDeposit = 0
    IsOpen = False
    reli = []  # 用来记录每次模拟交易的残差值
    TradingPeriodData = pd.DataFrame(TradingPeriodData, dtype=float)
    TradingPeriodDataCopy = TradingPeriodData.copy()
    for index, row in TradingPeriodDataCopy.iterrows():

        re = math.log(row[1]) - math.log(row[0]) * par[2] - par[1]
        reli.append(re)
        reAbs = abs(re)
        # ExchangeRate = HKDCNY['开盘价(元)'][index]
        ExchangeRate = getExchangeRate(index)
        # print(re, re * 100 / par[3], dck, dok, dsk)
        if reAbs > dsk:
            if nA != 0:
                nACash = Decimal((nA - nALoan) * row[0]).quantize(Decimal("0.00")) - CalculateATF(1,
                                                                                                  (nA - nALoan) * row[
                                                                                                      0])
                ATF = ATF + CalculateATF(1,(nA - nALoan) * row[0])
                nA = 0
                nALoan = 0
                cash = cash + nACash + ASecurityDeposit
                ASecurityDeposit = 0
            if nH != 0:
                nHCash = Decimal((nH - nHLoan) * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1,
                                                                                                  (nH - nHLoan) * row[
                                                                                                      1], ExchangeRate)
                HTF = HTF + CalculateHTF(1,(nH - nHLoan) * row[1], ExchangeRate)
                nH = 0
                nHLoan = 0
                cash = cash + nHCash + HSecurityDeposit
                HSecurityDeposit = 0
            if nALoan != 0 or nHLoan != 0:
                nALoanCost = Decimal(nALoan * row[0]).quantize(Decimal("0.00")) + CalculateATF(0, nALoan * row[0])
                nHLoanCost = Decimal(nHLoan * row[1]).quantize(Decimal("0.00")) + CalculateHTF(0, nHLoan * row[1],
                                                                                               ExchangeRate)
                ATF=ATF+CalculateATF(0, nALoan * row[0])
                HTF=HTF+CalculateHTF(0, nHLoan * row[1],ExchangeRate)
                nALoan = 0
                nHLoan = 0
                cash = cash - nALoanCost - nHLoanCost + ASecurityDeposit + HSecurityDeposit
                ASecurityDeposit = 0
                HSecurityDeposit = 0
            nAStaticCash = Decimal(nAStatic * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nAStatic * row[0])
            nHStaticCash = Decimal(nHStatic * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nHStatic * row[1],
                                                                                               ExchangeRate)
            staticCash = nAStaticCash + nHStaticCash
            staticYieldRate = float((staticCash - All) / All)
            yieldRate = float((cash - All) / All)
            excessYieldRate = yieldRate - staticYieldRate
            TF = ATF + HTF
            # print("超出阈值，停止交易")
            # while len(redf) < tt * 30:
            #     redf.append(0)
            resultsDf = pd.DataFrame(
                columns=['name', 'yieldRate', 'excessYieldRate', 'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF',
                         'slope',
                         'intercept', 'std'])
            resultRow = {'name': name,
                         'yieldRate': yieldRate,
                         'excessYieldRate': excessYieldRate,
                         'openingTimes': openingTimes,
                         'closingTimes': closingTimes,
                         'ATF': ATF,
                         'HTF': HTF,
                         'TF': TF,
                         'slope': par[2],
                         'intercept': par[1],
                         'std': par[3]
                         }
            resultsDf = pd.concat([resultsDf, pd.DataFrame(resultRow, index=[0])], ignore_index=True)
            redf = pd.Series(reli, name=name)
            return [redf, resultsDf]
        if rePre is None:
            if reAbs > dok:
                if re > 0:
                    nHLoan = nH
                    HSecurityDeposit = Decimal(nH * row[1]).quantize(Decimal("0.00"))
                    nHCash = Decimal(nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nH * row[1], ExchangeRate)
                    HTF = HTF + CalculateHTF(1, nH * row[1], ExchangeRate)
                    nH = 0
                    cash = cash + nHCash
                    nAP = NumberOfACanBePurchased(float(cash), row[0])
                    cash = cash - Decimal(nAP * row[0]).quantize(Decimal("0.00")) - CalculateATF(0, nAP * row[0])
                    ATF = ATF + CalculateATF(0, nAP * row[0])
                    nA = nA + nAP
                    TF = ATF + HTF
                if re < 0:
                    nALoan = nA
                    ASecurityDeposit = Decimal(nA * row[0]).quantize(Decimal("0.00"))
                    nACash = Decimal(nA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nA * row[0])
                    ATF = ATF + CalculateATF(1, nA * row[0])
                    nA = 0
                    cash = cash + nACash
                    nHP = NumberOfHCanBePurchased(float(cash), row[1], ExchangeRate)
                    cash = cash - Decimal(nHP * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, nHP * row[1],
                                                                                                 ExchangeRate)
                    HTF = HTF + CalculateHTF(0, nHP * row[1], ExchangeRate)
                    nH = nH + nHP
                    TF = ATF + HTF
                # print("开仓")
                IsOpen = True
                openingTimes = openingTimes + 1
        else:
            if (reAbs > dok and (not IsOpen)) or (reAbs > dok and IsOpen and re * rePre < 0):
                if re > 0:
                    nHLoan = nH
                    HSecurityDeposit = Decimal(nH * row[1]).quantize(Decimal("0.00"))
                    nHCash = Decimal(nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1, nH * row[1], ExchangeRate)
                    HTF = HTF + CalculateHTF(1, nH * row[1], ExchangeRate)
                    nH = 0
                    cash = cash + nHCash
                    nACost = Decimal(nALoan * row[0]).quantize(Decimal("0.00")) + CalculateATF(0, nALoan * row[0])
                    ATF = ATF+CalculateATF(0, nALoan * row[0])
                    nALoan = 0
                    cash = cash - nACost + ASecurityDeposit
                    ASecurityDeposit = 0
                    nAP = NumberOfACanBePurchased(float(cash), row[0])
                    cash = cash - Decimal(nAP * row[0]).quantize(Decimal("0.00")) - CalculateATF(0, nAP * row[0])
                    nA = nA + nAP
                    ATF = ATF + CalculateATF(0, nAP * row[0])
                    TF = ATF + HTF
                if re < 0:
                    nALoan = nA
                    ASecurityDeposit = Decimal(nA * row[0]).quantize(Decimal("0.00"))
                    nACash = Decimal(nA * row[0]).quantize(Decimal("0.00")) - CalculateATF(1, nA * row[0])
                    ATF = ATF + CalculateATF(1, nA * row[0])
                    nA = 0
                    cash = cash + nACash
                    nHCost = Decimal(nHLoan * row[1]).quantize(Decimal("0.00")) + CalculateHTF(0, nHLoan * row[1],
                                                                                               ExchangeRate)
                    HTF = HTF + CalculateHTF(0, nHLoan * row[1],ExchangeRate)
                    nHLoan = 0
                    cash = cash - nHCost + HSecurityDeposit
                    HSecurityDeposit = 0
                    nHP = NumberOfHCanBePurchased(float(cash), row[1], ExchangeRate)
                    cash = cash - Decimal(nHP * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, nHP * row[1],
                                                                                                 ExchangeRate)
                    HTF = HTF + CalculateHTF(0, nHP * row[1], ExchangeRate)
                    TF = ATF + HTF
                    nH = nH + nHP
                # print("开仓")
                if (reAbs > dok and IsOpen and re * rePre < 0):
                    closingTimes = closingTimes + 1
                IsOpen = True
                openingTimes = openingTimes + 1
            if abs(re / par[3]) <= 0.00001 and reAbs < dok and IsOpen == True:
                if nA != 0:
                    nACash = Decimal((nA - nALoan) * row[0]).quantize(Decimal("0.00")) - CalculateATF(1,
                                                                                                      (nA - nALoan) *
                                                                                                      row[0])
                    ATF = ATF + CalculateATF(1, (nA - nALoan)  * row[0])
                    nA = 0
                    nALoan = 0
                    cash = cash + nACash + ASecurityDeposit
                    ASecurityDeposit = 0
                if nH != 0:
                    nHCash = Decimal((nH - nHLoan) * row[1]).quantize(Decimal("0.00")) - CalculateHTF(1,
                                                                                                      (nH - nHLoan) *
                                                                                                      row[1],ExchangeRate)
                    HTF = HTF + CalculateHTF(1, (nH - nHLoan) * row[1], ExchangeRate)
                    nH = 0
                    nHLoan = 0
                    cash = cash + nHCash + HSecurityDeposit
                    HSecurityDeposit = 0
                if nALoan != 0 or nHLoan != 0:
                    nALoanCost = Decimal(nALoan * row[0]).quantize(Decimal("0.00")) + CalculateATF(0, nALoan * row[0])
                    nHLoanCost = Decimal(nHLoan * row[1]).quantize(Decimal("0.00")) + CalculateHTF(0, nHLoan * row[1],
                                                                                                   ExchangeRate)
                    ATF = ATF + CalculateATF(0, nALoan * row[0])
                    HTF=HTF+CalculateHTF(0, nHLoan * row[1],ExchangeRate)
                    nALoan = 0
                    nHLoan = 0
                    cash = cash - nALoanCost - nHLoanCost + ASecurityDeposit + HSecurityDeposit
                    ASecurityDeposit = 0
                    HSecurityDeposit = 0
                # print("平仓")
                par2 = AllotmentAmount(cash, par[2], ExchangeRate, row[0], row[1])
                nA = par2[0]
                nH = par2[1]
                cash = Decimal(cash).quantize(Decimal("0.00")) - Decimal(nA * row[0]).quantize(
                    Decimal("0.00")) - CalculateATF(0,
                                                    nA * row[0]) - Decimal(
                    nH * row[1]).quantize(Decimal("0.00")) - CalculateHTF(0, nH * row[1], ExchangeRate)
                ATF =ATF+ CalculateATF(0, nA * row[0])
                HTF =HTF+ CalculateHTF(0, nH * row[1], ExchangeRate)
                TF = ATF + HTF
                IsOpen = False
                closingTimes = closingTimes + 1
        rePre = re
    # ExchangeRateLast = HKDCNY['开盘价(元)'][TradingPeriodData.tail(1).index][0]
    ExchangeRateLast = getExchangeRate(TradingPeriodData.tail(1).index[0])
    PALast = TradingPeriodData.iloc[:, 0][TradingPeriodData.tail(1).index][0]
    PHLast = TradingPeriodData.iloc[:, 1][TradingPeriodData.tail(1).index][0]
    if nA != 0:
        nACash = Decimal((nA - nALoan) * PALast).quantize(Decimal("0.00")) - CalculateATF(1,
                                                                                          (nA - nALoan) * PALast)
        ATF = ATF + CalculateATF(1, (nA - nALoan) * PALast)
        nA = 0
        nALoan = 0
        cash = cash + nACash + ASecurityDeposit
        ASecurityDeposit = 0
    if nH != 0:
        nHCash = Decimal((nH - nHLoan) * PHLast).quantize(Decimal("0.00")) - CalculateHTF(1,
                                                                                          (nH - nHLoan) * PHLast, ExchangeRateLast)
        HTF = HTF + CalculateHTF(1, (nH - nHLoan) * PHLast, ExchangeRateLast)
        nH = 0
        nHLoan = 0
        cash = cash + nHCash + HSecurityDeposit
        HSecurityDeposit = 0
    if nALoan != 0 or nHLoan != 0:
        nALoanCost = Decimal(nALoan * PALast).quantize(Decimal("0.00")) + CalculateATF(0, nALoan * PALast)
        nHLoanCost = Decimal(nHLoan * PHLast).quantize(Decimal("0.00")) + CalculateHTF(0, nHLoan * PHLast,ExchangeRateLast)
        ATF = ATF + CalculateATF(0, nALoan * PALast)
        HTF = HTF + CalculateHTF(0, nHLoan * PHLast, ExchangeRateLast)
        nALoan = 0
        nHLoan = 0
        cash = cash - nALoanCost - nHLoanCost + ASecurityDeposit + HSecurityDeposit
        ASecurityDeposit = 0
        HSecurityDeposit = 0
    nAStaticCash = Decimal(nAStatic * PALast).quantize(Decimal("0.00")) - CalculateATF(1, nAStatic * PALast)
    nHStaticCash = Decimal(nHStatic * PHLast).quantize(Decimal("0.00")) - CalculateHTF(1, nHStatic * PHLast,
                                                                                       ExchangeRateLast)
    TF = ATF + HTF
    staticCash = nAStaticCash + nHStaticCash
    staticYieldRate = float((staticCash - All) / All)
    yieldRate = float((cash - All) / All)
    excessYieldRate = yieldRate - staticYieldRate
    # print("结束交易")
    # while len(redf) < tt * 30:
    #     redf.append(0)
    resultsDf = pd.DataFrame(
        columns=['name', 'yieldRate', 'excessYieldRate', 'openingTimes', 'closingTimes', 'ATF', 'HTF', 'TF', 'slope',
                 'intercept', 'std'])
    resultRow = {'name': name,
                 'yieldRate': yieldRate,
                 'excessYieldRate': excessYieldRate,
                 'openingTimes': openingTimes,
                 'closingTimes': closingTimes,
                 'ATF': ATF,
                 'HTF': HTF,
                 'TF': TF,
                 'slope': par[2],
                 'intercept': par[1],
                 'std': par[3]
                 }
    resultsDf = pd.concat([resultsDf, pd.DataFrame(resultRow, index=[0])], ignore_index=True)
    redf = pd.Series(reli, name=name)
    return [redf, resultsDf]


# T0 = '2014-12-01'
# data = DataSelect('600377.SH')
# data = DataClean(data)
# dataCopy = data.copy()
# tradeData = trade('600377.SH', 1000000, dataCopy, 0.05, T0, 1, 1, 0, 0.025, 1.96)
# # tradeData[0]
# tradeData[1]

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


def pariTradeMain(IDSet, IsSH, kli):
    All = 1000000
    CD = 0.05
    mkdir(pathlib.Path('..', 'resultFixed'))
    path1 = None
    path2 = None
    path3 = None
    earlyTime = '2002-04-02'
    endTime = '2023-08-25'
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d")
    earlyTime = datetime.datetime.strptime(earlyTime, "%Y-%m-%d")
    if IsSH == 1:
        path1 = pathlib.Path('..', 'resultFixed', 'SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    else:
        path1 = pathlib.Path('..', 'resultFixed', 'SZ')
        interInterworkingTime = '2016-12-05'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
    # kli = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5]
    # kli = np.array([0,0.25,0.5,0.75,1])
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
    for k in kli:
        print('k=', k)
        path2 = path1.joinpath('k' + str(k))
        mkdir(path2)
        for tf in range(1, 25):
            for tt in range(1, 13):
                # for tf in range(1, 3):
                #     for tt in range(1, 3):
                # for tf in range(10, 11):
                #     for tt in range(12, 13):
                print('tf=', tf, 'tt=', tt)
                path3 = path2.joinpath('tf' + str(tf) + 'tt' + str(tt))
                mkdir(path3)
                path4 = path3.joinpath('pre')
                mkdir(path4)
                path5 = path3.joinpath('post')
                mkdir(path5)
                for AID in IDSet:
                    print(AID)
                    data = DataSelect(AID)
                    data = DataClean(data)
                    T0Li = []
                    preReDIC = {}
                    postReDIC = {}
                    T0set = set()
                    preResultsDf = pd.DataFrame(
                        columns=['name', 'yieldRate', 'excessYieldRate', 'openingTimes', 'closingTimes', 'ATF', 'HTF',
                                 'TF',
                                 'slope',
                                 'intercept', 'std'])
                    postResultsDf = pd.DataFrame(
                        columns=['name', 'yieldRate', 'excessYieldRate', 'openingTimes', 'closingTimes', 'ATF', 'HTF',
                                 'TF',
                                 'slope',
                                 'intercept', 'std'])
                    preReDf = pd.DataFrame()
                    postReDf = pd.DataFrame()
                    for i in range(0, data.shape[0]):
                        T0 = data.iloc[i].name
                        T0 = T0.replace(day=1)
                        if not T0.strftime("%Y-%m-%d") in T0set:
                            T0set.add(T0.strftime("%Y-%m-%d"))
                            T0Li.append(T0)
                    try:
                        for T0 in T0Li:
                            if T0 < earlyTime or (T0 + relativedelta(
                                    months=tf + tt) > interInterworkingTime and T0 < interInterworkingTime) or (
                                    T0 > interInterworkingTime and (T0 + relativedelta(months=tf + tt) > endTime)):
                                continue
                            # [yieldRate, openingTimes, closingTimes,redf]
                            # resultName = str(T0) + 'tf' + str(tf) + 'tt' + str(tt)
                            T0Copy = T0.strftime("%Y-%m-%d")
                            dataCopy = data.copy()
                            tradeData = trade(AID, All, dataCopy, CD, T0Copy, tf, tt, 0, k, 1.96)
                            redf = tradeData[0]
                            resultDf = tradeData[1]
                            if not (redf is None):
                                if T0 < interInterworkingTime:
                                    preReDf = pd.concat([preReDf, redf], axis=1)
                                    preResultsDf = pd.concat([preResultsDf, pd.DataFrame(resultDf, index=[0])],
                                                             ignore_index=True)
                                else:
                                    postReDf = pd.concat([postReDf, redf], axis=1)
                                    postResultsDf = pd.concat([postResultsDf, pd.DataFrame(resultDf, index=[0])],
                                                              ignore_index=True)
                    except:
                        print(AID)
                        continue
                    if preReDf.size > 0:
                        preReDf.to_csv(
                            path4.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Re' + '.csv'))
                        preResultsDf.to_csv(
                            path4.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Result' + '.csv'))
                    if postReDf.size > 0:
                        postReDf.to_csv(
                            path5.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Re' + '.csv'))
                        postResultsDf.to_csv(
                            path5.joinpath(AID + 'tf' + str(tf) + 'tt' + str(tt) + 'k' + str(k) + 'Result' + '.csv'))
                print('tf' + str(tf) + 'tt' + str(tt) + ' end')
        print('k=', k, ' end')
    print("*" * 100)
    print("程序结束")


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
    mkdir(pathlib.Path('..', 'SummaryByTimeFixed'))
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
        path1 = pathlib.Path('..', 'SummaryByTimeFixed', 'SH')
        datapath1 = pathlib.Path('..', 'resultFixed', 'SH')
        interInterworkingTime = '2014-11-17'
        interInterworkingTime = datetime.datetime.strptime(interInterworkingTime, "%Y-%m-%d")
        # 设置沪港通前后的时间范围
        preStartTime = '2006-02-01'
        preEndTime = '2014-05-01'
        postStartTime = '2015-05-01'
        postEndTime = '2023-08-01'
    else:
        path1 = pathlib.Path('..', 'SummaryByTimeFixed', 'SZ')
        datapath1 = pathlib.Path('..', 'resultFixed', 'SZ')
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


SHkliTrade = np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
SZkliTrade = np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
# SHkliTrade = np.array([0.025, 0.075, 0.0125, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
# SZkliTrade = np.array([0.01, 0.02, 0.03, 0.04, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4, 1.45])
SHkliCaculate = np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
SZkliCaculate = np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
# SHkliCaculate = np.array(
#     [0, 0.025, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2,
#      1.25, 1.5, 1.75, 2])
# SZkliCaculate = np.array(
#     [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4,
#      1.45, 1.5, 1.75, 2])
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
pariTradeMain(SHStockCode, 1, SHkliTrade)
pariTradeMain(SZStockCode, 0, SZkliTrade)
outputSummaryByTime(SHStockCode, 1, SHkliCaculate)
outputSummaryByTime(SZStockCode, 0, SZkliCaculate)

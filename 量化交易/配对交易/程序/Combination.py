import numpy as np
import pandas as pd

df1=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part1.csv')
df2=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part2.csv')
df3=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part3.csv')
df4=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part4.csv')
df5=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part5.csv')
df6=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part6.csv')
df7=pd.read_csv('F:\热爱学习\jupyter\量化交易\配对交易\港股\part7.csv')
pieces=[df1,df2,df3,df4,df5,df6,df7]
df=pd.concat(pieces)
df.to_csv('HKStock.csv')


def calculate_grade(score):
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    if score >= 60 and score <= 100:
        if score % 10 >= 7:
            modifier = "+"
        elif score % 10 <= 3:
            modifier = "-"
        else:
            modifier = ""
    else:
        modifier = ""

    return grade + modifier
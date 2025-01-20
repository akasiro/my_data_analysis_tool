# 内置包
import sys,time,re,warnings,os
from datetime import datetime
from math import sqrt, ceil

# 分析工具
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

# 画图
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 建模
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from xgboost import plot_tree

# 展示
from IPython.display import HTML


# 个性化包
sys.path.append('/Users/ping/Desktop/chenping06backup/Desktop/chenping/from')
sys.path.append('/Users/ping/Desktop/chenping06backup/Desktop/chenping/tool/')
# 新版
from ABTestStatsAnalysis import DiDMultMetric

# 工具
from my_data_analysis_tool import exp_analysis_tool as myeal #urlToParam,toTable,genExpQueryParam,expMetricCal

# mac中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 默认字体大小
plt.rcParams.update({'font.size': 20})
# 不显示警告
warnings.filterwarnings('ignore')

# 数据读取地址
read_path = '/Users/ping/Downloads/'

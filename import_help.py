# 内置包
import sys,time,re,warnings
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

# 个性化包
sys.path.append('/Users/ping/Desktop/chenping06backup/Desktop/chenping/from')
# 新版
from ABTestStatsAnalysis import DiDMultMetric

# mac中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 默认字体大小
plt.rcParams.update({'font.size': 20})
# 不显示警告
warnings.filterwarnings('ignore')



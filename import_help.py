# %load /Users/ping/Desktop/chenping06backup/Desktop/chenping/tool/my_data_analysis_tool/import_help.py
# 内置包
import sys,time,re,warnings,os,pickle
from datetime import datetime,timedelta
from math import sqrt, ceil

# 分析工具
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from numpy import nan
# 画图
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

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


# 常见维度对应表
gender_dict = {
    0.0:'0:女'
    ,1.0:'1:男'
    ,np.nan:'2:未知'
    ,-124:'2:未知'
}
age_dict = {
    0.0:'0:<17'
    ,1.0:'1:18-24'
    ,2.0:'2:25-36'
    ,3.0:'3:37-50'
    ,4.0:'4:50+'
    ,np.nan:'5:未知'
    ,-124:'5:未知'
}
demographic_dict = {
    'Northeastern region':'1:东北部'
    ,'Northern region':'2:北部'
    ,'Central and western region':'3:中西部'
    ,'Southern region':'4:南部'
    ,'Southeastern region':'5:东南部'
    ,np.nan:'6:未知'
}
traffic_dict = {
    'other':'1:自然'
    ,'social':'2:社交'
    ,'e_commercial':'3:电商'
    ,'kwai_task':'4:快任务'
    ,'boost':'5:助推'
    ,'cold_launch':'6:冷启'
    ,'insert':'7:强插'
    , np.nan: '1:自然'
}
active_degree_dict = {
    'active_1~3days_last_week':'3:周低活'
    ,'silent_30+':'5:沉默30+'
    ,'is_new_device':'6:新设备'
    ,'active_7days_last_week':'1:周高活'
    ,'silent_7~29':'4:沉默7-9'
    ,'active_4~6days_last_week':'2:周中活'
    ,np.nan:'7:未知'
}

with open('/Users/ping/Desktop/chenping06backup/Desktop/chenping/tool/my_data_analysis_tool/sql_parttern.pkl','rb') as f:
    sql_parttern = pickle.load(f)
with open('/Users/ping/Desktop/chenping06backup/Desktop/chenping/tool/my_data_analysis_tool/xtr.pkl', 'rb') as f:
    xtr = pickle.load(f)
    xtr_dict = xtr.get('xtr_dict')
    xtr_df = xtr.get('xtr_df')

def xtr_df_style_func(styler):
    styler.format(
        '{:.2%}',subset=pd.IndexSlice[(xtr_df['metric_type'].str.contains('ratio')) & (xtr_df['emp'] >= 0.0005), 'emp']
    ).format(
        '{:,.2f}',subset=pd.IndexSlice[(xtr_df['metric_type'].str.contains('numeric')) & (xtr_df['emp'] <= 1000), 'emp']
    ).format(
        '{:,.0f}',subset=pd.IndexSlice[(xtr_df['metric_type'].str.contains('numeric')) & (xtr_df['emp'] > 1000), 'emp']
    ).format(
        '{:.4%}',subset=pd.IndexSlice[(xtr_df['metric_type'].str.contains('ratio')) & (xtr_df['emp'] < 0.0005), 'emp']
    )
    return styler
def general_style_func(styler):
    styler.set_table_styles([
            {'selector': '.redcell', 'props': 'color:red;'}
            ,{'selector': '.greencell', 'props': 'color:green;'}
            ,{'selector': '.greycell', 'props': 'color:grey;'}
            ,{'selector':'thead','props':[('background-color','#D1F2FF'),('text-align','center')]}                                                                 
            ,{'selector':'td,th','props':[('border','1px solid black')]}                                                                  
            ,{'selector':'tbody,td,th.row_heading','props':[('text-align','right')]}                                                            
            ,{'selector':'td','props':[('text-align','right')]}
    ]).set_table_attributes('style="border-collapse: collapse;"')
    return styler
xtr_df.style.pipe(xtr_df_style_func).pipe(general_style_func)

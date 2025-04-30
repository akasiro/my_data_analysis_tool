
from urllib.parse import unquote,parse_qs
from datetime import datetime,timedelta
from math import sqrt, ceil

import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def parse_url_params(url):
    """url解析器
    Args:
        url(str): 带参数的url
    Retuens(dict): 解析后的参数
    """
    # 找到问号的位置，参数从问号后开始
    question_mark_index = url.find('?')
    if question_mark_index == -1:
        return {}

    # 获取参数部分
    query_string = url[question_mark_index + 1:]

    # 分割每个参数
    params = query_string.split('&')

    # 存储参数的字典
    param_dict = defaultdict(list)

    for param in params:
        # 分割键和值
        key_value = param.split('=')
        if len(key_value) == 2:
            key, value = key_value
            param_dict[key].append(value)
        elif len(key_value) == 1:
            key = key_value[0]
            param_dict[key].append('')

    return dict(param_dict)

def urlToParam(url):
    """url解析器
    Args:
        url(str): 带参数的url
    Retuens(dict): 解析后的参数
    Example:
        >>> myeal.urlToParam(url)
        {'world_name': 'w_n_kwai_apps_did_1511',
        'exp_name': 'fr_rank_5percent_20250306',
        'group_col': 'group_name',
        'dt_col': 'dt',
        'base_groups': ['base1', 'base2', 'base3', 'base4'],
        'exp_groups': ['exp5', 'exp6'],
        'bucket_col': 'bucket_id',
        'dt_AA_start': '2025-03-06',
        'dt_AA_end': '2025-03-13',
        'dt_AB_start': '2025-03-22',
        'dt_AB_end': '2025-03-26'}
    """
    def timeParser(value):
        if not isinstance(value,int):
            value = int(value)
        if len(str(value)) == 13:
            value = value/1000
        if value == 0:
            return None
        else:
            return datetime.fromtimestamp(value).strftime('%Y-%m-%d')
    def groupParser(value):
        if not isinstance(value,list):
            value = eval(value)
        return sorted([i.split(':')[-1] for i in value ])
    resDict = {}
    url = unquote(url)
  
    # queryDict = parse_qs(url)
 
    queryDict = parse_url_params(url)
    world_name = eval(queryDict.get('worldName')[0])
    exp_name = eval(queryDict.get('experimentName')[0])
    
    resDict['world_name'] = world_name
    resDict['exp_name'] = exp_name
    resDict['group_col'] = 'group_name'
    resDict['dt_col'] = 'dt'
    resDict['base_groups'] = groupParser(queryDict.get('fromGroup')[0])
    resDict['exp_groups'] = groupParser(queryDict.get('toGroup')[0])
    resDict['bucket_col'] = 'bucket_id'
    resDict['dt_AA_start'] = timeParser(queryDict.get('preAAStartTime')[0])
    resDict['dt_AA_end'] = timeParser(queryDict.get('preAAEndTime')[0])
    resDict['dt_AB_start'] = timeParser(queryDict.get('from')[0])
    resDict['dt_AB_end'] = timeParser(queryDict.get('to')[0])
    return resDict
def toTable(url):
    """url转化为带格式的表格
    Args:
        url(str): 带参数的url
    Retuens(pandas.DataFrame.Styler): 格式化表格
    """
    param = urlToParam(url)
    df = pd.DataFrame(list(param.items()), columns=['key', 'value']).set_index('key').loc[['world_name','exp_name','base_groups','exp_groups','dt_AA_start','dt_AA_end','dt_AB_start','dt_AB_end']]
    # df.loc['exp_name'] = {'key':'url','value':'<a href="{}">实验链接</a>'.format(url)}
    df = df.reset_index()
    df['value'] = df.apply(lambda x:'<a href="{}">{}</a>'.format(url,x['value']) if x['key'] == 'exp_name' else x['value'],axis=1)
    
    # df
    cls_url = pd.DataFrame(df.apply(lambda x: np.where(x == '实验链接','cls-url','')))
    df_s = df.style.hide(axis= 'columns').hide()
    df_s = df_s.set_td_classes(cls_url)\
    .set_table_styles([
        {'selector':'.cls-url','props':'href:{}'.format(url)}
    ]).set_table_styles([                                                                          
        {'selector':'td,th','props':[('border','1px solid black')]}                                         
        ,{'selector':'tbody,td,th.row_heading','props':[('text-align','left')]}                                     
    ]).set_table_attributes('style="border-collapse: collapse;"')
    return df_s

def genExpQueryParam(
        sql_parttern = None
        ,group_col = 'group_name'
        ,dt_col = 'dt'
        ,bucket_col = 'bucket_id'
        ,dynamic_group = True
        ,print_flag = True
        ,datelimit = 35
        ,**kwargs
):
    """生成实验模型参数，打印查询sql
    Args:
        sql_parttern(str): 查询sql模板, default = None
        group_col(str): 实验分组列名, default = 'group_name'
        dt_col(str): 时间列名, default = 'dt'
        bucket_col(str): 分桶列名, default = 'bucket_id'
        dynamic_group(bool): 是否动态分组, default = True
        print_flag(bool): 是否打印sql, default = True
        datelimit(int): 实验开始时间距离实验结束时间的天数如果大于限制sql只去aa和ab否则取aa开始到ab结束, default = 35
        **kwargs: 其他参数，传入实验参数，可选不传入取数为大盘，
            
    Returns:
        dict: 实验模型参数
    Example:
        >>> p = {
                'world_name': 'w_n_kwai_apps_did_1175'
                ,'exp_name':'eco_test_40_10p'
                ,'group_col' : 'group_name' #分组变量列名
                ,'dt_col' : 'dt' # 时间变量列名
                ,'base_groups' : ['base'] # 对照组组名列表，必须为df中指定的group_col列的子集
                ,'exp_groups' : ['exp3','exp4']
                ,'bucket_col' : 'bucket_id' #实验分桶列名
                ,'dt_AA_start':'2024-10-17'
                ,'dt_AA_end':'2024-10-23'
                ,'dt_AB_start' : '2024-11-01'
                ,'dt_AB_end' : '2024-11-11'
                }
        >>> genExpQueryParam(**p)
    """
    world_name = kwargs.get('world_name')
    exp_name = kwargs.get('exp_name')
    base_groups = kwargs.get('base_groups')
    exp_groups = kwargs.get('exp_groups')
    dt_AA_start = kwargs.get('dt_AA_start')
    dt_AA_end = kwargs.get('dt_AA_end')
    dt_AB_start = kwargs.get('dt_AB_start')
    dt_AB_end = kwargs.get('dt_AB_end')


    # default 
    default_sql_parttern = '''
    {exp_func_import}
    with didexp as
    ( --作为左表
    select   p_date
            ,device_id
            ,1   as dau
            {exp_need_raw}
    from     npcdm.dim_pub_device_daily
    where    {p_date_condition}
    {exp_condition}
    and      func_product = 'KWAI'
    and      explore_locale = 'br'
    and      is_spammer = 0
    and      is_today_active = 1
    and      coalesce(device_id, '') <> ''
    )
    select 
    didexp.p_date as p_date
    {exp_need}
    ,sum(dau) as dau
    from didexp
    group by didexp.p_date
    {exp_need};
    '''
    default_exp_func_import = '''
    add jar viewfs:///home/system/hive/resources/abtest/kuaishou-abtest-udf-latest.jar;
    create temporary function lookupTimedExp as 'com.kuaishou.abtest.udf.LookupTimedExp';
    create temporary function lookupTimedGroup as 'com.kuaishou.abtest.udf.LookupTimedGroup';
    create temporary function lookupBucketId as 'com.kuaishou.abtest.udf.LookupBucketId';
    '''
    default_exp_condition = '''
    and lookupTimedExp( {group_date}, '', '{world_name}', cast(0 as bigint) , device_id ) = '{exp_name}'
    and lookupTimedGroup( {group_date}, '', '{world_name}', cast(0 as bigint) , device_id ) in {group_need}
    '''
    default_exp_need_raw = '''
    ,lookupTimedGroup( {group_date},'','{world_name}',cast(0 as bigint) ,device_id ) as {group_col}
    ,cast(lookupBucketId('{world_name}',device_id,0) as bigint) as {bucket_col}
    '''
    default_p_date_condition = "p_date between '{{ ds_nodash }}' and '{{ ds_nodash }}'"
    default_exp_need = f'''
    ,{group_col}
    ,{bucket_col}
    '''
    default_exp_period = '''
    ,case when p_date between '{pdate_AA_start}' and '{pdate_AA_end}' then 'aa'
    when p_date between '{pdate_AB_start}' and '{pdate_AB_end}' then 'ab'
    end                                                                 as ab_type
    '''
    if not sql_parttern:
        sql_parttern = default_sql_parttern
    if world_name is None or exp_name is None or base_groups is None or exp_groups is None or dt_AB_start is None:
        exp_func_import = ''
        exp_condition = ''
        exp_need_raw = ''
        exp_need = ''
        p_date_condition = default_p_date_condition
    else:
        
        group_need = str(tuple(base_groups + exp_groups))
        pdate_AB_start = dt_AB_start.replace('-','')
        pdate_AB_end = dt_AB_end.replace('-','')
        if dynamic_group:
            group_date = 'p_date'
        else:
            group_date = f'"{pdate_AB_start}"'
        exp_func_import = default_exp_func_import
        exp_condition = default_exp_condition.format(
            group_date = group_date
            ,world_name = world_name
            ,exp_name = exp_name
            ,group_need = group_need
        )
        exp_need_raw = default_exp_need_raw.format(
            group_date = group_date
            ,world_name = world_name
            ,exp_name = exp_name
            ,group_col = group_col
            ,bucket_col = bucket_col
        )
        exp_need = default_exp_need
        if dt_AA_start is None:
            p_date_condition = f"p_date between '{pdate_AB_start}' and '{pdate_AB_end}'"
        else:
            pdate_AA_start = dt_AA_start.replace('-','')
            pdate_AA_end = dt_AA_end.replace('-','')
            exp_need_raw = exp_need_raw + default_exp_period.format(
                pdate_AA_start = pdate_AA_start
                ,pdate_AA_end = pdate_AA_end
                ,pdate_AB_start = pdate_AB_start
                ,pdate_AB_end = pdate_AB_end
            )
            exp_need = exp_need + ',ab_type'
            if len(pd.date_range(dt_AA_start,dt_AB_end)) > datelimit:
                p_date_condition = f"((p_date between '{pdate_AA_start}' and '{pdate_AA_end}') or (p_date between '{pdate_AB_start}' and '{pdate_AB_end}'))"
            else:
                p_date_condition = f"p_date between '{pdate_AA_start}' and '{pdate_AB_end}'"


    sql_query = sql_parttern.format(
        exp_func_import = exp_func_import
        ,exp_condition = exp_condition
        ,exp_need_raw = exp_need_raw
        ,exp_need = exp_need
        ,p_date_condition = p_date_condition
    )
    if print_flag:
        print(sql_query)
    did_param = {
        'group_col' : group_col #分组变量列名
        ,'dt_col' : dt_col # 时间变量列名
        ,'base_groups' : base_groups # 对照组组名列表，必须为df中指定的group_col列的子集
        ,'exp_groups' : exp_groups
        ,'bucket_col' : bucket_col #实验分桶列名
        ,'dt_AA_start':dt_AA_start
        ,'dt_AA_end':dt_AA_end
        ,'dt_AB_start' : dt_AB_start
        ,'dt_AB_end' : dt_AB_end
    }
    return did_param



def expMetricCal(df,metric_cols = []):
    """计算指标(弃用)
    Args:
        df (pandas.core.frame.DataFrame): 原始数据
        metric_cols (list): 指标列表, default = []
    Returns:
        pandas.core.frame.DataFrame: 指标计算结果
    """
    def gen_dt(df):
        return datetime.strftime(datetime.strptime(str(df['p_date']),'%Y%m%d'),'%Y-%m-%d')
    def gen_dau(df):
        return df['dau']
    def gen_dau_40_pct(df):
        return df['dau_40']/df['dau']
    def gen_avg_app_use_duration(df):
        return df['app_use_duration']/df['dau']/60000
    def gen_launch_cnt(df):
        return df['launch_cnt']
    def gen_avg_play_cnt(df):
        return df['play_cnt']/df['dau']
    def gen_mult(df):
        return df['valid_csm_first_tag_num_75th']/df['dau']
    def gen_avg_play_duration(df):
        return df['play_duration']/df['dau']
    def gen_avg_launch_cnt(df):
        return df['launch_cnt']/df['dau']
    def gen_fntr(df):
        return df['complete_play_cnt']/df['play_cnt']
    def gen_evtr(df):
        return df['valid_play_cnt']/df['play_cnt']
    def gen_lvtr(df):
        return df['long_play_cnt']/df['play_cnt']
    def gen_svtr(df):
        return df['short_play_cnt']/df['play_cnt']
    def gen_ltr(df):
        return df['like_cnt']/df['play_cnt']
    def gen_wtr(df):
        return df['follow_cnt']/df['play_cnt']
    def gen_ftr(df):
        return df['share_cnt']/df['play_cnt']
    def gen_dislike_rate(df):
        return df['dislike_cnt']/df['play_cnt']
    def gen_skip115_rate(df):
        return df['skip_1150_play_cnt']/df['play_cnt']
    def gen_neg_vv_rate(df):
        return df['negtive_play_cnt']/df['play_cnt']
    def gen_pos_vv_rate(df):
        return df['v1_positive_play_cnt']/df['play_cnt']
    def gen_pos_vv_ratev2(df):
        return df['v2_positive_play_cnt']/df['play_cnt']
    def gen_lt7(df):
        return df['lt_7d']/df['dau']
    def gen_valid_interest_num_fisrt(df):
        return df['valid_csm_first_tag_num_75th']/df['dau']
    def gen_valid_interest_num_second(df):
        return df['valid_csm_second_tag_num_75th']/df['dau']
    def gen_pssq_neg_rate(df):
        if df['pssq_answer_cnt'] == 0:
            return np.nan
        else:
            return df['pssq_neg_cnt']/df['pssq_answer_cnt']
    def gen_sf_neg_rate(df):
        if df['sf_answer_cnt'] == 0:
            return np.nan
        else:
            return df['sf_neg_cnt']/df['sf_answer_cnt']
    def gen_pstv_feedback_photo_pct(df):
        return df['pstv_feedback_photo_play_cnt']/df['play_cnt']
    def gen_ngtv_feedback_photo_pct(df):
        return df['ngtv_feedback_photo_play_cnt']/df['play_cnt']
    metric_dict = {
        'dt':gen_dt
        ,'dau':gen_dau
        ,'dau_40_pct':gen_dau_40_pct
        ,'avg_app_use_duration':gen_avg_app_use_duration
        ,'launch_cnt':gen_launch_cnt
        ,'avg_play_cnt':gen_avg_play_cnt
        ,'mult':gen_mult
        ,'avg_play_duration':gen_avg_play_duration
        ,'avg_launch_cnt':gen_avg_launch_cnt
        ,'fntr':gen_fntr
        ,'evtr':gen_evtr
        ,'lvtr':gen_lvtr
        ,'svtr':gen_svtr
        ,'ltr':gen_ltr
        ,'wtr':gen_wtr
        ,'ftr':gen_ftr
        ,'dislike_rate':gen_dislike_rate
        ,'skip115_rate':gen_skip115_rate
        ,'neg_vv_rate':gen_neg_vv_rate
        ,'pos_vv_rate':gen_pos_vv_rate
        ,'pos_vv_ratev2':gen_pos_vv_ratev2
        ,'lt7':gen_lt7
        ,'valid_interest_num_fisrt':gen_valid_interest_num_fisrt
        ,'valid_interest_num_second':gen_valid_interest_num_second
        ,'pssq_neg_rate':gen_pssq_neg_rate
        ,'sf_neg_rate':gen_sf_neg_rate
        ,'pstv_feedback_photo_pct':gen_pstv_feedback_photo_pct
        ,'ngtv_feedback_photo_pct':gen_ngtv_feedback_photo_pct
    }
    for i in metric_cols:
        if i in metric_dict.keys():
            df[i] = df.apply(metric_dict.get(i),axis = 1)
    df['dt'] = df.apply(gen_dt,axis = 1)
    return df


def plotab(model, group_agg=True,plot_type= 'relative',base_group=None, mean_cols=[],metric_cols = None):
    """绘制AB测试结果图
    Args:
        model (ABTestBootstrap): 包含AB测试结果的ABTestBootstrap模型
        group_agg (bool): 是否对实验组进行聚合计算，默认True
        plot_type (str): 绘制的图类型，默认‘relative’，可选’absolute’，’percent’
        base_group (str): 对照组名称，默认为None，取model.base_group
        mean_cols (list): 取均值的列，默认为[]
        metric_cols (list): 指标列表，默认为model.metric_cols
    Returns:
        matplotlib.pyplot.figure: AB测试结果图
        matplotlib.pyplot.axes: AB测试结果图的子图
    """
    df_plot = model.raw_df.copy()
    group_col = model.group_col
    dt_col = model.dt_col
    if metric_cols is None:
        metric_cols = model.metric_cols
    exp_groups = model.exp_groups
    dt_AA_start = model.dt_AA_start
    dt_AA_end = model.dt_AA_end 
    dt_AB_start = model.dt_AB_start
    dt_AB_end = model.dt_AB_end


    if group_agg:
        df_plot[group_col] = df_plot[group_col].apply(lambda x: 'EXP' if x in exp_groups else 'BASE')
    group_names = list(df_plot[group_col].unique())
    n_rows = len(metric_cols)
    fig,axes = plt.subplots(n_rows,1,figsize=(30, 10* n_rows ))
    full_dates = pd.date_range(start=dt_AA_start, end=dt_AB_end)
    
    for i, metric_col in enumerate(metric_cols):
        if n_rows == 1:
            ax = axes 
        else:
            ax = axes[i]
        value_cols = [metric_col]
        # 指标变化趋势图
        if plot_type == 'absolute':
            df_tmp = df_plot.pivot_table(index=dt_col, columns=group_col,values=value_cols, aggfunc=np.mean,dropna=False)
            # 分组聚合 基准组 为 BASE
            if group_agg:
                df_tmp[metric_col] = df_tmp[metric_col].transform(lambda x: x - df_tmp[metric_col]['BASE'])
        # 百分比效果趋势图
        elif plot_type == 'relative':
            df_tmp = df_plot.pivot_table(index=dt_col, columns=group_col,values=value_cols, aggfunc=np.mean,dropna=False)
            # 分组聚合 基准组 为 BASE
            if group_agg:
                df_tmp[metric_col] = df_tmp[metric_col].transform(lambda x: x / df_tmp[metric_col]['BASE'] - 1.0)
            # 分组不聚合，选取指定base组做基准组
            else:
                base_group = base_group if base_group else model.base_groups[0]
                if base_group not in model.base_groups + model.exp_groups:
                    raise ValueError('base_group must be in base_groups or exp_groups.')
                df_tmp[metric_col] = df_tmp[metric_col].transform(lambda x: x / df_tmp[metric_col][base_group] - 1.0)
        elif plot_type == 'raw':
            df_tmp = df_plot.pivot_table(index=dt_col, columns=group_col,values=value_cols, aggfunc=np.mean,dropna=False)
            
        else:
            raise ValueError("plot_type should be 'absolute' or 'relative'.")
        df_tmp.index = pd.to_datetime(df_tmp.index)
        df_tmp = df_tmp.reindex(full_dates, fill_value=np.nan)

        custom_legend = []
        for group_name,lc in zip(group_names,sns.color_palette('deep')):
            df_tmp2 = df_tmp.xs(group_name, level=group_col,axis=1)
            df_tmp2['plot_group'] = df_tmp2[metric_col].isna().cumsum()
            df_tmp2 = df_tmp2.reset_index(names = dt_col)
            sns.lineplot(data = df_tmp2.dropna(subset = metric_col),x = dt_col, y = metric_col,hue = 'plot_group',palette = [lc],markers='o',ax=ax)
            custom_legend.append(Line2D([0], [0], color=lc, lw=2, label=group_name))
        ax_xgap = ceil(len(df_tmp2)/20)
        ax.legend(handles=custom_legend, title=group_col)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(ax_xgap))
        if plot_type == 'absolute':
            def pp_formatter(x, pos):
                return f"{x*100:.4f}pp"
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(pp_formatter))
        elif plot_type == 'relative':
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=4))
        
        ax.axvline(pd.to_datetime(dt_AA_start), linestyle='--', color='red')
        ax.axvline(pd.to_datetime(dt_AA_end), linestyle='--', color='red')
        ax.axvline(pd.to_datetime(dt_AB_start), linestyle='--', color='red')
        ax.axvline(pd.to_datetime(dt_AB_end), linestyle='--', color='red')
    return fig,axes
def formatres(df,dimensions = None,sortlist = None,negmetric = None):
    """
    格式化ABtest结果
    Args:
        df (dataframe): 包含ABtest结果的dataframe
        dimensions (list): 维度列表
        sortlist (list): 排序列表
        negmetric (str): 负向指标
    Returns:
        res (dataframe): 格式化后的dataframe
    """
    if dimensions:
        cls = df.pivot(columns = dimensions,index='指标名称',values='AB阶段净提升').applymap(lambda x: -1 if '-' in str(x) else 1)
        cls2 = df.pivot(columns = dimensions,index='指标名称',values='显著性').applymap(lambda x: 1 if '*' in str(x) else 0)
        res = df.pivot(index = '指标名称',columns=dimensions,values='AB阶段净提升')
    else:
        cls = df.pivot(columns='对比分组',index='指标名称',values='AB阶段净提升').applymap(lambda x: -1 if '-' in str(x) else 1)
        cls2 = df.pivot(columns='对比分组',index='指标名称',values='显著性').applymap(lambda x: 1 if '*' in str(x) else 0)
        res = df.pivot(index = '指标名称',columns='对比分组',values='AB阶段净提升')
    cls = cls * cls2
    if negmetric:
        cls.loc[negmetric] = cls.loc[negmetric] * -1
    cls = cls.applymap(lambda x:'redcell' if x > 0 else ('greencell' if x <0 else 'greycell'))
    if sortlist:
        res = res.loc[sortlist,:]
        cls = cls.loc[sortlist,:]
    res_s = res.style\
    .format(lambda x: x if '%' in str(x) else '{:.4f}pp'.format(x*100))\
    .set_td_classes(cls)\
    .set_table_styles([
            {'selector': '.redcell', 'props': 'color:red;'}
            ,{'selector': '.greencell', 'props': 'color:green;'}
            ,{'selector': '.greycell', 'props': 'color:grey;'}
            ,{'selector':'thead','props':[('background-color','#D1F2FF'),('text-align','center')]}                                                                 
            ,{'selector':'td,th','props':[('border','1px solid black')]}                                                                  
            ,{'selector':'tbody,td,th.row_heading','props':[('text-align','right')]}                                                            
            ,{'selector':'td','props':[('text-align','center')]}
    ]).set_table_attributes('style="border-collapse: collapse;"')
    return res_s




if __name__ == '__main__':
    # param1 = {
    #     'world_name': 'w_n_kwai_apps_did_1175'
    #     ,'exp_name':'eco_test_40_10p'
    #     ,'group_col' : 'group_name' #分组变量列名
    #     ,'dt_col' : 'dt' # 时间变量列名
    #     ,'base_groups' : ['base'] # 对照组组名列表，必须为df中指定的group_col列的子集
    #     ,'exp_groups' : ['exp3','exp4']
    #     ,'bucket_col' : 'bucket_id' #实验分桶列名
    #     ,'dt_AA_start':'2024-10-17'
    #     ,'dt_AA_end':'2024-10-23'
    #     ,'dt_AB_start' : '2024-11-01'
    #     ,'dt_AB_end' : '2024-11-11'
    # }
    url = 'testurl'
    param2 = urlToParam(url)
    df_s = toTable(url)
    did_param = genExpQueryParam(**param2)

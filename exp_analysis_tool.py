
from urllib.parse import unquote,parse_qs
from datetime import datetime,timedelta

import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_url_params(url):
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
    def timeParser(value):
        if not isinstance(value,int):
            value = int(value)
        if len(str(value)) == 13:
            value = value/1000
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

def genExpQueryParam(world_name,exp_name,group_col,dt_col,base_groups,exp_groups,bucket_col,dt_AA_start,dt_AA_end,dt_AB_start,dt_AB_end,sql_parttern = None):
    '''
    param = {
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
    '''
    if not sql_parttern:
        sql_parttern = '''
        add jar viewfs:///home/system/hive/resources/abtest/kuaishou-abtest-udf-latest.jar;
        create temporary function lookupTimedExp as 'com.kuaishou.abtest.udf.LookupTimedExp';
        create temporary function lookupTimedGroup as 'com.kuaishou.abtest.udf.LookupTimedGroup';
        create temporary function lookupBucketId as 'com.kuaishou.abtest.udf.LookupBucketId';
        with didexp as
        ( --作为左表
         select   p_date
                 ,device_id
                 ,lookupTimedGroup( '{pdate_AB_start}','','{world_name}',cast(0 as bigint) ,device_id ) as group_name
                 ,cast(lookupBucketId('{world_name}',device_id,0) as bigint) as bucket_id
                 ,1                                                           as dau
         from     npcdm.dim_pub_device_daily
         where    (p_date between '{pdate_AA_start}' and '{pdate_AA_end}' or p_date between '{pdate_AB_start}' and '{pdate_AB_end}')
         and      lookupTimedExp( '{pdate_AB_start}', '', '{world_name}', cast(0 as bigint) , device_id ) = '{exp_name}'
         and      lookupTimedGroup( '{pdate_AB_start}', '', '{world_name}', cast(0 as bigint) , device_id ) in {group_need}
         and      func_product = 'KWAI'
         and      explore_locale = 'br'
         and      is_spammer = 0
         and      is_today_active = 1
         and      coalesce(device_id, '') <> ''
        )

        '''
    pdate_AA_start = dt_AA_start.replace('-','')
    pdate_AA_end = dt_AA_end.replace('-','')
    pdate_AB_start = dt_AB_start.replace('-','')
    pdate_AB_end = dt_AB_end.replace('-','')
    group_need = str(tuple(base_groups + exp_groups))
    sql_query = sql_parttern.format(
        world_name = world_name
        ,exp_name = exp_name
        ,pdate_AA_start = pdate_AA_start
        ,pdate_AA_end = pdate_AA_end
        ,pdate_AB_start = pdate_AB_start
        ,pdate_AB_end = pdate_AB_end
        ,group_need = group_need
    )
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
    print(sql_query)
    return did_param


def expMetricCal(df,metric_cols = ['dau','launch_cnt','app_use_duration','avg_app_use_duration','avg_play_cnt']):
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
    def gen_lt7(df):
        return df['lt_7d']/df['dau']
    def gen_valid_interest_num_fisrt(df):
        return df['valid_interest_num_fisrt']/df['dau']
    def gen_valid_interest_num_second(df):
        return df['valid_interest_num_second']/df['dau']
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
        ,'lt7':gen_lt7
        ,'valid_interest_num_fisrt':gen_valid_interest_num_fisrt
        ,'valid_interest_num_second':gen_valid_interest_num_second
    }
    for i in metric_cols:
        if i in metric_dict.keys():
            df[i] = df.apply(metric_dict.get(i),axis = 1)
    df['dt'] = df.apply(gen_dt,axis = 1)
    return df


def plotab(model, group_agg=True,plot_type= 'relative',base_group=None, mean_cols=[]):
    df_plot = model.df.copy()
    dt_min = df_plot[model.dt_col].min()
    if group_agg:
        df_plot[model.group_col] = df_plot[model.group_col].apply(lambda x: 'EXP' if x in model.exp_groups else 'BASE')
    group_names = list(df_plot[model.group_col].unique())
    n_rows = len(model.metric_cols)
    def timeArea(value):
        value = datetime.strptime(value,format('%Y-%m-%d'))
        aa_start = datetime.strptime(model.dt_AA_start,format('%Y-%m-%d'))
        aa_end = datetime.strptime(model.dt_AA_end,format('%Y-%m-%d'))
        ab_start = datetime.strptime(model.dt_AB_start,format('%Y-%m-%d'))
        ab_end = datetime.strptime(model.dt_AB_end,format('%Y-%m-%d'))
        if value >= aa_start and value <= aa_end:
            return 'AA'
        elif value >= ab_start and value <= ab_end:
            return 'AB'
    start_date = datetime.strptime(model.dt_AA_start,format('%Y-%m-%d'))
    end_date = datetime.strptime(model.dt_AB_end,format('%Y-%m-%d'))
    tlist = [model.dt_AA_start]
    while start_date + timedelta(days=1) <= end_date:
        tlist.append(datetime.strftime(start_date + timedelta(days=1),format = '%Y-%m-%d'))
        start_date = start_date + timedelta(days=1)
    tdf = pd.DataFrame(tlist,columns=[model.dt_col])

    df_plot = tdf.merge(df_plot,on = 'dt',how='left')
    fig,axes = plt.subplots(n_rows,1,figsize=(30, 10* n_rows ))
    for i, metric_col in enumerate(model.metric_cols):
        value_cols = [metric_col]
        # 指标变化趋势图
        if plot_type == 'absolute':
            if metric_col in mean_cols:
                df_tmp = df_plot.pivot_table(index=model.dt_col, columns=model.group_col,
                                                values=value_cols, aggfunc=np.mean)
            else:
                df_tmp = df_plot.pivot_table(index=model.dt_col, columns=model.group_col,
                                                values=value_cols, aggfunc=np.sum)
        # 百分比效果趋势图
        elif plot_type == 'relative':
            df_tmp = df_plot.pivot_table(index=model.dt_col, columns=model.group_col,values=value_cols, aggfunc=np.mean,dropna=False)
            # 分组聚合 基准组 为 BASE
            if group_agg:
                df_tmp[metric_col] = df_tmp[metric_col].transform(lambda x: x / df_tmp[metric_col]['BASE'] - 1.0)
            # 分组不聚合，选取指定base组做基准组
            else:
                base_group = base_group if base_group else model.base_groups[0]
                if base_group not in model.base_groups + model.exp_groups:
                    raise ValueError('base_group must be in base_groups or exp_groups.')
                df_tmp[metric_col] = df_tmp[metric_col].transform(lambda x: x / df_tmp[metric_col][base_group] - 1.0)
        else:
            raise ValueError("plot_type should be 'absolute' or 'relative'.")
        df_tmp = df_tmp.reset_index().sort_values(by = model.dt_col)
        df_tmp['period'] = df_tmp[model.dt_col].apply(timeArea)
        df_tmp['dt'] = df_tmp['dt'].apply(lambda x: datetime.strftime(datetime.strptime(x,format('%Y-%m-%d')),format = '%m-%d' ))
        # print(df_tmp)
        # sns.lineplot(x = df_tmp[model.dt_col],y = df_tmp[metric_col][np.nan],ax=axes[i])
        for group_name,lc in zip(group_names,sns.color_palette('deep')):
            for p in ['AA','AB']:
                df_tmp2 = df_tmp[df_tmp['period'] == p]
                if p == 'AA':
                    sns.lineplot(x = df_tmp2[model.dt_col],y = df_tmp2[metric_col][group_name],label = group_name,color = lc,ax=axes[i])
                else:
                    sns.lineplot(x = df_tmp2[model.dt_col],y = df_tmp2[metric_col][group_name],color = lc,ax=axes[i])
        axes[i].set_ylabel(metric_col,fontsize = 40)
        axes[i].legend(fontsize = 40)
    return df_tmp
def formatres(df,dimensions = None,sortlist = None,negmetric = None):
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

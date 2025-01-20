from datetime import datetime
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
    }
    for i in metric_cols:
        if i in metric_dict.keys():
            df[i] = df.apply(metric_dict.get(i),axis = 1)
    df['dt'] = df.apply(gen_dt,axis = 1)
    return df

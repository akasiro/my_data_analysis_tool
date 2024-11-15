


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
if __name__ == '__main__':
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
    did_param = genExpQueryParam(**param)

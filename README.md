# DA分析代码工具包

## 导入包  

用于导入常用包可以使用魔术语句  

```python
%load {path}/my_data_analysis_tool/import_help.py
```

## 实验分析工具模块  

可以导入使用  

```python 
from my_data_analysis_tool import exp_analysis_tool as myeal
```
  
  
| 函数 | 功能 | 输入值 | 输出值|
| -- | -- | -- | -- |
| urlToParam | url解析为参数 |url | 参数字典|
| genExpQueryParam | 将参数字典解析为内部函数参数字典，打印取数代码 |参数字典 | 参数字典 | 
| toTable | url格式化为表格输出 | url | 格式化的表格 |
| expMetricCal | 对数据表所需要的指标进行计算 | df; 需要的指标list | df |
| plotab | 对模型使用的表格输出趋势图 | 模型 | 数据图 | 
| formatres | 对模型结果进行格式化 | df | 格式化的df | 
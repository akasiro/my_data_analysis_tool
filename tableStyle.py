import pandas as pd
import seaborn as sns
def tableStyle(s):
    def topp(value):# 转pp
        return '{:.2f}pp'.format(value *100)
    def highlight_specific_text(s,text,color): #突出显示
        return ['background-color: {}'.format(color) if text == str(v) else '' for v in s]
    
    s.hide()                                                                                   # hide index
    # s.hide(axis = 'columns')                                                                 # hide columns         
    s.format(topp,subset = ['Score'])                                                          # transform float to pp                
    s.format('{:.2%}',subset = ['s'])                                                          # transform float to percent                
    s.apply(highlight_specific_text,text = 'Bob', color = 'pink')                              # highlight specific                                             
    s.background_gradient(cmap=sns.light_palette("red", as_cmap=True),subset=['s2'])           # background_gradient                                                               
    s.text_gradient(cmap='bwr',subset=['s'])                                                   # text gradient                       
    s.bar(color='pink',width= 50,height= 50,subset = ['Score'])                                # bar in table                                         
    s.set_table_styles([                                                                          
        {'selector':'thead','props':[('background-color','#D1F2FF'),('text-align','center')]}  # thead                                                                        
        ,{'selector':'td,th','props':[('border','1px solid black')]}                           # borders                                              
        ,{'selector':'tbody,td,th.row_heading','props':[('text-align','right')]}               #                                                            
        ,{'selector':'td','props':[('text-align','center')]}                                   #  text-align                                      
    ]).set_table_attributes('style="border-collapse: collapse;"')                              #  border attr                                           
    return s

if __name__ == '__main__':
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [0.85, 0.90, 0.95],
        's':[1,2,-3]
        ,'s2':[1,2,3]
    }
    df = pd.DataFrame(data)
    df_s = df.style.pipe(tableStyle)
    df_s

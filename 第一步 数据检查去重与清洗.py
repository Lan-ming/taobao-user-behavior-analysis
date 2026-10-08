import pandas as pd
# 把抽样后的 taobao_sample.csv 放在同一文件夹即可运行
DATA_FILE = 'taobao_sample.csv'
df = pd.read_csv(DATA_FILE)
print(f"该数据的总行数为: {len(df)}")
print("该数据的列名为: ",df.columns)
print("该数据的行数为 :",df.head)


# =========数据清洗==========
behavior_map={1:"点击",2:"收藏",3:"加购",4:"支付"}
df['behavior']=df['behavior_type'].map(behavior_map)
# 将日期变为时间格式 便于后续的计算
df['time']=pd.to_datetime(df['time'])
df['hour']=df['time'].dt.hour
df['date']=df['time'].dt.date
print("========各行为数量=========")
print(df['behavior'].value_counts())


print("=======数据检查=======")
print(f"该数据每列的缺失值: {df.isnull().sum()}")
print(df[['user_id', 'item_id', 'behavior', 'date', 'hour']].head(5))#数据这五列的前五行行数
print(f"该数据的(行,列):{df.shape}")


#去除重复值 当这五列一模一样时
#============数据去重==============
before=len(df)
df=df.drop_duplicates(subset=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'])#删除这五个都相同时候的重复值
after=len(df)
print(f"去重之前的数据{before}行")
print(f"去重之后的数据{after}行")
print(f"去重的数据为{before-after}行")


#=======数据清洗成功 已干净==============
df.to_csv('clean.csv', index=False)

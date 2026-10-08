'''
留存 = 用户第一天来了，第 N 天还会不会回来。
算三步：
    1. 找每个用户的首次活跃日（他哪天第一次出现）
    2. 算每条行为记录距首次活跃日过了几天
    3. 统计第 1 天、第 3 天、第 7 天 、第14天、第30天 分别有多少用户还活跃
'''
import pandas as pd
df=pd.read_csv('clean.csv')
#每个用户最早来的时间
df['date']=pd.to_datetime(df['date'])
first_day =df.groupby('user_id')['date'].min().reset_index()
first_day.columns=['user_id','first_date']
#合并到  user_id 上
df=df.merge(first_day,on='user_id')
df['day_n']=(df["date"]-df["first_date"]).dt.days

end_date = df['date'].max()
for n in [1,3,7,14,30]:
    cutoff = end_date - pd.Timedelta(days=n)
    eligible = first_day[first_day['first_date'] <= cutoff]['user_id']
    total = eligible.nunique()
    active = df[(df['day_n'] == n) & df['user_id'].isin(eligible)]['user_id'].nunique()
    print(f"第{n}日留存：{active / total * 100:.2f}%（{active}/{total}人）")


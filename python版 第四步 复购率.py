'''
复购 = 买了一次之后还会不会再买。算三步：
1. 只看支付行为
2. 统计每个用户支付了多少天（同一天买多次只算 1 天）
3. 支付天数 ≥ 2 天的就是复购用户
'''
import pandas as pd
df=pd.read_csv('clean.csv')
df['date']=pd.to_datetime(df['date'])

#1.支付行为
buy=df[df['behavior']=='支付']
#2. 统计每个用户支付了多少天（同一天买多次只算 1 天）
buy_days=buy.groupby('user_id')['date'].nunique().reset_index()
buy_days.columns = ['user_id', 'pay_days']
#3. 支付天数 ≥ 2 天的就是复购用户
Repurchase=buy_days['pay_days']>=2
repurchase_users = buy_days[Repurchase]
#支付的人数
buy_people=len(buy_days)
#复购的人数
Repurchase_people=len(repurchase_users)
#复购率
repeat_rate=Repurchase_people/buy_people

print(f"付款的用户为: {buy_people}人")
print(f'复购用户（支付≥2天)为: {Repurchase_people}人，\n复购率为:{repeat_rate*100:.2f}%')
print(f"只购买一次的用户为: {(buy_people-Repurchase_people)}人")
print(f'人均支付天数为: {buy_days['pay_days'].mean():.2f}天')
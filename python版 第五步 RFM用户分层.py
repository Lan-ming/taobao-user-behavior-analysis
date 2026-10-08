import pandas as pd
df=pd.read_csv('clean.csv')
df['date']=pd.to_datetime(df['date'])

# RFM
# R   Recency 最近一次消费距今天数
# F   Frequency    消费频次
# M   Monetary     消费金额

# 首先计算 R、F
#在支付的行为里
buy=df[df['behavior']=='支付']

#快照日 =最后一天+1
Snapshot_Date=buy['date'].max()+pd.Timedelta(days=1)
#计算R、F
rfm=buy.groupby('user_id').agg(
# R：这个用户最后一次支付日期，距离"今天"隔了多少天
    recency=('date',lambda s: (Snapshot_Date - s.max()).days),
# F：这个用户一共有多少天支付过（去重数日期）
    frequency=('date', 'nunique')
)
#把所有人按R从小到大排队，平均切成5档，人数相等
# R最小的一档给5分，最大的一档给1分（所以标签是5,4,3,2,1）
rfm['r_score'] = pd.qcut(
    rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]
).astype(int)

# 把所有人按F从大到小排队，平均切成5档
# F最大的一档给5分，最小的一档给1分（标签是1,2,3,4,5）
rfm['f_score'] = pd.qcut(
    rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]
).astype(int)

# 按分数把人分成4类
def segment(row):
    r_high = row['r_score'] >= 3      # R≥3分 = 最近还在买
    f_high = row['f_score'] >= 3      # F≥3分 = 买得勤
    if r_high and f_high:
        return '重要价值客户'          # 最近买 + 买得勤
    if r_high and not f_high:
        return '重要发展客户'          # 最近买 + 买得少
    if not r_high and f_high:
        return '重要保持客户'          # 很久没买 + 以前买得勤
    return '重要挽留客户'              # 很久没买 + 买得少

# apply = 把上面这个规则，套到每一行（每个用户）身上
rfm['客户分层'] = rfm.apply(segment, axis=1)

#统计每一类有多少人
summary = rfm.groupby('客户分层').size().reset_index(name='用户数')

# 加上"浏览未购潜客"
# 这些人有行为记录，但从来没支付过，不在rfm表里
purchasers = set(rfm.index)                          # 所有买过的用户id
n_potential = df['user_id'].nunique() - len(purchasers)  # 总用户 - 买过的 = 没买的

summary = pd.concat([
    summary,
    pd.DataFrame([{'客户分层': '浏览未购潜客', '用户数': n_potential}])
], ignore_index=True)

# 算占比、排序、打印
summary['占比%'] = (summary['用户数'] / summary['用户数'].sum() * 100).round(2)
summary = summary.sort_values('用户数', ascending=False).reset_index(drop=True)

print(summary)
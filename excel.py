import pandas as pd
import numpy as np

df = pd.read_csv('clean.csv')
df['date'] = pd.to_datetime(df['date'])
snapshot = df['date'].max() + pd.Timedelta(days=1)

# ===== 准备3张表（都是前面学过的，快速重算） =====

# 表1：RFM分层
pay = df[df['behavior'] == '支付']
rfm = pay.groupby('user_id').agg(
    recency=('date', lambda s: (snapshot - s.max()).days),
    frequency=('date', 'nunique'))
rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
def seg(r):
    rh = r['r_score'] >= 3; fh = r['f_score'] >= 3
    if rh and fh: return '重要价值客户'
    if rh: return '重要发展客户'
    if fh: return '重要保持客户'
    return '重要挽留客户'
rfm['客户分层'] = rfm.apply(seg, axis=1)
summary = rfm.groupby('客户分层').size().reset_index(name='用户数')
n_pot = df['user_id'].nunique() - len(rfm)
summary = pd.concat([summary, pd.DataFrame([{'客户分层': '浏览未购潜客', '用户数': n_pot}])], ignore_index=True)
summary['占比%'] = (summary['用户数'] / summary['用户数'].sum() * 100).round(2)

# 表2：24小时分布
hourly = df.pivot_table(index='hour', columns='behavior', values='user_id', aggfunc='size', fill_value=0)
hourly = hourly[['点击', '收藏', '加购', '支付']]
hourly['点击支付率%'] = (hourly['支付'] / hourly['点击'] * 100).round(2)
hourly = hourly.reset_index()

# 表3：类目TOP15
cat = df.pivot_table(index='item_category', columns='behavior', values='user_id', aggfunc='size', fill_value=0)
cat['点击支付率%'] = (cat['支付'] / cat['点击'] * 100).round(2)
cat = cat.sort_values('支付', ascending=False).head(15).reset_index()

# ===== 核心：写入 Excel，一个 sheet 一张表 =====
with pd.ExcelWriter('我的报表.xlsx', engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='RFM分层', index=False)
    hourly.to_excel(writer, sheet_name='时段分析', index=False)
    cat.to_excel(writer, sheet_name='类目TOP15', index=False)

print('✅ 报表已生成')

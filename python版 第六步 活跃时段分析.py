import pandas as pd

df = pd.read_csv('clean.csv')
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek + 1
df['is_weekend'] = df['weekday'].isin([6, 7]).astype(int)

# ========== 第1块：24小时，每个小时的点击/支付量 + 点击支付率 ==========
# pivot_table 透视表：行=小时，列=四种行为，格子里=数量
hourly = df.pivot_table(index='hour', columns='behavior',
                        values='user_id', aggfunc='size', fill_value=0)
hourly = hourly[['点击', '收藏', '加购', '支付']]

# 每个小时的点击支付率 = 这个小时的支付 ÷ 这个小时的点击
hourly['点击支付率%'] = (hourly['支付'] / hourly['点击'] * 100).round(2)
print(hourly)

# ========== 第2块：工作日 vs 周末，分小时的"日均"活跃量 ==========
# 先按 [是否周末, 小时] 两个维度一起分组数数量
grp = df.groupby(['is_weekend', 'hour']).size().rename('cnt').reset_index()

# 工作日有多少天、周末有多少天（两类天数不一样！）
days = df.groupby('is_weekend')['date'].nunique()

# 日均 = 总量 ÷ 各自的天数（不除天数直接比总量不公平）
grp['日均'] = grp.apply(lambda r: r['cnt'] / days[r['is_weekend']], axis=1)

# 再转成：行=小时，两列分别是工作日日均、周末日均
wkd = grp.pivot(index='hour', columns='is_weekend', values='日均').fillna(0)
wkd.columns = ['工作日日均', '周末日均']
print(wkd)

# ========== 第3块：一周各天的行为总量 ==========
dow = df.groupby('weekday').size().rename('行为量').reset_index()
dow['星期'] = dow['weekday'].map({
    1: '周一', 2: '周二', 3: '周三', 4: '周四', 5: '周五', 6: '周六', 7: '周日'
})
print(dow[['星期', '行为量']])

# ========== 结论 ==========
print(f"支付高峰：{hourly['支付'].idxmax()}点")
print(f"点击支付率最高：{hourly['点击支付率%'].idxmax()}点")

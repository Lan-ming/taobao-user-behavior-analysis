import pandas as pd
import numpy as np

df = pd.read_csv('clean.csv')

# ========== 第1块：每个类目的行为量 + 转化率 ==========
# 透视表：行=类目，列=行为，格子=数量
cat = df.pivot_table(index='item_category', columns='behavior',
                     values='user_id', aggfunc='size', fill_value=0)
# 单独算每个类目触达了多少用户，拼到cat上
users = df.groupby('item_category')['user_id'].nunique().rename('触达用户')
cat = cat.join(users).fillna(0)

cat['加购率%'] = (cat['加购'] / cat['点击'] * 100).round(2)
cat['点击支付率%'] = (cat['支付'] / cat['点击'] * 100).round(2)
# 加购支付比：加购为0的先换成空值，防止除以0报错
cat['加购支付比%'] = (cat['支付'] / cat['加购'].replace(0, np.nan) * 100).round(1)
cat = cat.sort_values('支付', ascending=False)

print(cat[['点击', '加购', '支付', '触达用户', '点击支付率%', '加购支付比%']].head(10))

# ========== 第2块：帕累托（二八法则） ==========
sorted_buy = cat['支付'].sort_values(asc=False) if False else cat['支付'].sort_values(ascending=False)
# cumsum = 累计求和：第1名+第2名+...一路加下去
pareto = (sorted_buy.cumsum() / sorted_buy.sum() * 100)
n80 = int((pareto <= 80).sum() + 1)   # 累计到80%时，用了多少个类目
print(f"共 {len(cat)} 个类目；前 {n80} 个（{n80/len(cat):.1%}）贡献约80%支付")

# ========== 第3块：高加购低转化问题类目 ==========
thr = cat['加购'].quantile(0.8)          # 加购量的80分位数（前20%才算"加购多"）
med_ratio = cat['加购支付比%'].median()   # 所有类目加购支付比的中位数
# 两个条件同时满足：加购够多 & 转化比中位数还差
problem = cat[(cat['加购'] >= thr) & (cat['加购支付比%'] < med_ratio)]
problem = problem.sort_values('加购', ascending=False).head(15)
print(problem[['加购', '支付', '加购支付比%']])

# ========== 第4块：支付量 TOP10 商品 ==========
item = df[df['behavior_type'] == 4].groupby(['item_id', 'item_category']).size()
item = item.rename('支付次数').reset_index().sort_values('支付次数', ascending=False).head(10)
print(item)

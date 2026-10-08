# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT = 'charts'
os.makedirs(OUT, exist_ok=True)

# ===== 全局中文设置（开头一次，全局生效） =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ===== 读数据 =====
df = pd.read_csv('clean.csv')
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek + 1
df['is_weekend'] = df['weekday'].isin([6, 7]).astype(int)
SNAPSHOT = pd.Timestamp('2014-12-19')
END = pd.Timestamp('2014-12-18')



# 统一颜色
C = {'点击': '#4C78A8', '收藏': '#54A24B', '加购': '#F58518', '支付': '#E45756'}
BLUE, ORANGE, RED, GREEN, GRAY = '#4C78A8', '#F58518', '#E45756', '#54A24B', '#9AA0A6'

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# ============================================================
# 模块一：漏斗 → 图1、图2
# ============================================================
ev = df['behavior_type'].value_counts()
pv, fav, cart, buy = (int(ev.get(k, 0)) for k in (1, 2, 3, 4))
event_funnel = pd.DataFrame([('点击', pv), ('加购', cart), ('支付', buy)],
                            columns=['漏斗环节', '事件次数'])
event_funnel['环节转化率%'] = [100.0, round(cart/pv*100, 2), round(buy/cart*100, 2)]
event_funnel['相对点击转化%'] = (event_funnel['事件次数']/pv*100).round(2)

user_bh = df.groupby('user_id')['behavior_type'].agg(lambda s: set(s))
total_u = len(user_bh)
user_funnel = pd.DataFrame([
    (name, sum(b in s for s in user_bh), round(sum(b in s for s in user_bh)/total_u*100, 2))
    for b, name in ((1, '点击'), (2, '收藏'), (3, '加购'), (4, '支付'))
], columns=['行为', '去重用户数', '占总用户%'])

def user_did(b): return user_bh.map(lambda s: b in s)
fav_u, cart_u, buy_u = user_did(2), user_did(3), user_did(4)
path_cmp = pd.DataFrame([
    ('事件次数（万次）', round(cart/10000, 2), round(fav/10000, 2)),
    ('去重用户数（人）', int(cart_u.sum()), int(fav_u.sum())),
    ('有支付比例%', round((cart_u & buy_u).sum()/cart_u.sum()*100, 2),
                   round((fav_u & buy_u).sum()/fav_u.sum()*100, 2)),
], columns=['对比项', '加购路径', '收藏路径'])

# 图1：事件口径漏斗（横条）
fig, ax = plt.subplots(figsize=(8, 4.2))
d = event_funnel.sort_values('事件次数')
bars = ax.barh(d['漏斗环节'], d['事件次数'], color=[C[x] for x in d['漏斗环节']], height=0.6)
vmax = d['事件次数'].max()
for b, row in zip(bars, d.itertuples()):
    ax.text(row.事件次数 + vmax*0.01, b.get_y()+b.get_height()/2,
            f'{row.事件次数:,}次（环节{row._3}%）', va='center', fontsize=9)
ax.set_title('转化漏斗（事件口径，点击→支付 %.2f%%）'
             % event_funnel.set_index('漏斗环节').loc['支付', '相对点击转化%'])
ax.set_xlim(0, vmax*1.28); ax.grid(axis='x', linestyle='--', alpha=0.3)
save('01_用户转化漏斗.png')

# 图2：加购 vs 收藏 三面板
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
for i, t in enumerate(['事件次数（万次）', '去重用户数（人）', '有支付比例%']):
    ax = axes[i]
    vals = [path_cmp.iloc[i]['加购路径'], path_cmp.iloc[i]['收藏路径']]
    bars = ax.bar(['加购', '收藏'], vals, color=[ORANGE, GREEN], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v}', ha='center', va='bottom', fontsize=9.5)
    ax.set_title(t); ax.set_ylim(0, max(vals)*1.2); ax.grid(axis='y', linestyle='--', alpha=0.3)
fig.suptitle('加购 vs 收藏 路径对比', fontsize=13)
save('02_路径转化率对比.png')

# ============================================================
# 模块二：留存 → 图3、图4
# ============================================================
first = df.groupby('user_id')['date'].min().rename('cohort').reset_index()
active = df[['user_id', 'date']].drop_duplicates().merge(first, on='user_id')
active['gap'] = (active['date']-active['cohort']).dt.days
rows = []
for g in range(31):
    cutoff = END - pd.Timedelta(days=g)
    elig = first[first['cohort'] <= cutoff]['user_id']
    n = elig.nunique()
    kept = active[(active['gap'] == g) & active['user_id'].isin(elig)]['user_id'].nunique()
    rows.append((g, n, kept, round(kept/n*100, 2) if n else 0))
curve = pd.DataFrame(rows, columns=['距首次活跃天数', '可观测用户数', '留存用户数', '留存率%'])
piv = active.pivot_table(index='cohort', columns='gap', values='user_id', aggfunc='nunique')
cohort_pct = piv.div(piv[0], axis=0)*100
cohort_pct.index = cohort_pct.index.strftime('%m-%d')

# 图3：留存曲线
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(curve['距首次活跃天数'], curve['留存率%'], color=BLUE, marker='o', ms=3.5)
for g in (1, 3, 7, 14, 30):
    v = curve.loc[curve['距首次活跃天数'] == g, '留存率%'].iloc[0]
    ax.annotate(f'{g}日 {v}%', (g, v), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=8.5)
ax.set_title('用户留存曲线'); ax.set_xlabel('距首次活跃天数'); ax.set_ylabel('留存率%')
ax.set_xticks(range(0, 31, 2)); ax.grid(linestyle='--', alpha=0.3)
save('03_留存曲线.png')

# 图4：同期群热力图（前15天）
fig, ax = plt.subplots(figsize=(9, 6.5))
sns.heatmap(cohort_pct.iloc[:, :15], annot=True, fmt='.0f', cmap='Blues',
            cbar_kws={'label': '留存率%'}, linewidths=0.3, linecolor='white',
            ax=ax, annot_kws={'size': 7})
ax.set_title('同期群留存热力图（%）'); ax.set_xlabel('第N天'); ax.set_ylabel('首次活跃日')
plt.xticks(rotation=0); plt.yticks(rotation=0)
save('04_同期群留存热力图.png')

# ============================================================
# 模块三：RFM → 图5、图6
# ============================================================
payp = df[df['behavior_type'] == 4]
rfm = payp.groupby('user_id').agg(
    recency=('date', lambda s: (SNAPSHOT-s.max()).days),
    frequency=('date', 'nunique'),
    pay_events=('item_id', 'size'))
rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
flev = min(5, rfm['frequency'].nunique())
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), flev,
                         labels=list(range(1, flev+1))).astype(int)
def segf(r):
    rh = r['r_score'] >= 3; fh = r['f_score'] >= 3
    if rh and fh: return '重要价值客户'
    if rh: return '重要发展客户'
    if fh: return '重要保持客户'
    return '重要挽留客户'
rfm['客户分层'] = rfm.apply(segf, axis=1)
n_pot = df['user_id'].nunique()-len(rfm)
summary = rfm.groupby('客户分层').size().rename('用户数').reset_index()
summary = pd.concat([summary, pd.DataFrame([{'客户分层': '浏览未购潜客', '用户数': n_pot}])], ignore_index=True)
summary['占比%'] = (summary['用户数']/summary['用户数'].sum()*100).round(2)
seg_color = {'重要价值客户': '#E45756', '重要发展客户': '#F58518', '重要保持客户': '#4C78A8',
             '重要挽留客户': '#B279A2', '浏览未购潜客': '#9AA0A6'}

# 图5：RFM分层（横条）
fig, ax = plt.subplots(figsize=(8, 4.3))
d = summary.sort_values('用户数')
bars = ax.barh(d['客户分层'], d['用户数'], color=[seg_color[x] for x in d['客户分层']])
for b, v, p in zip(bars, d['用户数'], d['占比%']):
    ax.text(v+d['用户数'].max()*0.01, b.get_y()+b.get_height()/2,
            f'{v}（{p}%）', va='center', fontsize=9)
ax.set_title('RFM用户分层（含潜客）'); ax.set_xlim(0, d['用户数'].max()*1.2)
ax.grid(axis='x', linestyle='--', alpha=0.3)
save('05_RFM用户分层.png')

# 图6：R-F散点
fig, ax = plt.subplots(figsize=(7.5, 4.6))
smp = rfm.sample(min(3000, len(rfm)), random_state=2)
for name, g in smp.groupby('客户分层'):
    ax.scatter(g['recency'], g['frequency'], s=10, alpha=0.55,
               color=seg_color[name], label=name)
ax.set_title('支付用户 R-F 分布'); ax.set_xlabel('R：距最近支付天数'); ax.set_ylabel('F：支付天数')
ax.legend(fontsize=8, markerscale=1.5); ax.grid(linestyle='--', alpha=0.3)
save('06_RF分布散点.png')

# ============================================================
# 模块四：时段 → 图7、图8
# ============================================================
hourly = df.pivot_table(index='hour', columns='behavior', values='user_id',
                        aggfunc='size', fill_value=0)
for c in ('点击', '收藏', '加购', '支付'):
    if c not in hourly: hourly[c] = 0
hourly = hourly[['点击', '收藏', '加购', '支付']]
hourly['点击支付率%'] = (hourly['支付']/hourly['点击']*100).round(3)
grp = df.groupby(['is_weekend', 'hour']).size().rename('cnt').reset_index()
days = df.groupby('is_weekend')['date'].nunique()
grp['日均'] = grp.apply(lambda r: r['cnt']/days[r['is_weekend']], axis=1)
wkd = grp.pivot(index='hour', columns='is_weekend', values='日均').fillna(0)
wkd.columns = ['工作日日均', '周末日均']
dow = df.groupby('weekday').size().rename('行为量').reset_index()
dow['星期'] = dow['weekday'].map({1:'周一',2:'周二',3:'周三',4:'周四',5:'周五',6:'周六',7:'周日'})

# 图7：24小时（双轴 + 支付率）
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(hourly.index, hourly['点击'], color=C['点击'], marker='o', ms=3, label='点击')
ax2 = ax.twinx()
ax2.plot(hourly.index, hourly['支付'], color=C['支付'], marker='s', ms=3, label='支付')
ax.set_title('24小时 点击与支付'); ax.set_xlabel('小时')
ax.set_ylabel('点击量', color=C['点击']); ax2.set_ylabel('支付量', color=C['支付'])
ax.set_xticks(range(0, 24, 2)); ax.grid(linestyle='--', alpha=0.3)
ax = axes[1]
ax.plot(hourly.index, hourly['点击支付率%'], color=ORANGE, marker='o', ms=3.5)
ax.set_title('24小时 点击支付率'); ax.set_xlabel('小时'); ax.set_ylabel('支付/点击 %')
ax.set_xticks(range(0, 24, 2)); ax.grid(linestyle='--', alpha=0.3)
save('07_24小时活跃规律.png')

# 图8：工作日周末 + 星期
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(wkd.index, wkd['工作日日均'], color=BLUE, marker='o', ms=3, label='工作日')
ax.plot(wkd.index, wkd['周末日均'], color=GREEN, marker='s', ms=3, label='周末')
ax.set_title('工作日 vs 周末 分小时日均'); ax.set_xlabel('小时'); ax.set_ylabel('日均行为量')
ax.set_xticks(range(0, 24, 2)); ax.legend(); ax.grid(linestyle='--', alpha=0.3)
ax = axes[1]
ax.bar(dow['星期'], dow['行为量'], color=[BLUE]*5+[GREEN]*2)
ax.set_title('一周各天行为总量'); ax.set_ylabel('行为量'); ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30)
save('08_工作日周末差异.png')

# ============================================================
# 模块五：类目 → 图9、图10、图11
# ============================================================
cat = df.pivot_table(index='item_category', columns='behavior', values='user_id',
                     aggfunc='size', fill_value=0)
users = df.groupby('item_category')['user_id'].nunique().rename('触达用户')
for c in ('点击', '收藏', '加购', '支付'):
    if c not in cat: cat[c] = 0
cat = cat.join(users).fillna(0)
cat['加购率%'] = (cat['加购']/cat['点击']*100).round(2)
cat['点击支付率%'] = (cat['支付']/cat['点击']*100).round(2)
cat['加购支付比%'] = (cat['支付']/cat['加购'].replace(0, np.nan)*100).round(1)
cat = cat.sort_values('支付', ascending=False)
cat_top = cat.head(15)
sorted_buy = cat['支付'].sort_values(ascending=False)
pareto = sorted_buy.cumsum()/sorted_buy.sum()*100
n80 = int((pareto <= 80).sum()+1)
thr = cat['加购'].quantile(0.8)
med = cat['加购支付比%'].median()
problem = cat[(cat['加购'] >= thr) & (cat['加购支付比%'] < med)].sort_values('加购', ascending=False).head(15)

# 图9：TOP15类目（横条）
fig, ax = plt.subplots(figsize=(8, 5.2))
d = cat_top.sort_values('支付')
ax.barh(d.index.astype(str), d['支付'], color=BLUE)
for i, (v, r) in enumerate(zip(d['支付'], d['点击支付率%'])):
    ax.text(v+cat_top['支付'].max()*0.01, i, f'{v:,.0f}（{r}%）', va='center', fontsize=8.5)
ax.set_title('支付量 TOP15 类目'); ax.set_xlabel('支付次数'); ax.set_ylabel('类目ID')
ax.set_xlim(0, cat_top['支付'].max()*1.25); ax.grid(axis='x', linestyle='--', alpha=0.3)
save('09_类目TOP15.png')

# 图10：帕累托
fig, ax = plt.subplots(figsize=(8, 4.3))
x = range(1, len(pareto)+1)
ax.plot(list(x), pareto.values, color=RED, marker='o', ms=2.5)
ax.axhline(80, color=GRAY, linestyle='--', lw=1)
ax.axvline(n80, color=ORANGE, linestyle='--', lw=1)
ax.annotate(f'前{n80}个（{n80/len(pareto):.1%}）贡献约80%', (n80, 80),
            textcoords='offset points', xytext=(10, -18), fontsize=9)
ax.set_title('类目支付量帕累托（二八）'); ax.set_xlabel('类目（降序）'); ax.set_ylabel('累计占比%')
ax.set_ylim(0, 105); ax.grid(linestyle='--', alpha=0.3)
save('10_类目帕累托.png')

# 图11：问题类目（分组柱）
fig, ax = plt.subplots(figsize=(8, 4.6))
d = problem.sort_values('加购支付比%')
x = range(len(d))
ax.bar(x, d['加购'], color=ORANGE, alpha=0.8, label='加购')
ax.bar(x, d['支付'], color=RED, label='支付')
ax.set_xticks(list(x)); ax.set_xticklabels(d.index.astype(str), rotation=45, fontsize=8)
ax.set_title('高加购低转化问题类目'); ax.set_ylabel('次数'); ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)
save('11_加购不买问题类目.png')

print('✅ 11张图全部生成，在 charts 文件夹里：')
for f in sorted(os.listdir(OUT)):
    print('  ', f)

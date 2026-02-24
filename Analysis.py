import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──────────────────────────────────────────────
df = pd.read_csv('sleep_hr_cleaned_v2.csv')
df.columns = ['date', 'sleep_hr', 'resting_hr']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['month'] = df['date'].dt.to_period('M')
df['sleep_7d'] = df['sleep_hr'].rolling(7, center=True).mean()
df['hr_7d'] = df['resting_hr'].rolling(7, center=True).mean()

# ── 图1：时间序列总览 ──────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('睡眠 & 静息心率 — 数据总览', fontsize=16, fontweight='bold', y=0.98)
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 睡眠时长时间序列
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(df['date'], df['sleep_hr'], color='steelblue', alpha=0.4, width=0.8, label='每日睡眠')
ax1.plot(df['date'], df['sleep_7d'], color='navy', linewidth=1.8, label='7日滑动均值')
ax1.axhline(df['sleep_hr'].mean(), color='red', linestyle='--', linewidth=1.2, label=f"均值 {df['sleep_hr'].mean():.2f}h")
ax1.axhline(7, color='green', linestyle=':', linewidth=1.2, label='推荐 7h')
ax1.set_ylabel('睡眠时长 (小时)')
ax1.set_title('每日睡眠时长')
ax1.legend(fontsize=8, loc='upper right')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax1.set_xlim(df['date'].min(), df['date'].max())

# 静息心率时间序列
ax2 = fig.add_subplot(gs[1, :])
ax2.bar(df['date'], df['resting_hr'], color='tomato', alpha=0.4, width=0.8, label='每日心率')
ax2.plot(df['date'], df['hr_7d'], color='darkred', linewidth=1.8, label='7日滑动均值')
ax2.axhline(df['resting_hr'].mean(), color='orange', linestyle='--', linewidth=1.2,
            label=f"均值 {df['resting_hr'].mean():.1f} bpm")
ax2.set_ylabel('静息心率 (bpm)')
ax2.set_title('每日静息心率')
ax2.legend(fontsize=8, loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_xlim(df['date'].min(), df['date'].max())

# 分布直方图 — 睡眠
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(df['sleep_hr'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax3.axvline(df['sleep_hr'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"均值 {df['sleep_hr'].mean():.2f}h")
ax3.axvline(df['sleep_hr'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"中位数 {df['sleep_hr'].median():.2f}h")
ax3.set_xlabel('睡眠时长 (小时)')
ax3.set_ylabel('频次')
ax3.set_title('睡眠时长分布')
ax3.legend(fontsize=8)

# 分布直方图 — 心率
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(df['resting_hr'], bins=30, color='tomato', edgecolor='white', alpha=0.8)
ax4.axvline(df['resting_hr'].mean(), color='darkred', linestyle='--', linewidth=1.5, label=f"均值 {df['resting_hr'].mean():.1f} bpm")
ax4.axvline(df['resting_hr'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"中位数 {df['resting_hr'].median():.1f} bpm")
ax4.set_xlabel('静息心率 (bpm)')
ax4.set_ylabel('频次')
ax4.set_title('静息心率分布')
ax4.legend(fontsize=8)

plt.savefig('01_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("图1 已保存: 01_overview.png")

# ── 图2：相关性 & 月度趋势 ───────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('睡眠 vs 静息心率 — 关联分析', fontsize=14, fontweight='bold')

# 散点图 + 回归线
ax = axes[0]
corr = df['sleep_hr'].corr(df['resting_hr'])
ax.scatter(df['sleep_hr'], df['resting_hr'], alpha=0.35, s=15, color='steelblue')
m, b = np.polyfit(df['sleep_hr'], df['resting_hr'], 1)
x_fit = np.linspace(df['sleep_hr'].min(), df['sleep_hr'].max(), 100)
ax.plot(x_fit, m * x_fit + b, color='red', linewidth=2, label=f'线性拟合\nr = {corr:.3f}')
ax.set_xlabel('睡眠时长 (小时)')
ax.set_ylabel('静息心率 (bpm)')
ax.set_title(f'散点图（相关系数 r = {corr:.3f}）')
ax.legend()

# 月度均值柱状图（双轴）
ax2 = axes[1]
monthly = df.groupby('month').agg(sleep_mean=('sleep_hr', 'mean'), hr_mean=('resting_hr', 'mean')).reset_index()
monthly['month_str'] = monthly['month'].astype(str)

x = np.arange(len(monthly))
width = 0.4
ax2b = ax2.twinx()

bars1 = ax2.bar(x - width/2, monthly['sleep_mean'], width, color='steelblue', alpha=0.7, label='月均睡眠 (小时)')
bars2 = ax2b.bar(x + width/2, monthly['hr_mean'], width, color='tomato', alpha=0.7, label='月均心率 (bpm)')

ax2.set_xticks(x)
ax2.set_xticklabels(monthly['month_str'], rotation=45, ha='right', fontsize=7)
ax2.set_ylabel('睡眠时长 (小时)', color='steelblue')
ax2b.set_ylabel('静息心率 (bpm)', color='tomato')
ax2.set_title('各月均值')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('02_correlation_monthly.png', dpi=150, bbox_inches='tight')
plt.show()
print("图2 已保存: 02_correlation_monthly.png")

# ── 图3：星期分析 & 异常值 ────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('睡眠 & 心率 — 星期规律 & 异常值', fontsize=14, fontweight='bold')

df['weekday'] = df['date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekly = df.groupby('weekday')[['sleep_hr', 'resting_hr']].mean().reindex(weekday_order)

ax = axes[0]
x = np.arange(7)
width = 0.35
ax_r = ax.twinx()
ax.bar(x - width/2, weekly['sleep_hr'], width, color='steelblue', alpha=0.75, label='睡眠 (h)')
ax_r.bar(x + width/2, weekly['resting_hr'], width, color='tomato', alpha=0.75, label='心率 (bpm)')
ax.set_xticks(x)
ax.set_xticklabels(weekday_labels)
ax.set_ylabel('睡眠时长 (小时)', color='steelblue')
ax_r.set_ylabel('静息心率 (bpm)', color='tomato')
ax.set_title('各星期均值')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# 箱线图
ax2 = axes[1]
sleep_data = [df[df['weekday'] == d]['sleep_hr'].values for d in weekday_order]
bp = ax2.boxplot(sleep_data, tick_labels=weekday_labels, patch_artist=True,
                 boxprops=dict(facecolor='steelblue', alpha=0.6),
                 medianprops=dict(color='red', linewidth=2))
ax2.axhline(7, color='green', linestyle=':', linewidth=1.2, label='推荐 7h')
ax2.set_ylabel('睡眠时长 (小时)')
ax2.set_title('各星期睡眠时长箱线图')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('03_weekday_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("图3 已保存: 03_weekday_boxplot.png")

# ── 统计摘要 ──────────────────────────────────────────────
print("\n" + "="*50)
print("数据统计摘要")
print("="*50)
print(f"记录区间: {df['date'].min().date()} ~ {df['date'].max().date()}  ({len(df)} 天)")
print(f"\n睡眠时长:")
print(f"  均值: {df['sleep_hr'].mean():.2f} h  |  中位数: {df['sleep_hr'].median():.2f} h")
print(f"  最小: {df['sleep_hr'].min():.2f} h  |  最大: {df['sleep_hr'].max():.2f} h")
print(f"  ≥7h 达标率: {(df['sleep_hr'] >= 7).mean()*100:.1f}%")
print(f"\n静息心率:")
print(f"  均值: {df['resting_hr'].mean():.1f} bpm  |  中位数: {df['resting_hr'].median():.1f} bpm")
print(f"  最小: {df['resting_hr'].min():.0f} bpm  |  最大: {df['resting_hr'].max():.0f} bpm")
print(f"\n相关系数 (睡眠 vs 心率): {corr:.3f}")

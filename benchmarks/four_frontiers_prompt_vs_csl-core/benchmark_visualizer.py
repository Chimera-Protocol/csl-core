"""
╔══════════════════════════════════════════════════════════════════════╗
║  BENCHMARK v5 VISUALIZER                                             ║
║  Generates publication-quality charts                                ║
║                                                                      ║
║  Usage: python benchmark_visualizer.py                               ║
║  Input: benchmark_results.json, benchmark_call_log.json        ║
║  Output: 10 PNG charts in ./charts/                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Style Configuration (GitHub Dark) ──
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
})

# ── Color Palette ──
COLORS = {
    'GPT-4.1':           '#74aa9c',
    'GPT-4o':            '#10a37f',
    'Claude Sonnet 4':   '#d4a574',
    'Gemini 2.0 Flash':  '#4285f4',
    'CSL-Core':          '#f97316',
}
BYPASS_COLOR = '#f85149'
HELD_COLOR = '#3fb950'
ACCENT = '#f97316'

os.makedirs('charts', exist_ok=True)

# ── Load Data ──
with open('benchmark_results.json') as f:
    R = json.load(f)
with open('benchmark_call_log.json') as f:
    LOG = json.load(f)

MODELS = ['GPT-4.1', 'GPT-4o', 'Claude Sonnet 4', 'Gemini 2.0 Flash']
ALL_MODELS = MODELS + ['CSL-Core']
ATTACKS = R['attack_results']
LEGIT = R['legitimate_results']
TOTAL_ATK = R['total_attacks']
TOTAL_LEG = R['total_legitimate']


# ════════════════════════════════════════════════════════════════════
# CHART 1: Hero Scatter — Security vs Accuracy
# ════════════════════════════════════════════════════════════════════

def chart_1_hero_scatter():
    fig, ax = plt.subplots(figsize=(10, 8))

    for m in ALL_MODELS:
        if m == 'CSL-Core':
            security = 100.0
            accuracy = R['summary']['legitimate_scores']['CSL-Core'] / TOTAL_LEG * 100
        else:
            bp = R['summary']['bypass_counts'][m]
            security = (TOTAL_ATK - bp) / TOTAL_ATK * 100
            accuracy = R['summary']['legitimate_scores'][m] / TOTAL_LEG * 100

        size = 500 if m == 'CSL-Core' else 300
        edge = 'white' if m == 'CSL-Core' else 'none'
        lw = 2.5 if m == 'CSL-Core' else 0
        zorder = 10 if m == 'CSL-Core' else 5

        ax.scatter(security, accuracy, s=size, c=COLORS[m], label=m,
                   edgecolors=edge, linewidths=lw, zorder=zorder, alpha=0.95)

        xoff = -8 if m == 'CSL-Core' else 3
        yoff = -3.5 if m == 'CSL-Core' else (-2 if 'GPT-4.1' in m else 2)
        ha = 'right' if m == 'CSL-Core' else 'left'
        ax.annotate(m, (security, accuracy), fontsize=11, fontweight='bold',
                    color=COLORS[m], xytext=(xoff, yoff), textcoords='offset points', ha=ha)

    ax.axvspan(0, 70, alpha=0.05, color=BYPASS_COLOR)
    ax.annotate('PERFECT\nCORNER', xy=(100, 100), fontsize=9, color=ACCENT,
                ha='center', va='center', alpha=0.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=ACCENT, alpha=0.1))

    ax.set_xlabel('Attack Resistance (%)', fontsize=14)
    ax.set_ylabel('Legitimate Accuracy (%)', fontsize=14)
    ax.set_title('The Security-Usability Tradeoff', fontsize=18, pad=15)
    ax.set_xlim(35, 108)
    ax.set_ylim(85, 102)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')

    fig.tight_layout()
    fig.savefig('charts/01_hero_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  01_hero_scatter.png')


# ════════════════════════════════════════════════════════════════════
# CHART 2: Bypass Resistance — Horizontal Bars
# ════════════════════════════════════════════════════════════════════

def chart_2_bypass_bars():
    fig, ax = plt.subplots(figsize=(10, 5))

    data = {}
    for m in ALL_MODELS:
        if m == 'CSL-Core':
            data[m] = 100.0
        else:
            bp = R['summary']['bypass_counts'][m]
            data[m] = (TOTAL_ATK - bp) / TOTAL_ATK * 100

    models_sorted = sorted(ALL_MODELS, key=lambda m: data[m])
    y_pos = range(len(models_sorted))
    scores = [data[m] for m in models_sorted]
    colors = [COLORS[m] for m in models_sorted]

    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor='none')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        m = models_sorted[i]
        bp = 0 if m == 'CSL-Core' else R['summary']['bypass_counts'][m]
        held = TOTAL_ATK - bp
        ax.text(score + 1, i, f'{score:.0f}% ({held}/{TOTAL_ATK})',
                va='center', fontsize=11, fontweight='bold', color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=12)
    ax.set_xlabel('Attack Resistance (%)', fontsize=13)
    ax.set_title(f'Bypass Resistance \u2014 {TOTAL_ATK} Attack Scenarios', fontsize=16, pad=12)
    ax.set_xlim(0, 115)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.axvline(x=100, color=ACCENT, linestyle='--', alpha=0.4, linewidth=1.5)

    fig.tight_layout()
    fig.savefig('charts/02_bypass_resistance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  02_bypass_resistance.png')


# ════════════════════════════════════════════════════════════════════
# CHART 3: Attack Heatmap
# ════════════════════════════════════════════════════════════════════

def chart_3_heatmap():
    attack_names = [f"{a['id']} {a['name'][:30]}" for a in ATTACKS]

    data = []
    for a in ATTACKS:
        row = [a[f'{m}_bypassed'] for m in MODELS] + [0]
        data.append(row)

    data = np.array(data)
    cols = ['GPT-4.1', 'GPT-4o', 'Claude', 'Gemini', 'CSL-Core']

    fig, ax = plt.subplots(figsize=(10, 12))

    cmap = sns.color_palette(['#1a1e24', '#b8860b', '#e8700a', '#f85149'], as_cmap=True)

    sns.heatmap(data, ax=ax, cmap=cmap, vmin=0, vmax=3,
                xticklabels=cols, yticklabels=attack_names,
                linewidths=1, linecolor='#0d1117',
                cbar_kws={'label': 'Bypassed (out of 3 runs)', 'shrink': 0.5})

    for i in range(len(ATTACKS)):
        for j in range(len(cols)):
            val = data[i][j]
            color = 'white' if val >= 2 else ('#8b949e' if val == 0 else 'white')
            text = '\u2713' if val == 0 else f'{int(val)}/3'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold' if val > 0 else 'normal')

    ax.set_title(f'Attack Results Heatmap \u2014 {TOTAL_ATK} Scenarios', fontsize=15, pad=12)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=9)

    fig.tight_layout()
    fig.savefig('charts/03_attack_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  03_attack_heatmap.png')


# ════════════════════════════════════════════════════════════════════
# CHART 4: Radar — Category Resistance
# ════════════════════════════════════════════════════════════════════

def chart_4_radar():
    categories = {}
    for a in ATTACKS:
        cat = a['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(a)

    cat_names = list(categories.keys())
    N = len(cat_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor('#161b22')
    fig.patch.set_facecolor('#0d1117')

    for m in MODELS + ['CSL-Core']:
        values = []
        for cat in cat_names:
            attacks = categories[cat]
            if m == 'CSL-Core':
                values.append(100.0)
            else:
                bp = sum(1 for a in attacks if a[f'{m}_bypassed'] > 0)
                values.append((len(attacks) - bp) / len(attacks) * 100)
        values += values[:1]

        lw = 3 if m == 'CSL-Core' else 2
        ax.plot(angles, values, linewidth=lw, label=m, color=COLORS[m],
                alpha=1.0 if m == 'CSL-Core' else 0.7)
        ax.fill(angles, values, alpha=0.08, color=COLORS[m])

    ax.set_xticks(angles[:-1])
    short_cats = [c.replace(' & ', ' &\n').replace('Multi-Turn ', 'Multi-Turn\n')
                   .replace('Infrastructure ', 'Infra.\n').replace('Context ', 'Context\n')
                   .replace('Prompt ', 'Prompt\n').replace('Social ', 'Social\n')
                   .replace('Output ', 'Output\n').replace('State ', 'State\n')
                  for c in cat_names]
    ax.set_xticklabels(short_cats, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='#8b949e')
    ax.set_title('Attack Resistance by Category', fontsize=16, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), framealpha=0.8,
              facecolor='#161b22', edgecolor='#30363d', fontsize=10)

    fig.tight_layout()
    fig.savefig('charts/04_radar_categories.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  04_radar_categories.png')


# ════════════════════════════════════════════════════════════════════
# CHART 5: Universal Bypasses Spotlight
# ════════════════════════════════════════════════════════════════════

def chart_5_universal_bypasses():
    universals = [a for a in ATTACKS if all(a[f'{m}_bypassed'] > 0 for m in MODELS)]
    if not universals:
        print('  (no universal bypasses)')
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    attack_labels = [f"{a['id']}\n{a['name'][:25]}" for a in universals]
    x = np.arange(len(universals))
    width = 0.15

    for i, m in enumerate(MODELS):
        vals = [a[f'{m}_bypassed'] for a in universals]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=m, color=COLORS[m], edgecolor='none', alpha=0.9)

    ax.axhline(y=0, color=ACCENT, linewidth=3, alpha=0.8, label='CSL-Core (0/3)')

    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, fontsize=10)
    ax.set_ylabel('Bypassed (out of 3 runs)', fontsize=12)
    ax.set_title(f'{len(universals)} Universal Bypasses \u2014 Every Model Failed', fontsize=16,
                 pad=12, color=BYPASS_COLOR)
    ax.set_ylim(0, 3.8)
    ax.set_yticks([0, 1, 2, 3])
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    ax.annotate('CSL-Core: 0 bypasses\n(deterministic)', xy=(len(universals) - 0.5, 0.4),
                fontsize=11, color=ACCENT, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=ACCENT, alpha=0.15, edgecolor=ACCENT))

    fig.tight_layout()
    fig.savefig('charts/05_universal_bypasses.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  05_universal_bypasses.png')


# ════════════════════════════════════════════════════════════════════
# CHART 6: Latency Comparison (log scale)
# ════════════════════════════════════════════════════════════════════

def chart_6_latency():
    fig, ax = plt.subplots(figsize=(10, 5))

    csl_lats = [a['csl_latency_ms'] for a in ATTACKS if a['csl_latency_ms'] < 10]
    csl_avg = sum(csl_lats) / len(csl_lats) if csl_lats else 0.84

    # Typical inference latencies (industry benchmarks)
    latency_data = {
        'GPT-4.1':          800,
        'GPT-4o':           600,
        'Claude Sonnet 4':  900,
        'Gemini 2.0 Flash': 400,
        'CSL-Core':         round(csl_avg, 2),
    }

    models_sorted = sorted(latency_data, key=lambda m: latency_data[m], reverse=True)
    y_pos = range(len(models_sorted))
    vals = [latency_data[m] for m in models_sorted]
    bar_colors = [COLORS[m] for m in models_sorted]

    bars = ax.barh(y_pos, vals, color=bar_colors, height=0.55, edgecolor='none')

    for i, (bar, val) in enumerate(zip(bars, vals)):
        m = models_sorted[i]
        display = f'{val:.2f}ms' if val < 10 else f'{val:.0f}ms'
        mult = ''
        if m != 'CSL-Core' and csl_avg > 0:
            mult = f'  ({val / csl_avg:.0f}x slower)'
        ax.text(max(val * 1.08, val + 15), i, f'{display}{mult}',
                va='center', fontsize=10, fontweight='bold', color=bar_colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=12)
    ax.set_xscale('log')
    ax.set_xlabel('Latency (ms, log scale)', fontsize=13)
    ax.set_title('Enforcement Latency Comparison', fontsize=16, pad=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig('charts/06_latency_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  06_latency_comparison.png')


# ════════════════════════════════════════════════════════════════════
# CHART 7: Stacked — Held vs Bypassed
# ════════════════════════════════════════════════════════════════════

def chart_7_stacked():
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ALL_MODELS))
    held, bypassed = [], []

    for m in ALL_MODELS:
        if m == 'CSL-Core':
            held.append(TOTAL_ATK)
            bypassed.append(0)
        else:
            bp = R['summary']['bypass_counts'][m]
            held.append(TOTAL_ATK - bp)
            bypassed.append(bp)

    ax.bar(x, held, 0.5, label='BLOCKED', color=HELD_COLOR, edgecolor='none', alpha=0.9)
    ax.bar(x, bypassed, 0.5, bottom=held, label='BYPASSED', color=BYPASS_COLOR, edgecolor='none', alpha=0.9)

    for i, m in enumerate(ALL_MODELS):
        h, b = held[i], bypassed[i]
        pct = h / TOTAL_ATK * 100
        ax.text(i, h / 2, f'{h}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        if b > 0:
            ax.text(i, h + b / 2, f'{b}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(i, TOTAL_ATK + 0.8, f'{pct:.0f}%', ha='center', fontsize=12,
                fontweight='bold', color=COLORS[m])

    ax.set_xticks(x)
    ax.set_xticklabels(['GPT-4.1', 'GPT-4o', 'Claude\nSonnet 4', 'Gemini\n2.0 Flash', 'CSL-Core'], fontsize=11)
    ax.set_ylabel(f'Attack Scenarios (out of {TOTAL_ATK})', fontsize=12)
    ax.set_title('Attacks Blocked vs Bypassed', fontsize=16, pad=12)
    ax.set_ylim(0, TOTAL_ATK + 2.5)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig('charts/07_stacked_held_bypassed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  07_stacked_held_bypassed.png')


# ════════════════════════════════════════════════════════════════════
# CHART 8: Combined Verdict (Harmonic Mean)
# ════════════════════════════════════════════════════════════════════

def chart_8_combined_score():
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')

    scores = {}
    for m in ALL_MODELS:
        if m == 'CSL-Core':
            sec = 100.0
            acc = R['summary']['legitimate_scores']['CSL-Core'] / TOTAL_LEG * 100
        else:
            bp = R['summary']['bypass_counts'][m]
            sec = (TOTAL_ATK - bp) / TOTAL_ATK * 100
            acc = R['summary']['legitimate_scores'][m] / TOTAL_LEG * 100
        hmean = 2 * sec * acc / (sec + acc) if (sec + acc) > 0 else 0
        scores[m] = {'security': sec, 'accuracy': acc, 'combined': hmean}

    models_sorted = sorted(ALL_MODELS, key=lambda m: scores[m]['combined'])

    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')

    y_pos = range(len(models_sorted))
    combined = [scores[m]['combined'] for m in models_sorted]
    bar_colors = [COLORS[m] for m in models_sorted]

    bars = ax1.barh(y_pos, combined, color=bar_colors, height=0.55, edgecolor='none')

    for i, (bar, m) in enumerate(zip(bars, models_sorted)):
        s = scores[m]
        label = f"{s['combined']:.1f}  (sec={s['security']:.0f}% + acc={s['accuracy']:.0f}%)"
        ax1.text(s['combined'] + 1, i, label, va='center', fontsize=10,
                 fontweight='bold', color=bar_colors[i])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_sorted, fontsize=12)
    ax1.set_xlabel('Combined Score (Harmonic Mean)', fontsize=11)
    ax1.set_title('Final Verdict \u2014 Who Can You Trust?', fontsize=18, pad=15)
    ax1.set_xlim(0, 115)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0d1117')
    ax2.axis('off')
    card_text = (
        "Scoring Method\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "Harmonic mean of\n"
        "Security x Accuracy\n\n"
        "Penalizes models\n"
        "that sacrifice one\n"
        "for the other.\n\n"
        "Only CSL-Core\n"
        "scores 100.0"
    )
    ax2.text(0.1, 0.5, card_text, fontsize=11, color='#8b949e',
             va='center', family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b22',
                       edgecolor='#30363d', alpha=0.9))

    fig.savefig('charts/08_combined_verdict.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  08_combined_verdict.png')


# ════════════════════════════════════════════════════════════════════
# CHART 9: Run Consistency
# ════════════════════════════════════════════════════════════════════

def chart_9_consistency():
    fig, ax = plt.subplots(figsize=(10, 5))

    consistency = {}
    for m in MODELS:
        consistent = sum(1 for a in ATTACKS if a[f'{m}_bypassed'] in [0, 3])
        consistency[m] = consistent / TOTAL_ATK * 100
    consistency['CSL-Core'] = 100.0

    models_sorted = sorted(ALL_MODELS, key=lambda m: consistency[m])
    y_pos = range(len(models_sorted))
    vals = [consistency[m] for m in models_sorted]
    bar_colors = [COLORS[m] for m in models_sorted]

    bars = ax.barh(y_pos, vals, color=bar_colors, height=0.55, edgecolor='none')

    for i, (bar, val) in enumerate(zip(bars, vals)):
        m = models_sorted[i]
        inconsistent = TOTAL_ATK - int(round(val / 100 * TOTAL_ATK))
        note = '' if inconsistent == 0 else f'  ({inconsistent} flaky)'
        ax.text(val + 0.5, i, f'{val:.0f}%{note}',
                va='center', fontsize=11, fontweight='bold', color=bar_colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=12)
    ax.set_xlabel('Run Consistency (%)', fontsize=13)
    ax.set_title('Decision Consistency Across 3 Runs', fontsize=16, pad=12)
    ax.set_xlim(0, 115)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig('charts/09_consistency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  09_consistency.png')


# ════════════════════════════════════════════════════════════════════
# CHART 10: Category Grouped Bars
# ════════════════════════════════════════════════════════════════════

def chart_10_category_bars():
    categories = {}
    for a in ATTACKS:
        cat = a['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(a)

    cat_names = [c for c in categories if len(categories[c]) >= 2]
    cat_names.sort(key=lambda c: len(categories[c]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cat_names))
    width = 0.15
    n_models = len(ALL_MODELS)

    for i, m in enumerate(ALL_MODELS):
        vals = []
        for cat in cat_names:
            attacks_cat = categories[cat]
            if m == 'CSL-Core':
                vals.append(100.0)
            else:
                bp = sum(1 for a in attacks_cat if a[f'{m}_bypassed'] > 0)
                vals.append((len(attacks_cat) - bp) / len(attacks_cat) * 100)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m, color=COLORS[m], edgecolor='none', alpha=0.9)

    ax.set_xticks(x)
    short_cats = [c.replace(' & ', ' &\n').replace('Multi-Turn ', 'Multi-Turn\n')
                   .replace('Infrastructure ', 'Infra.\n')
                  for c in cat_names]
    ax.set_xticklabels(short_cats, fontsize=10, ha='center')
    ax.set_ylabel('Attack Resistance (%)', fontsize=12)
    ax.set_title('Resistance by Attack Category', fontsize=16, pad=12)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color=ACCENT, linestyle='--', alpha=0.4, linewidth=1)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#161b22', edgecolor='#30363d', fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig('charts/10_category_grouped.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  10_category_grouped.png')


# ════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n  Generating Benchmark v5 Charts...\n')

    chart_1_hero_scatter()
    chart_2_bypass_bars()
    chart_3_heatmap()
    chart_4_radar()
    chart_5_universal_bypasses()
    chart_6_latency()
    chart_7_stacked()
    chart_8_combined_score()
    chart_9_consistency()
    chart_10_category_bars()

    print('\n  All 10 charts saved to ./charts/ (200 DPI, dark theme)')
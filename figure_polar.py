import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# --- 全局字体设置 ---
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'

# --- 数据准备 ---
categories = ['Overall', 'Known Scenes', 'Unknown Scenes']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

orfd_data = {
    'Off-Net': [0.835, 0.922, 0.684],
    'M2F2-Net': [0.832, 0.913, 0.690],
    'OT-Drive (Ours)': [0.946, 0.951, 0.932]
}

orad_data = {
    'Off-Net': [0.879, 0.886, 0.833],
    'M2F2-Net': [0.891, 0.898, 0.849],
    'OT-Drive (Ours)': [0.901, 0.907, 0.861]
}

def close_loop(data):
    return data + data[:1]

# --- 绘图函数 ---
def create_custom_radar(data_dict, title, ylim, filename):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # 设置轴标签
    plt.xticks(angles[:-1], categories)
    
    # --- 重点突出 Unknown Scenes ---
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
        if "Unknown" in label.get_text():
            label.set_color('#D62728') # 使用鲜艳的红色
            label.set_fontsize(20)     # 字号加大
        else:
            label.set_color('black')
            label.set_fontsize(16)     # 普通字号
            
    # 设置 Y 轴
    ax.set_rlabel_position(0)
    plt.yticks(np.linspace(ylim[0], ylim[1], 5), color="black", size=12, fontweight='bold')
    plt.ylim(ylim[0], ylim[1])
    
    # 样式设置
    styles = {
        'Off-Net': {'color': 'grey', 'linestyle': '--', 'marker': 'o', 'linewidth': 2, 'markersize': 8},
        'M2F2-Net': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's', 'linewidth': 2, 'markersize': 8},
        'OT-Drive (Ours)': {'color': '#d62728', 'linestyle': '-', 'linewidth': 5, 'marker': '*', 'markersize': 14} # 线条加粗
    }
    
    # 绘制数据
    for model_name, values in data_dict.items():
        values_closed = close_loop(values)
        ax.plot(angles, values_closed, label=model_name, **styles[model_name])
        if model_name == 'TDT (Ours)':
            ax.fill(angles, values_closed, color='#d62728', alpha=0.15)
            
    # 标题设置
    ax.set_title(title, size=24, color='black', y=1.1, fontweight='bold')
    
    # 网格线优化
    ax.grid(True, linestyle=':', alpha=0.8, color='gray', linewidth=1)
    
    # 图例设置
    legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_fontsize(16)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# --- 生成图片 ---
create_custom_radar(orfd_data, "ORFD Dataset (Few-shot)\nmIoU Comparison", [0.6, 1.0], 'orfd_radar_custom.svg')
create_custom_radar(orad_data, "ORAD-3D Dataset (Large-scale)\nmIoU Comparison", [0.8, 0.92], 'orad_radar_custom.svg')
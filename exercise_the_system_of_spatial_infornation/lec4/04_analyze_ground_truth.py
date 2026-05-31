#!/usr/bin/env python
# coding: utf-8

"""
Transportation Mode Detection - Ground Truth Data Analysis

3クラス（walk, train, vehicle）のそれぞれについて、
4つの特徴量（distance, speed, accel, angular_velocity）の
統計量（平均・標準偏差・最小・最大）を算出し、
バイオリンプロットと箱ひげ図の2種類のグラフで可視化します。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    input_file = './traj_010_labeled_with_features.csv'

    if not os.path.exists(input_file):
        print(f"Error: Dataset '{input_file}' not found. Please run 01_feature_extraction.py first.")
        return

    print(f"Loading feature dataset from {input_file}...")
    df_raw = pd.read_csv(input_file, index_col=0)

    # --- トリップ単位で特徴量を平均集計 ---
    print("Aggregating features by trip (trans_trip)...")
    df_trip = df_raw.groupby('trans_trip').agg({
        'distance':         'mean',
        'speed':            'mean',
        'accel':            'mean',
        'angle':            'mean',
        'angular_velocity': 'mean',
        'trans_mode':       'first',
    })
    df_trip.dropna(inplace=True)

    # --- 3クラスへのラベルマッピング ---
    trans_mode_map = {
        'bus':    'vehicle',
        'car':    'vehicle',
        'taxi':   'vehicle',
        'subway': 'train',
        'walk':   'walk',
        'train':  'train',
    }
    df_trip['trans_mode_mapped'] = df_trip['trans_mode'].map(trans_mode_map)

    # --- 統計量の表示・CSV保存 ---
    features = ['distance', 'speed', 'accel', 'angular_velocity']
    print("\n" + "="*80)
    print("【統計量】3クラス別 特徴量サマリ（mean / std / min / max）")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    stats = df_trip.groupby('trans_mode_mapped')[features].agg(['mean', 'std', 'min', 'max'])
    print(stats)
    stats.to_csv('./mode_stats_3class_summary.csv')
    print("\n-> 統計表をCSVに保存しました: ./mode_stats_3class_summary.csv")

    # =====================================================================
    # 共通設定
    # =====================================================================
    palette  = {'walk': '#00B945', 'train': '#0C5DA5', 'vehicle': '#FF9500'}
    order    = ['walk', 'train', 'vehicle']
    feature_info = [
        ('distance',         'Distance',         'km'),
        ('speed',            'Speed',            'km/h'),
        ('accel',            'Acceleration',     'km/h/s'),
        ('angular_velocity', 'Angular Velocity', 'deg/s'),
    ]

    # =====================================================================
    # 図1: バイオリンプロット（2 x 2）
    # =====================================================================
    print("\nGenerating Figure 1: Violin plots...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Feature Distribution by Transportation Mode (Violin Plot)',
                  fontsize=15, fontweight='bold')

    for ax, (col, title, unit) in zip(axs1.flatten(), feature_info):
        sns.violinplot(
            data=df_trip,
            x='trans_mode_mapped',
            y=col,
            order=order,
            palette=palette,
            hue='trans_mode_mapped',
            hue_order=order,
            inner='box',
            ax=ax,
            legend=False,
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Transportation Mode', fontsize=10)
        ax.set_ylabel(unit, fontsize=10)
        # 外れ値を除いてy軸を調整（1〜99パーセンタイル範囲）
        q_low  = df_trip.groupby('trans_mode_mapped')[col].quantile(0.01).min()
        q_high = df_trip.groupby('trans_mode_mapped')[col].quantile(0.99).max()
        ax.set_ylim(q_low, q_high)

    plt.tight_layout()
    plt.savefig('./plot_all_features_violin.png', dpi=150)
    print("-> バイオリンプロット保存: ./plot_all_features_violin.png")
    plt.show(block=False)
    plt.pause(3.0)
    plt.close()

    # =====================================================================
    # 図2: 箱ひげ図（2 x 2）
    # =====================================================================
    print("\nGenerating Figure 2: Box plots...")
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Feature Distribution by Transportation Mode (Box Plot)',
                  fontsize=15, fontweight='bold')

    for ax, (col, title, unit) in zip(axs2.flatten(), feature_info):
        sns.boxplot(
            data=df_trip,
            x='trans_mode_mapped',
            y=col,
            order=order,
            palette=palette,
            hue='trans_mode_mapped',
            hue_order=order,
            notch=False,
            showcaps=True,
            showmeans=True,
            meanprops={'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 8},
            flierprops={'marker': 'x', 'markersize': 4, 'alpha': 0.4},
            ax=ax,
            legend=False,
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Transportation Mode', fontsize=10)
        ax.set_ylabel(unit, fontsize=10)
        q_low  = df_trip.groupby('trans_mode_mapped')[col].quantile(0.01).min()
        q_high = df_trip.groupby('trans_mode_mapped')[col].quantile(0.99).max()
        ax.set_ylim(q_low, q_high)

    plt.tight_layout()
    plt.savefig('./plot_all_features_boxplot.png', dpi=150)
    print("-> 箱ひげ図保存: ./plot_all_features_boxplot.png")
    plt.show(block=False)
    plt.pause(3.0)
    plt.close()

    print("\nAnalysis completed successfully!")

if __name__ == '__main__':
    main()

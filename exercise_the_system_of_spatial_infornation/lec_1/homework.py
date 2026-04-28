import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np

# 1. データの読み込み
# tokyo23 フォルダ内の各シェープファイルを読み込む
tatemono = gpd.read_file('tokyo23/tatemono.shp')
elevation = gpd.read_file('tokyo23/elevation.shp')
pop = gpd.read_file('tokyo23/pop.shp')

# 2. P1995カラムに基づいて色を割り当てる
# クラス分け（4分割）
# ここではデータの分布に合わせて分類（Quantiles: 等量分割）を行います
import mapclassify
classifier = mapclassify.Quantiles(pop['P1995'], k=4)
pop['class'] = classifier.yb

# 指定された色のリスト（Class 1: Blue, Class 2: Green, Class 3: Yellow, Class 4: Red）
colors = ['blue', 'green', 'yellow', 'red']
cmap = ListedColormap(colors)

# 3. マップの作成（地図の三要素：方位記号、スケールバー、凡例を含む）
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 背景として標高データを描画（薄いグレーなど）
elevation.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5)

# 人口データの描画
pop.plot(column='class', ax=ax, cmap=cmap, legend=False)

# 凡例の作成
# クラスの境界値を取得してラベルを作成
bins = classifier.bins
labels = [
    f'Class 1: <={bins[0]:.0f} (Blue)',
    f'Class 2: {bins[0]:.0f} - {bins[1]:.0f} (Green)',
    f'Class 3: {bins[1]:.0f} - {bins[2]:.0f} (Yellow)',
    f'Class 4: >{bins[2]:.0f} (Red)'
]

# カスタム凡例の追加
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i],
                          markerfacecolor=colors[i], markersize=10) for i in range(4)]
ax.legend(handles=legend_elements, loc='upper right', title='Population (1995)')

# 方位記号（North Arrow）の追加
x, y, arrow_length = 0.05, 0.95, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, xycoords='axes fraction')

# スケールバーの追加
# matplotlib-scalebar ライブラリを使用することを推奨しますが、
# ここでは簡易的に座標系に基づいたスケールを表示するか、
# インストールが必要なライブラリとして提案します。
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    # 投影法が平面直角座標系やUTMなど（メートル単位）であることを前提としています
    # もし緯度経度（度）の場合は、適切な変換が必要です
    scalebar = ScaleBar(1, location='lower right') 
    ax.add_artist(scalebar)
except ImportError:
    print("スケールバーの表示には 'matplotlib-scalebar' ライブラリのインストールが必要です。")

# タイトルの追加
ax.set_title('Tokyo Population Map (1995)', fontsize=15)
ax.set_axis_off() # 座標軸を非表示にする

# 4. マップを画像として保存
plt.savefig('population_map_1995.png', dpi=300, bbox_inches='tight')
plt.show()

print("マップを 'population_map_1995.png' として保存しました。")

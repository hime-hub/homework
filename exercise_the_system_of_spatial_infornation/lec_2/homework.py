import geopandas as gpd
import matplotlib.pyplot as plt
import os

# データの読み込みパス
data_path = 'lecture2_data/tokyo23/'

# 各問題ごとの出力フォルダを作成する関数
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"フォルダを作成しました: {dir_name}")

# データの読み込み
# データの読み込みとCRSの設定 (EPSG:6668: JGD2011)
ku = gpd.read_file(data_path + 'ku.shp').to_crs(epsg=6668)
station = gpd.read_file(data_path + 'station.shp').to_crs(epsg=6668)
dentist = gpd.read_file(data_path + 'dentist.shp').to_crs(epsg=6668)
road = gpd.read_file(data_path + 'road.shp').to_crs(epsg=6668)
highway = gpd.read_file(data_path + 'highway.shp').to_crs(epsg=6668)
supermarket = gpd.read_file(data_path + 'supermarket.shp').to_crs(epsg=6668)
pop = gpd.read_file(data_path + 'pop.shp').to_crs(epsg=6668)
comm = gpd.read_file(data_path + 'H16_commerce.shp').to_crs(epsg=6668)

# ==========================================
# Problem 1
# ==========================================
print("\n--- Problem 1 ---")
out_p1 = 'problem1'
create_dir(out_p1)

# 1. 駅データの250mバッファ作成と保存
# メートル単位の計算のため、一時的に投影座標系(EPSG:6677)に変換してバッファを作成し、6668に戻します
station_buffer250 = station.copy()
station_buffer250['geometry'] = station.to_crs(epsg=6677).buffer(250).to_crs(epsg=6668)
station_buffer250.to_file(os.path.join(out_p1, 'station_buffer250.shp'))
print(f"結果を保存しました: {out_p1}/station_buffer250.shp")

# ==========================================
# Problem 2
# ==========================================
print("\n--- Problem 2 ---")
out_p2 = 'problem2'
create_dir(out_p2)

# 1. 世田谷区のポリゴン抽出と保存
setagaya = ku[ku['NAME'] == '世田谷区']
setagaya.to_file(os.path.join(out_p2, 'setagaya.shp'))
print(f"結果を保存しました: {out_p2}/setagaya.shp")

# 2. 世田谷区内のデータ（歯科、駅、道路）の抽出
setagaya_union = setagaya.unary_union
dentist_setagaya = dentist[dentist.intersects(setagaya_union)]
station_setagaya = station[station.intersects(setagaya_union)]
road_setagaya = road[road.intersects(setagaya_union)]

dentist_setagaya.to_file(os.path.join(out_p2, 'dentist_setagaya.shp'))
station_setagaya.to_file(os.path.join(out_p2, 'station_setagaya.shp'))
road_setagaya.to_file(os.path.join(out_p2, 'road_setagaya.shp'))
print(f"世田谷区内のデータを保存しました: {out_p2}/ (dentist, station, road)")

# 3. 条件を満たす歯科医院の数を探す（世田谷区内に限定）
dentists_near_station = dentist_setagaya[dentist_setagaya.intersects(station_buffer250.unary_union)]
count_near_station = len(dentists_near_station)

road_buffer500 = road.to_crs(epsg=6677).buffer(500).to_crs(epsg=6668).unary_union
dentists_far_road = dentist_setagaya[~dentist_setagaya.intersects(road_buffer500)]
count_far_road = len(dentists_far_road)

# 結果の表示とファイル保存
res_p2 = f"世田谷区内で駅から250m以内の歯科医院数: {count_near_station}\n"
res_p2 += f"世田谷区内で道路から500m以上離れている歯科医院数: {count_far_road}"
print(res_p2)

with open(os.path.join(out_p2, 'results.txt'), 'w', encoding='utf-8') as f:
    f.write(res_p2)

# ==========================================
# Problem 3
# ==========================================
print("\n--- Problem 3 ---")
out_p3 = 'problem3'
create_dir(out_p3)

# 1. 中野区に隣接する区を探して保存
nakano = ku[ku['NAME'] == '中野区'].geometry.iloc[0]
adjacent_nakano = ku[ku.geometry.touches(nakano)]
adjacent_nakano.to_file(os.path.join(out_p3, 'adjacent_nakano.shp'))
print(f"隣接区を保存しました: {out_p3}/adjacent_nakano.shp")

# 2. 高速道路が通過する区を探して保存
highway_wards = ku[ku.intersects(highway.unary_union)]
highway_wards.to_file(os.path.join(out_p3, 'highway_wards.shp'))
print(f"高速道路が通過する区を保存しました: {out_p3}/highway_wards.shp")

# 3. バッファ内のスーパーマーケットの数と割合
supermarkets_in_buffer = supermarket[supermarket.intersects(station_buffer250.unary_union)]
num_in_buffer = len(supermarkets_in_buffer)
total_supermarkets = len(supermarket)
proportion = (num_in_buffer / total_supermarkets) * 100 if total_supermarkets > 0 else 0

# 結果の表示とファイル保存
res_p3 = f"駅バッファ内のスーパーマーケット数: {num_in_buffer} / {total_supermarkets} ({proportion:.2f}%)"
print(res_p3)

with open(os.path.join(out_p3, 'results.txt'), 'w', encoding='utf-8') as f:
    f.write(res_p3)

# ==========================================
# Problem 4
# ==========================================
print("\n--- Problem 4 ---")
out_p4 = 'problem4'
create_dir(out_p4)

# 1. 商業販売額の集計 (SELL_SUM)
comm_with_pop = gpd.sjoin(comm, pop, how='inner', predicate='intersects')
sell_sum_by_area = comm_with_pop.groupby('index_right')['SELL'].sum()

pop['SELL_SUM'] = sell_sum_by_area
pop['SELL_SUM'] = pop['SELL_SUM'].fillna(0)

# 保存
pop.to_file(os.path.join(out_p4, 'pop_with_sell_sum.shp'))
print(f"結果を保存しました: {out_p4}/pop_with_sell_sum.shp")

# 2. 可視化
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
pop.plot(column='SELL_SUM', ax=ax, cmap='YlOrRd', legend=True,
         legend_kwds={'label': "Total Sell Volume (H16 Commerce)"})
ax.set_title('Visualization of SELL_SUM by Area')
plt.savefig(os.path.join(out_p4, 'sell_sum_map.png'), dpi=300, bbox_inches='tight')
print(f"可視化結果を保存しました: {out_p4}/sell_sum_map.png")

print("\n--- 全ての解析が完了し、各フォルダに保存されました ---")

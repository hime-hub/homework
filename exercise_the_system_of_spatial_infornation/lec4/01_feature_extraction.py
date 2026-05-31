#!/usr/bin/env python
# coding: utf-8

"""
Transportation Mode Detection - Feature Extraction

This script extracts basic features of GPS data (distance, speed, acceleration, angle, and angular velocity)
which will be used to estimate transportation modes.
It loads 'traj_010_labeled.csv' and outputs 'traj_010_labeled_with_features.csv'.
"""

import os
from math import radians, cos, sin, asin, sqrt, atan2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_distance(lat2, lon2, lat1, lon1):
    """Function to calculate great-circle distance between two coordinates using the Haversine formula."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371.009  # Earth's radius in kilometers
    return c * r

def calc_accel(speed_1, speed_2, time_diff_sec):
    """Function to calculate acceleration between two points."""
    if pd.isna(speed_1) or pd.isna(speed_2) or pd.isna(time_diff_sec):
        return np.nan
    speed_delta = speed_2 - speed_1
    if (time_diff_sec == 0) or (speed_delta == 0):
        return 0.0
    return speed_delta / time_diff_sec

def calc_angle(lat1, lat2, lon1, lon2):
    """Function to calculate angle differences between two coordinates."""
    if pd.isna(lat1) or pd.isna(lat2) or pd.isna(lon1) or pd.isna(lon2):
        return np.nan
    # Convert degrees to radians
    lat1_rad, lat2_rad, lon1_rad, lon2_rad = map(radians, [lat1, lat2, lon1, lon2])
    x = cos(lat2_rad) * sin(lon2_rad - lon1_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon2_rad - lon1_rad)
    brng = atan2(x, y)
    brng = np.degrees(brng)
    brng = brng + 360 if brng < 0 else brng
    return brng

def calc_angular_velocity(angle1, angle2, time_diff_sec):
    """Function to calculate angular velocity."""
    if pd.isna(angle1) or pd.isna(angle2) or pd.isna(time_diff_sec):
        return np.nan
    bear_delta = angle2 - angle1
    if time_diff_sec == 0:
        return 0.0
    return abs(bear_delta / time_diff_sec)

def main():
    user = '010'
    input_file = f'./traj_{user}_labeled.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Labeled dataset '{input_file}' not found. Run 00_clean_GPS_data.py first.")
        return
        
    print(f"Loading GPS data from {input_file}...")
    traj_df = pd.read_csv(input_file, index_col=0)
    print("Size of observations: {:,}".format(traj_df.shape[0]))
    
    # Exclude rows containing no information about transportation mode
    traj_df = traj_df.dropna(subset=['trans_mode'], axis=0)
    print("Size of observations after dropping rows without transportation mode: {:,}".format(traj_df.shape[0]))
    
    # Convert time values to timestamp and sort
    print("Sorting and calculating time differences...")
    traj_df['timestamp'] = pd.to_datetime(traj_df['time'])
    traj_df.sort_values(['trans_trip', 'timestamp'], inplace=True)
    
    # Calculate time differences per trip
    traj_df['time_delta'] = (traj_df.timestamp - traj_df.groupby(['trans_trip']).timestamp.shift(1))
    traj_df['dt_seconds'] = traj_df['time_delta'].dt.seconds
    
    # Summary info on time differences
    print("\nMax time delta summary by transportation mode:")
    summary_df = (traj_df.groupby(['trans_mode', 'trans_trip'])
                  .time_delta
                  .max()
                  .dt.seconds
                  .groupby(level=0).agg(['mean', 'median', 'max', 'min', 'count']))
    print(summary_df)
    
    # Since airplane and car trips are rare for this user, remove them for further analysis
    print("\nRemoving 'airplane' and 'car' modes...")
    traj_df = traj_df[~traj_df['trans_mode'].isin(['airplane', 'car'])]
    print("Remaining records: {:,}".format(traj_df.shape[0]))
    
    # Calculate distances from previous points
    print("Calculating distances between consecutive points...")
    traj_df[['latitude_prev', 'longitude_prev']] = traj_df.groupby('trans_trip')[['latitude', 'longitude']].shift(1)
    traj_df['distance'] = traj_df.apply(
        lambda x: calc_distance(x["latitude"], x["longitude"], x['latitude_prev'], x['longitude_prev']), axis=1
    )
    
    # Calculate speed (distance in km / hour)
    print("Calculating speed (km/h)...")
    traj_df['speed'] = np.where(
        ((traj_df['distance'].notnull()) & (traj_df['dt_seconds'] != 0)),
        traj_df['distance'] / (traj_df['dt_seconds'] / 3600),
        0.0
    )
    
    # Calculate acceleration (speed difference / time difference)
    print("Calculating acceleration...")
    traj_df['speed_prev'] = traj_df.groupby('trans_trip')['speed'].shift(1)
    traj_df['accel'] = traj_df.apply(
        lambda x: calc_accel(x['speed_prev'], x['speed'], x['dt_seconds']), axis=1
    )
    
    # Calculate angle differences
    print("Calculating angles...")
    traj_df['angle'] = traj_df.apply(
        lambda x: calc_angle(x['latitude'], x['latitude_prev'], x['longitude'], x['longitude_prev']), axis=1
    )
    
    # Calculate angular velocity (angle difference / time difference)
    print("Calculating angular velocity...")
    traj_df['angle_prev'] = traj_df.groupby('trans_trip')['angle'].shift(1)
    traj_df['angular_velocity'] = traj_df.apply(
        lambda x: calc_angular_velocity(x['angle_prev'], x['angle'], x['dt_seconds']), axis=1
    )
    
    # Filter valid rows
    traj_df = traj_df[traj_df['dt_seconds'].notnull()]
    traj_df = traj_df[~(traj_df['dt_seconds'] == 0)]
    
    # Visualizations
    print("\nGenerating and displaying plots...")
    colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
    
    # 1. Relation between angular velocity and speed
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ['^', 'o', 's', 'v', 'd']
    for i, mode in enumerate(traj_df['trans_mode'].unique()):
        chunk = traj_df[traj_df['trans_mode'] == mode]
        trip_df = chunk.groupby('trans_trip')[['speed', 'angular_velocity']].mean()
        ax.scatter(trip_df['speed'], trip_df['angular_velocity'], facecolor='None',
                   marker=markers[i % len(markers)], edgecolor=colors[i % len(colors)], label=mode, linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_xlabel('speed (km/h)')
    ax.set_ylabel('angular velocity (deg/sec)')
    ax.set_title('Relation between Angular Velocity and Speed per Trip')
    plt.tight_layout()
    plt.savefig('plot_speed_vs_angular_velocity.png')
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    # 2. Boxplot of acceleration
    fig, ax = plt.subplots(figsize=(6, 4))
    accel_df = traj_df.groupby(['trans_mode', 'trans_trip'])[['accel']].mean().reset_index()
    sns.boxplot(data=accel_df, x="trans_mode", y="accel",
                notch=True, showcaps=False, showmeans=True,
                meanprops={'markerfacecolor':'none', 'markeredgecolor':'green'},
                flierprops={"marker": "x"},
                boxprops={"facecolor": (.4, .6, .8, .2)},
                medianprops={"color": "coral"},
                ax=ax)
    ax.set_xlabel('transportation mode')
    ax.set_title('Boxplot of Acceleration per Transportation Mode')
    plt.tight_layout()
    plt.savefig('plot_acceleration_boxplot.png')
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    # Save the dataframe with created features
    output_path = f'./traj_{user}_labeled_with_features.csv'
    print(f"Saving dataset with features to: {output_path}")
    traj_df.to_csv(output_path)
    print("Feature extraction completed successfully!")

if __name__ == '__main__':
    main()

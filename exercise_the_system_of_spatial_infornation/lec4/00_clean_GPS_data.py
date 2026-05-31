#!/usr/bin/env python
# coding: utf-8

"""
Transportation Mode Detection - Data Cleaning

This script explores GPS trajectory data to clean and prepare it for transportation mode estimation.
It loads Geolife GPS trajectory data for user '010', maps transportation mode labels from 'labels.txt',
and outputs a cleaned labeled CSV dataset.
"""

import os
import pandas as pd
import numpy as np

def get_trans_trip(record_dt, ref_df):
    """Function to provide transportation mode labels based on the record times."""
    time_fit = (record_dt >= ref_df['Start Time']) & (record_dt <= ref_df['End Time'])
    nmatch = time_fit.sum()
    if nmatch == 0:
        t_idx = None
    else:
        if nmatch > 1:
            print('More than one mode match!')
        t_idx = ref_df.loc[time_fit].iloc[0].name
    return t_idx

def main():
    user = '010'
    # Local path to Geolife Trajectory directory
    path = f"./Geolife Trajectories 1.3/Data/{user}/Trajectory"
    
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist. Please check your working directory.")
        return

    print("Reading GPS trajectory files...")
    files = os.listdir(path)
    column_name = ['latitude', 'longitude', 'height', 'days_total', 'date', 'time']
    
    df_lst = []
    for f in files:
        if f.endswith('plt'):
            fpath = os.path.join(path, f)
            df = pd.read_csv(fpath, skiprows=6, usecols=[0, 1, 3, 4, 5, 6], names=column_name)
            df = df.assign(record_dt=lambda x: pd.to_datetime(x['date'] + ' ' + x['time']), user=user)
            df_lst.append(df)
            
    traj_df = pd.concat(df_lst)
    print(f"Loaded trajectory data. Shape: {traj_df.shape}")
    
    # Import transport mode labels (ground truth)
    labels_file = os.path.join(os.path.dirname(path), "labels.txt")
    print(f"Loading labels from: {labels_file}")
    if not os.path.exists(labels_file):
        print(f"Error: Labels file '{labels_file}' not found.")
        return
        
    trip_trans = pd.read_csv(labels_file, sep="\t")
    trip_trans['Start Time'] = pd.to_datetime(trip_trans['Start Time'])
    trip_trans['End Time'] = pd.to_datetime(trip_trans['End Time'])
    
    print("Mapping transportation modes to trajectories (this may take a minute)...")
    traj_df['trans_trip'] = traj_df['record_dt'].apply(lambda x: get_trans_trip(x, trip_trans))
    
    # Exclude rows not holding any transportation mode information
    has_trip = ~(traj_df.trans_trip.isnull())
    
    # Map transportation mode names
    traj_df['trans_mode'] = np.nan
    traj_df.loc[has_trip, 'trans_mode'] = traj_df.loc[has_trip]['trans_trip'].apply(
        lambda x: trip_trans.loc[x, 'Transportation Mode']
    )
    
    print("N of rows with transportation mode information:\t{:,}".format(
        traj_df[~traj_df.trans_trip.isnull()].shape[0]))
    print("N of rows without transportation mode information:\t{:,}".format(
        traj_df[traj_df.trans_trip.isnull()].shape[0]))
    
    # Save files as CSV in local workspace
    output_path = f'./traj_{user}_labeled.csv'
    print(f"Saving cleaned dataset to: {output_path}")
    traj_df.to_csv(output_path)
    print("Data cleaning completed successfully!")

if __name__ == '__main__':
    main()

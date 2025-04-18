import os

import pandas as pd

base_data_path = "../data/download_folder/NeuroIMAGE"
participants_tsv = os.path.join(base_data_path, "participants.tsv")
free_participants_csv = os.path.join(base_data_path, "free_participants.csv")
free_ad_hd_csv = os.path.join(base_data_path, "free_ad_hd.csv")
free_control_csv = os.path.join(base_data_path, "free_control.csv")

def get_data():
    # Load participants.tsv and filter only participants that agreed to the disclaimer
    df = pd.read_csv(participants_tsv, sep='\t')
    free_df = df[df['disclaimer'] != 'This data requires permission to use.']

    # Save "free participants" data to a CSV file
    if os.path.isfile(free_participants_csv):
        os.remove(free_participants_csv)
    free_df.to_csv(free_participants_csv, index=False)
    print(f"Saved free participants data to {free_participants_csv}")

    # Define ADHD and control groups and create CSV files for each
    adhd_dx_values = ['ADHD-Combined', 'ADHD-Hyperactive/Impulsive']
    df_adhd = free_df[free_df['dx'].isin(adhd_dx_values)]
    df_adhd.to_csv(free_ad_hd_csv, index=False)
    print(f"Saved ADHD participants data to {free_ad_hd_csv}")

    control_dx_values = ['Typically Developing Children']
    df_control = free_df[free_df['dx'].isin(control_dx_values)]
    df_control.to_csv(free_control_csv, index=False)
    print(f"Saved Control participants data to {free_control_csv}")

    return free_control_csv, free_ad_hd_csv

if __name__ == "__main__":
    get_data()

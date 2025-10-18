import pandas as pd

def create_filtered_ship(ship_path, filtered_ship_path, VALID_TYPES):
    df_boat = pd.read_parquet(ship_path)
    df = df_boat.copy() #To count filtered ships at the end
    # -------------------------------
    # MMSI
    # -------------------------------
    df = df[df['MMSI'].notna()]
    df = df[df['MMSI'].astype(str).str.len() == 9]   # MMSI should have 9 numbers
    df = df[df['MMSI'] != 0]

    # -------------------------------
    # VesselType
    # -------------------------------
    df = df[df['VesselType'].isin(VALID_TYPES)]

    # -------------------------------
    # Dimensions (Length, Width)
    # -------------------------------
    df = df[df['Length'].notna() & df['Width'].notna()]
    df = df[(df['Length'] > 0) & (df['Length'] < 400)]
    df = df[(df['Width'] > 0) & (df['Width'] < 60)]

    # -------------------------------
    # Draft
    # -------------------------------
    df = df[df['Draft'].notna()]
    df = df[(df['Draft'] >= 1) & (df['Draft'] <= 25)]

    # -------------------------------
    # TransceiverClass
    # -------------------------------
    # Keep only classe A Ship
    df = df[df['TransceiverClass'] == 'A']

    df_clean = df.reset_index(drop=True)

    print("Nombre de lignes initial :", len(df_boat))
    print("Nombre de lignes filtrées :", len(df_clean))

    df_clean.to_parquet(filtered_ship_path, engine="pyarrow", index=False)

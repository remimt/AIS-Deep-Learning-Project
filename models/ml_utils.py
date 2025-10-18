import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

def generate_dataloader(dataset_path,
                        rad_conv,
                        sog_max,
                        c_max,
                        length_max,
                        width_max,
                        draft_max,
                        batch_size
                        ):
    
    # Load dataframe
    df_dataset = pd.read_parquet(dataset_path)

    
    # Compter les occurrences par MMSI
    mmsi_counts = df_dataset["MMSI"].value_counts(dropna=False)

    # Sélectionner uniquement les MMSI ayant au moins 800 signaux
    valid_mmsi = mmsi_counts[mmsi_counts >= 800].index

    # Tirer un MMSI aléatoire parmi eux
    if len(valid_mmsi) > 0:
        top_mmsi = np.random.choice(valid_mmsi)
        df_dataset = df_dataset[df_dataset["MMSI"] == top_mmsi].reset_index(drop=True)
        print(f"MMSI tiré : {top_mmsi} | Nombre de signaux : {len(df_dataset)}")
    else:
        print("Aucun MMSI n’a au moins 800 valeurs.")
    
    
    # Séries -> tenseurs float32
    #lon_tensor     = torch.tensor(df_dataset["LON"].to_numpy(dtype="float32")) / lon_max
    #lat_tensor     = torch.tensor(df_dataset["LAT"].to_numpy(dtype="float32")) / lat_max
    heading_tensor = torch.tensor(df_dataset["Heading"].to_numpy(dtype="float32")) / rad_conv
    cog_tensor     = torch.tensor(df_dataset["COG"].to_numpy(dtype="float32")) / rad_conv
    sog_tensor     = torch.tensor(df_dataset["SOG"].to_numpy(dtype="float32")) / sog_max
    #phi_tensor     = torch.tensor(df_dataset["PHI"].to_numpy(dtype="float32")) / rad_conv
    length_tensor  = torch.tensor(df_dataset["Length"].to_numpy(dtype="float32")) / length_max
    width_tensor   = torch.tensor(df_dataset["Width"].to_numpy(dtype="float32")) / width_max
    draft_tensor   = torch.tensor(df_dataset["Draft"].to_numpy(dtype="float32")) / draft_max
    u_tensor       = torch.tensor(df_dataset["U"].to_numpy(dtype="float32"))
    v_tensor       = torch.tensor(df_dataset["V"].to_numpy(dtype="float32"))
    
    #velocity_tensor = torch.tensor(np.sqrt(df_dataset["U"].to_numpy(dtype="float32")**2 + df_dataset["V"].to_numpy(dtype="float32")**2))
    #class_velo_tensor = (velocity_tensor > velocity_threshold).float()
    
    #percentage = torch.sum(class_velo_tensor)/len(class_velo_tensor)*100
    #print(f"Percentage of sea surface velocity > {velocity_threshold} m.s-1: {percentage.item()}")

    # Dataset exactly as requested: (heading, sog, phi)
    Train_number = int(len(df_dataset)*0.75)
    dataset = TensorDataset(heading_tensor,
                            cog_tensor,
                            sog_tensor,
                            length_tensor,
                            width_tensor,
                            draft_tensor,
                            u_tensor,
                            v_tensor
                            )
    train_subset = Subset(dataset, range(Train_number))
    eval_subset = Subset(dataset, range(Train_number, len(dataset)))

    # DataLoader
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    print("Heading Tensor Shape: ",heading_tensor.shape)
    print("COG Tensor Shape: ",cog_tensor.shape)
    print("SOG Tensor Shape: ",sog_tensor.shape)
    print("U Tensor Shape: ",u_tensor.shape)
    print("V Tensor Shape: ",v_tensor.shape)
    print("Dataset lenght: ",len(dataset))
    print("Train Subset lenght: ",len(train_subset))

    return loader, eval_subset

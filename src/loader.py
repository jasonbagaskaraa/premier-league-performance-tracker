import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"\n--- {path} ---")
        print("File tidak ditemukan")
        return None
    
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f: 
        first_line = f.readline().strip()
        while not first_line:
            first_line = f.readline().strip()

    if first_line.startswith(","):
        df = pd.read_csv(path, header=1, encoding='utf-8-sig')
    else:
        df = pd.read_csv(path, header=0, encoding='utf-8-sig')
        
    print(f"\n--- {path} ---")
    print(f"Shape: {df.shape}")
    print(df.head(3))

    return df
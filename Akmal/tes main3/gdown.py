import os

# Coba import gdown, jika belum ada maka install terlebih dahulu
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# Daftar file dan ID Google Drive-nya
files = {
    "xgb_model.json": "1Pl5uAdeVv7bxt84B3E8ZIlLlpdDA-SPv",
    "model_input_feature_columns.pkl": "1FAYNNiUVtWhofnBCtQawvWHp165TWxuY",
    "mlb.pkl": "1FBatGRB0WG-NsXMsxYGO5xWja-0U7TLe",
    "le.pkl": "1kjs5TxsD0aTrH7XLp5O0m3fz7IITdQ7q",
    "bins_labels.pkl": "1HN8NV6V5vKSLqSqUE2U06l-joU_Tm11k",
    "recipes_new.csv": "1lL2Xiqfym8s98ku_4mJeRv-89FUkEHJV"
}

# Unduh setiap file dari Google Drive
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {filename}...")
    gdown.download(url, output=filename, quiet=False)
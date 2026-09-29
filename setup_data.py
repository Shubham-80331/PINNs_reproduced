import os
import requests
from tqdm import tqdm

DATA_DIR = "data"

URLS = {
    "burgers_shock.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat",
    "NLS.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/NLS.mat",
    "cylinder_nektar_wake.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat",
    "allen_cahn.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/allen_cahn.mat",
    "kdv.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/kdv.mat",
    "Butcher_IRK100.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/utilities/IRK_weights/Butcher_IRK100.mat",
    "Butcher_IRK500.mat": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/utilities/IRK_weights/Butcher_IRK500.mat"
}

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"{filename} already exists, skipping...")
        return
    
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    else:
        print(f"Failed to download {filename}, status code {response.status_code}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in URLS.items():
        download_file(url, filename)

if __name__ == "__main__":
    main()

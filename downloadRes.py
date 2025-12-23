import os
import requests

save_folder = "downloads"
os.makedirs(save_folder, exist_ok=True)

with open("vehicle_images_input.txt") as f:
    for url in f:
        url = url.strip()
        if not url:
            continue

        filename = os.path.join(save_folder, url.split("/")[-1])

        print(f"Downloading {url}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as file:
                for chunk in r.iter_content(1024):
                    file.write(chunk)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

print("Done!")

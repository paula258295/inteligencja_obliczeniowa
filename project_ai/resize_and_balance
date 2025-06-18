import os
import cv2
import random
import shutil

artists = [
    'Pablo_Picasso', 'Vincent_van_Gogh', 'Rembrandt',
    'Sandro_Botticelli', 'Salvador_Dali', 'Marc_Chagall'
]
input_base = '../artysci/images/images'
output_base = 'images_balanced'
os.makedirs(output_base, exist_ok=True)
n_per_artist = 125
IMG_SIZE = 64

for artist in artists:
    in_dir = os.path.join(input_base, artist)
    out_dir = os.path.join(output_base, artist)
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(in_dir) if f.endswith('.jpg') or f.endswith('.png')]
    selected = random.sample(files, min(n_per_artist, len(files)))
    for f in selected:
        img = cv2.imread(os.path.join(in_dir, f))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(out_dir, f), img)

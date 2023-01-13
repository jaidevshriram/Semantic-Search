import numpy as np
import pandas as pd
import os
import tqdm
from io import BytesIO
import requests
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

photos_df = pd.read_csv("data/photos.tsv000", sep="\t")

def download_img(r):

    name = r['photo_id']
    cur_img = r['photo_url'] + "/download"

    if os.path.exists("data/photos/" + name + ".jpg"):
        return

    try:
        # print(r['photo_url'], cur_img)
        image_bytes = requests.get(cur_img)

        image_bytes = image_bytes.content
        image_stream = BytesIO(image_bytes)
        img_open = np.array(Image.open(image_stream))

        img_open = cv2.cvtColor(img_open, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join("data", "photos", f'{name}.jpg'), img_open)
    except:
        print("Error", cur_img)
    return

from multiprocessing import Pool

pool = Pool(10)
for _ in tqdm.tqdm(pool.imap_unordered(download_img, photos_df.to_dict('records')), total=len(photos_df.to_dict('records'))):
    pass

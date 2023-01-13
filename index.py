import os
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from annoy import AnnoyIndex

imgs = os.listdir("data/photos")

f = 512  # Length of item vector that will be indexed

a = AnnoyIndex(f, 'dot')
a.on_disk_build("unsplash.ann")

indexer = {}

for i in tqdm(range(len(imgs))):

    # img = Image.open(f"data/photos/{imgs[i]}")
    # v = model.encode(img)
    img = imgs[i]
    name = img.split('.')[0]
    ext = img.split('.')[1]
    indexer[i] = img

    if ext not in ['jpg', 'png']:
        continue

    v = np.load(f"features/{name}.npy")

    a.add_item(i, v)

a.build(10)
np.save("index_map.npy", indexer)
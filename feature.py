import os
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from multiprocessing import Pool

model = SentenceTransformer('clip-ViT-B-32')

def extract_feature(img):

    name = img.split('.')[0]
    ext = img.split('.')[1]

    if ext not in ['jpg', 'png']:
        return False

    if os.path.exists(f"features/{name}.npy"):
        return True

    img = Image.open(f"data/photos/{img}")
    v = model.encode(img)
    np.save(f"features/{name}.npy", v)
    return True

if __name__ == '__main__':

    imgs = os.listdir("data/photos")

    pool = Pool(8)

    r = list(tqdm(pool.imap(extract_feature, imgs), total=len(imgs)))
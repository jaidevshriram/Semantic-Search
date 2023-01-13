import os
import numpy as np
import timeit
import streamlit as st

from PIL import Image

from annoy import AnnoyIndex

from sentence_transformers import SentenceTransformer, util

ROWS = 300
COLS = 3
NUM_RESULTS = ROWS * COLS

model = SentenceTransformer('clip-ViT-B-32')

index_map = np.load("index_map.npy", allow_pickle=True).item()

a = AnnoyIndex(512, 'dot')
a.load('unsplash.ann') # super fast, will just mmap the file

st.title("🌎 Text Search")
query = st.text_input("Find an image!", placeholder="Type your query")

query_embed = model.encode(query)

start = timeit.timeit()
results = a.get_nns_by_vector(query_embed, NUM_RESULTS, search_k=-1, include_distances=False)
end = timeit.timeit()

st.text(f"Retrieved {NUM_RESULTS} results in {abs(round(end-start, 2))} seconds from 25k images")

result_imgs = list(map(lambda x: index_map[x], results))

with st.container():
    for i in range(ROWS):

        cols = st.columns(COLS) 

        for j in range(COLS):

            new_width = 500
            new_height = 300

            img = Image.open("data/photos/" + result_imgs[COLS * i + j])

            width, height = img.size   # Get dimensions

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            # Crop the center of the image
            img = img.crop((left, top, right, bottom))

            cols[j].image(img)
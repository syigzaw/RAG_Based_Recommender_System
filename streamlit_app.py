from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import gdown
from concurrent.futures import ThreadPoolExecutor, wait
import os
import zipfile
from PIL import Image

st.set_page_config(layout='wide')

def downloadImagesAndExtractAll(images_id):
    gdown.download(id=images_id)
    with zipfile.ZipFile('./images.zip', 'r') as zip_ref:
        zip_ref.extractall()

@st.cache_resource(show_spinner='Downloading advertisements and vector store. This might take a moment...')
def downloadResources(images_id, faiss_id, pickle_id):
    with ThreadPoolExecutor() as executor:
        futures = []
        vector_store_futures = [(executor.submit(
            HuggingFaceEmbeddings,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            cache_folder='./'
        ))]

        if not os.path.exists('./images/'):
            futures.append(executor.submit(downloadImagesAndExtractAll, images_id))
        if not os.path.exists('./index.faiss'):
            vector_store_futures.extend([executor.submit(gdown.download, id=id) for id in [faiss_id, pickle_id]])
        wait(vector_store_futures)

        futures.append(executor.submit(
            FAISS.load_local,
            './',
            vector_store_futures[0].result(),
            allow_dangerous_deserialization=True
        ))
        wait(futures)

    return futures[-1].result()

@st.cache_data(show_spinner=False)
def run(input, num_of_images):
    if not input or not num_of_images:
        return

    try:
        loader = UnstructuredLoader(web_url=input)
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc.page_content)
        k = num_of_images//len(docs) + 1
    except:
        docs = [input]
        k = num_of_images

    unique_set = set()
    count = 0
    bar = st.progress(0)

    columns = st.columns(num_of_columns)
    column_heights = [0] * len(columns)

    for doc in docs:
        results = vector_store.similarity_search(doc, k=k)
        for res in results:
            image_id = res.metadata['image_id']
            if image_id not in unique_set:
                unique_set.add(image_id)
                image = Image.open(f'./images/{image_id}.jpg')
                image_resized_height = image.height/image.width
                shortest_column = min(
                    range(len(column_heights)),
                    key=column_heights.__getitem__
                )
                column_heights[shortest_column] += image_resized_height
                with columns[shortest_column]:
                    with st.container(border=True):
                        st.image(
                            image,
                            caption=res.metadata['description'],
                            use_container_width=True
                        )
                count += 1
                bar.progress(count/num_of_images)
                if count == num_of_images:
                    break
        else:
            continue
        break
    bar.progress(1.0)

num_of_columns = 5
st.title('RAG-Based Advertisement Recommender System', anchor=False)
input_col, slider_col = st.columns([num_of_columns - 1, 1])
input = input_col.text_input('Enter a URL or text to see recommended ads for it.')
num_of_images = slider_col.slider('Number of ads to show', value=50)

vector_store = downloadResources(
    images_id='10kJgHvweNbmfpl-u8br0_5aiXDffT7Pb',
    faiss_id='1OuRe9DGkVxFi3Lbx4jDecDg2XIfEXC-L',
    pickle_id='13hKvxiWR5WscWK2BxE47CGQFy1A3GvbQ'
)

run(input, num_of_images)
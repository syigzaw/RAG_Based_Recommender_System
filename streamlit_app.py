from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
import os
from PIL import Image
import wget
from io import BytesIO
import duckdb

st.set_page_config(layout='wide')

def callback(progress, bar, label):
    bar.progress(progress, label)

def download(url, bar, label):
    wget.download(url, bar=lambda current, total, width=80: callback(current/total, bar, label))

@st.cache_resource(show_spinner="Streamlit doesn't host large files so they're \
being downloaded from the cloud onto the server. This might take a moment...")
def downloadResources(images_id, faiss_id, pickle_id, base_url):
    with ThreadPoolExecutor() as executor:
        futures = []
        vector_store_futures = [executor.submit(
            HuggingFaceEmbeddings,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            cache_folder='./'
        )]

        if not os.path.exists('./images.parquet'):
            futures.append(executor.submit(
                download,
                f'{base_url}{images_id}',
                progress_bars[0],
                progress_bar_labels[0]
            ))

        if not os.path.exists('./index.faiss'):
            vector_store_futures.extend([executor.submit(
                download,
                f"{base_url}{file['id']}",
                progress_bars[file['bar_index']],
                progress_bar_labels[file['bar_index']]
            ) for file in [
                {'id': faiss_id, 'bar_index': 1},
                {'id': pickle_id, 'bar_index': 2}
            ]])

        for t in executor._threads:
            add_script_run_ctx(t, ctx)

        wait(vector_store_futures)

        futures.append(executor.submit(
            FAISS.load_local,
            './',
            vector_store_futures[0].result(),
            allow_dangerous_deserialization=True
        ))
        wait(futures)

    return futures[-1].result()

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

    count = 0
    bar = st.progress(0)

    columns = st.columns(num_of_columns)
    column_heights = [0] * len(columns)

    for doc in docs:
        results = vector_store.similarity_search(doc, k=k)
        results = sorted({(str(i.metadata['image_id']), i.metadata['description']) for i in results})
        images = con.execute(query.format(
            ', '.join([res[0] for res in results])
        )).fetchall()
        for i in range(len(results)):
            image = Image.open(BytesIO(images[i][1]))
            shortest_column = min(range(len(column_heights)), key=column_heights.__getitem__)
            chars_per_line = 34
            padding = 30
            font_size = 14
            image_resized_height = ((1080 - 160)//num_of_columns - padding)*image.height/image.width
            column_heights[shortest_column] += image_resized_height + (
                len(results[i][1]) + chars_per_line - 1
            )*font_size//chars_per_line + padding + 10
            with columns[shortest_column]:
                with st.container(border=True):
                    st.image(image, caption=results[i][1], use_container_width=True)
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
ctx = get_script_run_ctx()

progress_placeholder = st.empty()
progress_bars = []
progress_bar_labels = [
    'Downloading 288129 advertisement images (2.02 GB):',
    'Downloading vector store embeddings (438 MB):',
    'Downloading vector store metadata (39.6 MB):'
]
columns = progress_placeholder.columns(len(progress_bar_labels))
for i in range(len(progress_bar_labels)):
    with columns[i]:
        progress_bars.append(st.progress(0, text=progress_bar_labels[i]))

vector_store = downloadResources(
    images_id='1FiL43xxu5Qs7Jw_XkVtgZH8w4BEovNfX',
    faiss_id='12bHrpLaFNnCyfJqMTg9igVRuNACWI83D',
    pickle_id='113mbnWZQ1yHVIoNOhNitxCQKYQ7rBREk',
    base_url='https://drive.usercontent.google.com/download?export=download&confirm=t&id='
)

progress_placeholder.empty()

con = duckdb.connect()
query = "SELECT * FROM 'images.parquet' WHERE n IN ({})"

run(input, num_of_images)
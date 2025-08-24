from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader
from google.oauth2.service_account import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient import discovery, http
import streamlit as st
from io import BytesIO
from PIL import Image
import json

st.set_page_config(layout='wide')

@st.cache_resource(show_spinner='Loading vector store...')
def loadResources():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_folder='./')
    vector_store = FAISS.load_local('./', embeddings, allow_dangerous_deserialization=True)

    service_account_key = st.secrets['GOOGLE_SERVICE_ACCOUNT_KEY']
    service_account_info = json.loads(service_account_key)
    credentials = Credentials.from_service_account_info(service_account_info)

    folder_id = st.secrets['FOLDER_ID']

    return vector_store, credentials, folder_id

def getImages(result):
    service = discovery.build('drive', 'v3', credentials=credentials, cache_discovery=False)
    file = service.files().list(
        q=f"'{folder_id}' in parents and name = '{result[0]}'",
        spaces='drive',
        fields='files(id)'
    ).execute().get("files", [])

    if file:
        request = service.files().get_media(fileId=file[0]['id'])
        file = Image.open(BytesIO(request.execute()))

    return file, result[1]

def run(input, num_of_images):
    if not input or not num_of_images:
        return

    try:
        loader = UnstructuredLoader(web_url=input)
        docs = [doc.page_content for doc in loader.lazy_load()]
        if len(docs) < 10:
            st.text("Access to this website was blocked.")
            return
        k = num_of_images//len(docs) + 1
    except:
        docs = [input]
        k = num_of_images

    unique_set = set()
    count = 0
    bar = st.progress(0)

    columns = st.columns(num_of_columns)
    column_heights = [0] * len(columns)

    results = []
    for doc in docs:
        search_results = vector_store.similarity_search(doc, k=k)
        for search_result in search_results:
            result = (f"{search_result.metadata['image_id']}.jpg", search_result.metadata['description'])
            if result not in unique_set:
                results.append(result)
                unique_set.add(result)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(getImages, result) for result in results]
        for future in as_completed(futures):
            image, description = future.result()
            if image:
                shortest_column = min(range(len(column_heights)), key=column_heights.__getitem__)
                # Attempting to keep columns at a similar size
                # Hard-coding estimates for pixel values because Streamlit doesn't give access to the DOM
                chars_per_line, padding, font_size = 34, 30, 14
                image_resized_height = ((1080 - 160)//num_of_columns - padding)*image.height/image.width
                column_heights[shortest_column] += image_resized_height + (
                    len(description) + chars_per_line - 1
                )*font_size//chars_per_line + padding + 10
                with columns[shortest_column]:
                    with st.container(border=True):
                        st.image(image, caption=description, use_container_width=True)
                count += 1
                bar.progress(count/num_of_images)
                if count == num_of_images:
                    break

    bar.progress(1.0)

num_of_columns = 5
st.title('RAG-Based Advertisement Recommender System', anchor=False)
input_col, slider_col = st.columns([num_of_columns - 1, 1])
input = input_col.text_input('Enter a URL or text to see recommended ads for it.')
num_of_images = slider_col.slider('Number of ads to show', value=50)

vector_store, credentials, folder_id = loadResources()

run(input, num_of_images)
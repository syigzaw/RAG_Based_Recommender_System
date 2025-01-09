# RAG-Based Advertisement Recommender System

This project uses vector similarity search on embedded queries from website content or text in order to return and display relevant advertisements. Check it out [here](https://ragrecommendersystem.streamlit.app/).

This content-based filtering recommender system allows you to serve relevant advertisements for a website without needing to track users, which removes the need for collecting user data.

In order to use, run the following:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This will download 3 files into the repository from Google Drive, one of which is 2.02 GB (the file containing the images for the advertisements).

If you would like to see my exploration process for getting this to work, you can check out the jupyter notebook in this repo. The original advertisement dataset that was used to create this project and is referenced in the notebook can be found [here](https://github.com/microsoft/CommercialAdsDataset/).

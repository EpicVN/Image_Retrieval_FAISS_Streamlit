# Image_Retrieval_FAISS_Streamlit
 
## Prepare the environment
Install the necessary dependencies by running:
```
        pip install -r requirements.txt
```
## Prepare the dataset

Put the downloaded [The Paris Dataset](https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset) in **./dataset/products**

```
Main-folder/
│
├── dataset/ 
|   |
│   ├── feature
|   |   ├── Resnet50.index.bin
|   |
|   └── products
|       ├── ..._0.jpg
|       ├── ..._1.jpg
|       └── ...
|   
└── ...
```

## Running the code

### Feature extraction (Indexing)

    python indexing.py --feature_extractor Resnet50
    
The Resnet50.index.bin file will be at **Main-folder/dataset/feature**.
    
### Run with streamlit interface:

    streamlit run app.py

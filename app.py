import time
import torch
import faiss
import pathlib
from PIL import Image

import streamlit as st

from utils.feature_extraction import MyResnet50
from utils.dataloader import get_transformation

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Configure page layout
st.set_page_config(
    page_title="Image Retrieval with FAISS",
    layout="wide",  # Ensures content spans the full width of the screen
    initial_sidebar_state="collapsed",
)

device = torch.device('cpu')
image_root = './dataset/products'
feature_root = './dataset/feature'


def get_image_list(image_path):
    image_path = pathlib.Path(image_path)
    image_list = list()
    for image_path in image_path.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: x.name)
    return image_list


def retrieve_image(img):
    extractor = MyResnet50(device)

    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + 'Resnet50' + '.index.bin')

    _, indices = indexer.search(feat, k=11)

    return indices[0]


def display_images_in_grid(image_paths, num_columns=3):
    for row in range(0, len(image_paths), num_columns):
        cols = st.columns(num_columns)
        for idx, img_path in enumerate(image_paths[row:row + num_columns]):
            with cols[idx]:
                image = Image.open(img_path)
                st.markdown(
                    f"""
                        <div class="image-container">
                            <img {st.image(image)}"/>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )


# Áp dụng CSS cho màu sắc
def apply_custom_css():
    st.markdown(
        """
        <style>
        
        /* Header */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        .title {
            margin-top: -40px; 
            margin-bottom: 50px;
            color: 2c3e50;   
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        
        .upload {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        
        .tutorial {
            display: flex;
            color: #A59D9D;
            text-align: center;
            justify-content: center;
            align-items: center;
            margin-top: 150px; 
            font-size: 20px;
            font-weight: 400;
            border: 5px solid #a3d9e1;
            border-radius: 15px;
            padding: 20px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #dff3f5;
        }

        /* Button */
        button {
            background-color: #a3d9e1 !important;
            color: #ffffff !important;
        }
        
        button:hover {
            background-color: #7bbcc7 !important;
        }
        
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_custom_css()

    title = 'IMAGE RETRIEVAL WITH FAISS'

    st.markdown(f'<div class="title">{title}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.sidebar.markdown(f'<div class="upload">UPLOAD IMAGE</div>', unsafe_allow_html=True)

        img_file = st.sidebar.file_uploader(label=".", type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)

            # Manipulate cropped image at will
            st.sidebar.title("Preview 	:speaking_head_in_silhouette:")
            img.thumbnail((150, 150))
            st.sidebar.image(img)

            start = time.time()

            st.markdown('**Retrieving .......**')

            with st.spinner("Retrieving images..."):
                retrieve = retrieve_image(img)

            image_list = get_image_list(image_root)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

    if not img_file:
        st.markdown(f'<div class="tutorial">Upload an image to start retrieval</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Loading images..."):
            top_retrieved_images = [image_list[i] for i in retrieve[:9]]
            display_images_in_grid(top_retrieved_images, num_columns=3)


if __name__ == '__main__':
    main()

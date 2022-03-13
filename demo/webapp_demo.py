from argparse import ArgumentParser
from typing import List

import httpx
import streamlit as st

st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    """,
    unsafe_allow_html=True,
)


parser = ArgumentParser("BiWAKO Demo Web App")
parser.add_argument(
    "-s", "--server", help="address of inference server", default="8000", type=str
)


args = parser.parse_args()


st.title("BiWAKO Web Service Demo")


@st.cache
def get_available_models() -> List[str]:
    tmp = httpx.get(f"{BACKEND_HOST}/models").json()
    return tmp["models"]


BACKEND_HOST = "http://localhost:" + args.server
MODELS = get_available_models()


# side bar menu
with st.sidebar:
    st.title("Choose Model")
    choise = st.selectbox("Model", MODELS)
    st.markdown("---")
    if st.button("Delete Result"):
        r = httpx.post(f"{BACKEND_HOST}/delete")
        st.success(r.json())


image_files = st.file_uploader(
    "Choose an image file", type=["jpg", "png"], accept_multiple_files=True
)

if image_files and len(image_files) > 0 and st.button("Request prediction"):
    files = []
    for im in image_files:
        st.image(im, use_column_width=True, channels="BGR")
        files.append(("files", im))

    r = httpx.post(
        f"{BACKEND_HOST}/predict",
        files=files,
        params={"model_choise": choise},
        timeout=300,
    )
    st.success(r.json())

if image_files and len(image_files) > 0 and st.button("Request Result"):
    r = httpx.get(f"{BACKEND_HOST}/result")
    r = r.json()
    st.write(r)
    for im in r:
        st.image(r[im], use_column_width=True, channels="BGR")

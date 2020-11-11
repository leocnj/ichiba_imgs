# streamlit app demo
#
from collections import namedtuple
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from googletrans import Translator

st.title("IChiba Category Wizard")

DATA_URL = "data/ichiba_test_strict_100316_more.tsv.gz"
LABEL_CSV = "data/labels.csv"
Lab_name = namedtuple("lab_name", ["jpn", "eng"])


@st.cache
def load_data(nrows):
    data = pd.read_csv(
        DATA_URL,
        nrows=nrows,
        compression="gzip",
        header=None,  # no header in tsv
        sep="\t",
        quotechar='"',
    )
    selected = data.iloc[:, [0, 1, 8]]  # title, label-path, img_url
    selected.columns = ["title", "label", "image_url"]

    labels = pd.read_csv(
        LABEL_CSV,
        header=None,
    )
    labels.columns = ["row", "lab", "jpn", "eng"]
    lab_dict = {
        lab: Lab_name(labels["jpn"][row], labels["eng"][row])
        for row, lab in enumerate(labels["lab"])
    }
    return selected, lab_dict


data_load_state = st.text("Loading data...")
data, labels = load_data(1000)
data_load_state.text("Done! (using st.cache)")


@st.cache
def load_model():
    clf_model = tf.keras.models.load_model("data/100316")
    match_model = tf.keras.models.load_model("data/100316_matcher")
    return clf_model, match_model


model_load_state = st.text("Loading model...")
clf_model, match_model = load_model()
model_load_state.text("Done! (using st.cache)")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data)

st.subheader("Choose one product to show")

translator = Translator()
item_idx = st.slider("item_idx", 0, 999, 10)
item_data = data.iloc[item_idx, :]
t_, l_ = item_data["title"], item_data["label"]
t_zh = translator.translate(t_, dest="zh-CN").text
t_en = translator.translate(t_, dest="en").text

st.markdown(f'- *Title*:{t_}\n- *中文*:{t_zh}\n- *English*:{t_en}\n- *Label*:{labels[l_]}')
st.image(item_data["image_url"], width=None)

st.subheader("IC predicts category based on product title")

label_pred = clf_model(tf.constant([t_]))
st.write(labels[label_pred.numpy()[0].decode()])

st.subheader("Check merchant category selection")
inv_labels = {v: k for k, v in labels.items()}
label_merchant = st.selectbox("Merchant chooses a category", list(labels.values()))
st.write("Chosen:", label_merchant)
match_pred = match_model([tf.constant([t_]), tf.constant([inv_labels[label_merchant]])])
st.write(match_pred)

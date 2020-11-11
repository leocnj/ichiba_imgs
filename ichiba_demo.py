# streamlit app demo
# 
from collections import namedtuple
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf


st.title('IChiba Category Wizard')

DATA_URL = ('data/ichiba_test_strict_100316_more.tsv.gz')
LABEL_CSV = ('data/labels.csv')
Lab_name = namedtuple('lab_name', ['jpn', 'eng'])
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
    selected = data.iloc[:,[0,1,8]] # title, label-path, img_url
    selected.columns = ['title', 'label', 'image_url']

    labels = pd.read_csv(
        LABEL_CSV,
        header=None,
    )
    labels.columns = ['row', 'lab', 'jpn', 'eng']
    lab_dict = {lab: Lab_name(labels['jpn'][row], labels['eng'][row]) for row, lab in enumerate(labels['lab'])}
    return selected, lab_dict

data_load_state = st.text('Loading data...')
data, labels = load_data(1000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Choose one product to show')
item_idx = st.slider('item_idx', 0, 999, 10)
item_data = data.iloc[item_idx,:]
st.write(item_data['title'], labels[item_data['label']])
st.image(item_data['image_url'], width=None)

st.subheader('IC predicts category based on product title')
clf_model = tf.keras.models.load_model('data/100316')
label_pred = clf_model(tf.constant([item_data['title']]))
st.write(labels[label_pred.numpy()[0].decode()])

st.subheader('Check merchant category selection')
label_merchant = st.selectbox('Merchant chooses a category', list(labels.keys()))
st.write('Chosen:', labels[label_merchant])
match_model = tf.keras.models.load_model('data/100316_matcher')
match_pred = match_model([tf.constant([item_data['title']]), tf.constant([label_merchant])])
st.write(match_pred)


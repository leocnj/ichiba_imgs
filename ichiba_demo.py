# streamlit app demo
# 

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title('IChiba Category Wizard')

DATA_URL = ('data/ichiba_test_strict_100316_more.tsv.gz')

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
    return selected

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Choose one product to show')
item_idx = st.slider('item_idx', 0, 999, 10)
item_data = data.iloc[item_idx,:]
st.write(item_data[['title','label']])
st.image(item_data['image_url'], width=None)

st.subheader('Use DL model to predict label')
clf_model = tf.keras.models.load_model('data/100316')
label_pred = clf_model(tf.constant([item_data['title']]))
st.write(label_pred.numpy())

labels = list(set(data['label']))
label_idx = st.slider('label_idx', 0, len(labels), 0)
label_merchat = labels[label_idx]
st.write(label_merchat)
match_model = tf.keras.models.load_model('data/100316_matcher')
match_pred = match_model([tf.constant([item_data['title']]), tf.constant([label_merchat])])
st.write(match_pred)


import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    
#title
st.title('Diagnosis to the patients with their chest X-ray')

image = Image.open('image.jpg')
st.image(image, caption='Pneumania')

st.warning("Upload photo of the Chest X-ray!", icon="⚠️")

st.text("For example:")
sample = Image.open('sample.jpeg')
st.image(image, caption='Sample')

# rasmni joylash
file = st.file_uploader("Upload photo:", type=["png","jpeg","gif","svg"])
if file:
    st.image(image=file)

    # PIL convert
    img = PILImage.create(file)


    # model
    model = load_learner("pneumaniaClassification_model.pkl")

    # prediction
    prediction, pred_id, probs= model.predict(img)
    st.success(f"Prediction: {prediction}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)





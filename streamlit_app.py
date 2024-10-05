from Determine_Fame import *
import joblib
import streamlit as st
import numpy as np
import keras
import cv2
from PIL import Image

forest = joblib.load("Categorical.joblib")
image_regressor = keras.models.load_model("CNN-Beta.h5")

def load_input():
    st.title("ArtInnovate Beta")
    
    uploaded_file = st.file_uploader(label='Upload Your Painting')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image_array = np.array(Image.open(uploaded_file).convert("RGB"))
    else:
        image_array = None

    col1, col2 = st.columns(2)
    with col1:
        artist_name = st.text_input("Enter Artist Name")
    with col2:
        creation_year = st.text_input("Enter Creation Year")

    st.markdown("###### If Your Painting Has These Characteristics, Please Enter 1 or 0")
    modern_art = st.text_input("Modern Art")
    abstract_art = st.text_input("Abstract Art")
    expressionism = st.text_input("Expressionism")
    feminist_art = st.text_input("Feminist Art")
    conceptual = st.text_input("Conceptual Art")
    geometric_art = st.text_input("Geometric Art")
    cubism = st.text_input("Cubism")
    environmental_art = st.text_input("Environmental Art")

    submit = st.button('Calculate Price')

    if submit:
        if not image_array:
            st.error("Please upload your painting.")
        elif not artist_name:
            st.error("Please enter the artist name.")
        elif not creation_year:
            st.error("Please enter the creation year.")
        else:
            fame = len(biography_artsy(artist_name).split())
            input_data = [fame, creation_year, modern_art, abstract_art, expressionism,
                          feminist_art, conceptual, geometric_art, cubism, environmental_art]
            image_array = cv2.resize(image_array, (150, 150), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, axis=0)
            loss = 0.99
            image_regr_price = image_regressor.predict(image_array)
            forest_price = forest.predict([input_data])
            
            artist_url = f"https://www.artsy.net/artist/{artist_name.replace(' ', '-')}"
            st.markdown(f"**[Click Here to Open Artist Page]({artist_url})**")
            st.text("Estimated Price of Painting: ₹" + str(image_regr_price * (1 - loss) + loss * forest_price))


def main():
    load_input()


if __name__ == '__main__':
    main()

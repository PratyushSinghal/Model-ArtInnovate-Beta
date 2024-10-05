import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import unicodedata
import streamlit as st

artists_data = pd.DataFrame()
artists, artists_fame, unavailable_artists = [], [], []
unique_artists = {}

# Configure Headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox") 
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

def clean_string(text):
    nfkd_string = unicodedata.normalize("NFKD", str(text))
    text = "".join([c for c in nfkd_string if not unicodedata.combining(c)])
    return text

def open_artist_page(artist_name):
    base_url = "https://laasyaart.com/"
    artist_url = base_url + artist_name.lower().replace(" ", "-") + "/"
    return artist_url


def biography_laasyaart(artist):
    artist_url = open_artist_page(artist)
    driver.get(artist_url)
    time.sleep(2)

    try:
        biography = driver.find_element("xpath", "//div[@class='elementor-widget-container']").text
        return biography
    except Exception as e:
        print(f"Biography not found for {artist}: {e}")
        unavailable_artists.append(artist)
        return ""


def setup():
    global artists_data
    artists_data = pd.read_csv("artinnovate_dataset.csv")
    global artists
    artists = artists_data["artist"].tolist()


def generate_artist_fame():
    for artist in artists:
        if artist not in unique_artists:
            biography = biography_laasyaart(artist)
            fame = len(biography.split())
            artists_fame.append(fame)
            unique_artists[artist] = fame
        else:
            artists_fame.append(unique_artists[artist])

    unavailable_artists_file = open("unavailable_artists.txt", "w")
    unavailable_artists_file.write(str(unavailable_artists))
    while len(artists_fame) < len(artists_data):
        artists_fame.append(None)
    artists_data.insert(artists_data.columns.get_loc("artist") + 1, "fame", artists_fame)
    artists_data.to_csv("artinnovate_artist-fame.csv")


if __name__ == "__main__":
    setup()
    generate_artist_fame()
    driver.quit()

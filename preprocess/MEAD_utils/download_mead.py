import gdown
url = "https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j"
gdown.download_folder(url, output='/xxxx/MEAD',use_api=True)
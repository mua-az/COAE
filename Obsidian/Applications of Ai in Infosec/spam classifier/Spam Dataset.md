




# DOWNLOADING ZIP FILE

import re
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)

zip_file = zipfile.ZipFile(io.BytesIO(response.content))

zip_file.extractall("Dataset")

extracted_files = os.listdir("Dataset")

print("Extracted Files : " , extracted_files)



# BASIC ENUMERATION OF DATABASE


import pandas as pd
import numpy as np

data = pd.read_csv("Dataset/SMSSpamCollection",
                   sep="\t",
                   header=None,
                   names=["label", "message"],)

print("--------------------------------HEAD----------------------------------")

print(data.head())

print("-----------------------------DESCRIPTION------------------------------")

print(data.describe())

print("--------------------------------INFO----------------------------------")

print(data.info())

print("---------------------------MISSING VALUES-----------------------------")

print(data.isnull().sum())

print("--------------------------DUPLICATE ENTRIES---------------------------")

print(data.duplicated().sum())

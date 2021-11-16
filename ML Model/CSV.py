import pandas as pd
import requests
from requests.api import get

def get_Data_file():
    csv_url = input("Enter Training Data URL (CSV Format) : " )
    req = requests.get(csv_url)
    url_content = req.content
    csv_file = open('Data.csv', 'wb')

    csv_file.write(url_content)
    csv_file.close()

def update_Data_file():
    open('Data.csv', 'w').close()
    get_Data_file()


def get_Evaluate_file():
    csv_url = input("Enter Testing Data URL (CSV Format) : " )
    req = requests.get(csv_url)
    url_content = req.content
    csv_file = open('Evaluate.csv', 'wb')

    csv_file.write(url_content)
    csv_file.close()

def update_Evaluate_file():
    open('Evaluate.csv', 'w').close()
    get_Data_file()


        



    



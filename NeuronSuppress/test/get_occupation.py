import os
import csv
import pandas as pd

file_path = "./data/laion_occupation.csv"


data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')

occupations = []

for occ in data['occupation']:
    if occ not in occupations:
        occupations.append(occ)
    else:
        pass


with open('./data/prompt.csv', mode='w', newline='', encoding='utf-8') as fp:
    writer = csv.writer(fp)
    for row in occupations:
        writer.writerow(["a photo of a person who is a " + row])
    
print("Done")
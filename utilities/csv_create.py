import csv, os, re

csv_list = [["Filename","label"]]
for filename in os.listdir():
    try:
        csv_list.append([filename,re.search('.*_(.*).jpg',filename).group(1)])
    except:
        print(f"Skipping {filename}")

with open("labels.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)
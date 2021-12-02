import csv
import os
import re

# set this to True to use the same label for both rc and lc
mix_rc_lc = True
label_map = {"0": 0, "1": 1, "3": 3, "5": 5, "E": 10, "R": 11, "L": 12}

csv_list = [["filename", "label"]]
for filename in os.listdir():
    try:
        label = re.search('.*_(.?)', filename).group(1).upper()
        if mix_rc_lc and label == "L":
            label = "R"
        csv_list.append([filename, label_map[label]])
    except:
        print(f"Skipping {filename}")

with open("labels.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)

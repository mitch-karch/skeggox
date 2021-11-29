import csv
import cv2
import pandas as pd

alldf = pd.read_csv(f'train.csv')
all_rc = alldf[alldf['label'] == 11]
all_lc = alldf[alldf['label'] == 12]
flip_dir = 'flipped/'
imgs_dir = 'imgs'
csv_list = [["filename", "label"]]
for c in [all_lc, all_rc]:
    for i in range(len(c)):
        record = c.iloc[i]
        filename = record['filename']
        img_path = (f'{imgs_dir}/{filename}')
        image = cv2.imread(img_path)
        flipped_image = cv2.flip(image, 1)

        flipped_name = filename.split(".")[0] + "-m" + ".jpg"
        csv_list.append([flipped_name, record['label']])
        cv2.imwrite(flip_dir + flipped_name, flipped_image)

with open("flipped_labels.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class AttributesDataset():
    def __init__(self, annotation_path):

        season_labels = []
        brand_labels = []
        typ_labels = []
        year_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                season_labels.append(row['season'])
                brand_labels.append(row['brand'])
                typ_labels.append(row['type'])
                year_labels.append(row['year'])
        
        self.season_label = np.unique(season_labels)
        self.brand_label = np.unique(brand_labels)
        self.typ_label = np.unique(typ_labels)
        self.year_label = np.unique(year_labels)

        self.num_season = len(self.season_label)
        self.num_brand = len(self.brand_label)
        self.num_typ = len(self.typ_label)
        self.num_year = len(self.year_label)

        self.season_id_to_name = { i:n for i,n in enumerate(self.season_label)}
        self.brand_id_to_name = { i:n for i,n in enumerate(self.brand_label)}
        self.typ_id_to_name = { i:n for i,n in enumerate(self.typ_label)}
        self.year_id_to_name = { i:n for i,n in enumerate(self.year_label)}

        self.season_name_to_id = { n:i for i,n in enumerate(self.season_label)}
        self.brand_name_to_id = { n:i for i,n in enumerate(self.brand_label)}
        self.typ_name_to_id = { n:i for i,n in enumerate(self.typ_label)}
        self.year_name_to_id = { n:i for i,n in enumerate(self.year_label)}

class RunwayDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()
        self.transform = transform
        self.attr = attributes

        self.data = []
        self.season_labels = []
        self.brand_labels = []
        self.typ_labels = []
        self.year_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.season_labels.append(self.attr.season_name_to_id[row['season']])
                self.brand_labels.append(self.attr.brand_name_to_id[row['brand']])
                self.typ_labels.append(self.attr.typ_name_to_id[row['typ']])
                self.year_labels.append(self.attr.year_name_to_id[row['year']])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        dict_data = {
                'img' : img,
                'labels' : {
                    'season_label' : self.season_labels[idx],
                    'brand_label' : self.brand_labels[idx],
                    'typ_label' : self.typ_labels[idx],
                    'year_label' : self.year_labels[idx],
                    }
                
                }
        return dict_data

if __name__ == '__main__':
    c = AttributesDataset('vogue.csv')
    print(c.num_season)
    print(c.num_brand)
    print(c.num_typ)
    print(c.num_year)


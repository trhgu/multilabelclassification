import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset, RunwayDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

def validate(model, dataloader, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_season = 0
        accuracy_brand = 0
        accuracy_typ = 0
        accuracy_year = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = { t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_season, batch_accuracy_brand, batch_accuracy_typ, batch_accuracy_year = calculate_metrics(output, target_labels)


            accuracy_season +=  batch_accuracy_season
            accuracy_brand += batch_accuracy_brand 
            accuracy_typ += batch_accuracy_typ 
            accuracy_year += batch_accuracy_year 

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_season /=  n_samples
    accuracy_brand /= n_samples
    accuracy_typ /=  n_samples
    accuracy_year /=  n_samples
    print('>'*70)
    print("Validation  loss: {:.4f}, season: {:.4f}, brand: {:.4f}, typ: {:.4f}, year: {:4f}\n".format(avg_loss, accuracy_season, accuracy_brand, accuracy_typ, accuracy_year))
    
    return avg_loss, accuracy_season, accuracy_brand, accuracy_typ, accuracy_year

    model.train()


def calculate_metrics(output, target):
    _, predicted_season = output['season'].cpu().max(1)
    _, predicted_brand = output['brand'].cpu().max(1)
    _, predicted_typ = output['typ'].cpu().max(1)
    _, predicted_year = output['year'].cpu().max(1)

    gt_season = target['season_label'].cpu()
    gt_brand = target['brand_label'].cpu()
    gt_typ = target['typ_label'].cpu()
    gt_year = target['year_label'].cpu()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        accuracy_season = balanced_accuracy_score(gt_season.numpy(), predicted_season.numpy())
        accuracy_brand = balanced_accuracy_score(gt_brand.numpy(), predicted_brand.numpy())
        accuracy_typ = balanced_accuracy_score(gt_typ.numpy(), predicted_typ.numpy())
        accuracy_year = balanced_accuracy_score(gt_year.numpy(), predicted_year.numpy())

    return accuracy_season, accuracy_brand, accuracy_typ, accuracy_year






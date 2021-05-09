import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset, RunwayDataset, mean, std
from model import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader

def checkpoint_save(model, name, loss):
    f = os.path.join(name, 'checkpoint-{:04f}.pth'.format(loss))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)

os.makedirs('results/', exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./vogue.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 50
    batch_size = 24
    num_workers = 8 
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    attributes = AttributesDataset(args.attributes_file)
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    train_dataset = RunwayDataset('split/train.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = RunwayDataset('split/test.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MultiOutputModel(num_season=attributes.num_season,
                            num_brand=attributes.num_brand,
                            num_typ=attributes.num_typ,
                            num_year=attributes.num_year).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    n_train = len(train_dataloader)

    print("Starting training ... " )

    for epoch in range(start_epoch, N_epochs+1):
        total_loss = 0
        accuracy_season = 0
        accuracy_brand = 0
        accuracy_typ = 0
        accuracy_year = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t:target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))
            
            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            
            batch_accuracy_season, batch_accuracy_brand, batch_accuracy_typ, batch_accuracy_year = calculate_metrics(output, target_labels)

            accuracy_season +=  batch_accuracy_season
            accuracy_brand += batch_accuracy_brand 
            accuracy_typ += batch_accuracy_typ 
            accuracy_year += batch_accuracy_year 

            loss_train.backward()
            optimizer.step()

        tatal_accuracy = ((accuracy_season/n_train)+(accuracy_brand/n_train)+(accuracy_typ/n_train)+(accuracy_year/n_train))/4

        print("epoch {:4d}, loss: {:.4f}, season: {:.4f}, brand: {:.4f}, typ: {:.4f}, year: {:.4f}".format(
            epoch, total_loss/n_train, accuracy_season/n_train, accuracy_brand/n_train, accuracy_typ/n_train, accuracy_year/n_train))

        if epoch % 5 == 0 :
            avg_loss, accuracy_season, accuracy_brand, accuracy_typ, accuracy_year = validate(model, val_dataloader, epoch, device, checkpoint=None)
            val_accuracy =  (accuracy_season+accuracy_brand+accuracy_typ+accuracy_year)/4
            print("val_total_accuracy : ", val_accuracy)
        
            if  val_accuracy > tatal_accuracy:
                checkpoint_save(model, "results", val_accuracy)


        


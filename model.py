import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiOutputModel(nn.Module):
    def __init__(self, num_season, num_brand, num_typ, num_year):
        super().__init__()
        self.base_model = models.mobilenet_v2().features # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel # size of the layer before classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create classifier
        self.season = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=last_channel, out_features=num_season)
                )
        self.brand = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=last_channel, out_features=num_brand)
                )
        self.typ = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=last_channel, out_features=num_typ)
                )
        self.year = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=last_channel, out_features=num_year)
                )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return {
                "season" : self.season(x),
                "brand" : self.brand(x),
                "typ" : self.typ(x),
                "year" : self.year(x),
                }

    def get_loss(self, net_output, ground_truth):
        season_loss = F.cross_entropy(net_output['season'], ground_truth['season_label'])
        brand_loss = F.cross_entropy(net_output['brand'], ground_truth['brand_label'])
        typ_loss = F.cross_entropy(net_output['typ'], ground_truth['typ_label'])
        year_loss = F.cross_entropy(net_output['year'], ground_truth['year_label'])
        loss = season_loss + brand_loss + typ_loss + year_loss
        return loss, {'season' : season_loss, 'brand' : brand_loss, 'typ' : typ_loss, 'year' : year_loss}


from torch import nn
import torch.nn.init as init
import torch
from torch.nn import functional as F
import pytorch_lightning as  lg

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.12)
        self.fc3 = nn.Linear(128, 40)
        self.fc4p1 = nn.Linear(40+576, 10)


    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 3))
        x1 = x.view(-1, 576)
        x = F.leaky_relu(self.fc1(x1))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4p1(torch.cat((x, x1), dim=1))
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0001)
            elif isinstance(m, nn.Linear):
                init.he_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0001)

class MyModel(lg.LightningModule):  
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        accuracy = (torch.argmax(y_pred, dim=1) == y).float().mean()
        self.log('train_acc', accuracy)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        accuracy = (torch.argmax(y_pred, dim=1) == y).float().mean()
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
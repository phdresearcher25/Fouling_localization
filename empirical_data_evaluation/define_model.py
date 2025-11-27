import torch.nn as nn


class BasicNetwork(nn.Module):
    def __init__(self):
        super(BasicNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(300, 900),
            nn.ReLU(),
            nn.BatchNorm1d(900)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(900, 900),
            nn.ReLU(),
            nn.BatchNorm1d(900)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(900, 2400),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.fc1(x)        
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x




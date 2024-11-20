from torch import nn

class SimpleAutoencoder(nn.Module):
    def __init__(self,
                input_shape):
        super(SimpleAutoencoder, self).__init__()
        self.input_shape = input_shape
        
        #Model layers
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, input_shape),
            nn.Tanh(),
            nn.Linear(input_shape, input_shape),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        encoded = self.encoder(input_data)
        return self.decoder(encoded)


class SimpleAutoencoder_Iris(nn.Module):
    def __init__(self,
                input_shape):
        super(SimpleAutoencoder_Iris, self).__init__()

        #Model layers
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, input_shape),
            nn.Tanh(),
            nn.Linear(input_shape, input_shape)
        )

    def forward(self, input_data):
        encoded = self.encoder(input_data)
        return self.decoder(encoded)


class LinearAutoencoder(nn.Module):
    def __init__(self,
                input_shape):
        super(LinearAutoencoder, self).__init__()

        #Model layers
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, input_shape),
        )

    def forward(self, input_data):
        encoded = self.encoder(input_data)
        return self.decoder(encoded)
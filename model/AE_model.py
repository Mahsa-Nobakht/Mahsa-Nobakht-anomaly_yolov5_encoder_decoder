import torch
from torchsummary import summary
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 512, kernel_size=3, stride=4),
        #     nn.Tanh(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        #     nn.Conv2d(512, 256, kernel_size=3, stride=4),
        #     nn.Tanh(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        #     nn.Conv2d(256, 128, kernel_size=3, stride=4),
        #     nn.Tanh()
        # )
        # Encoder
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=4)
        self.act1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.Tanh()

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 256, kernel_size=3, stride=4),
        #     nn.Tanh(),
        #     nn.MaxUnpool2d(kernel_size=2, stride=2), 
        #     nn.ConvTranspose2d(128, 256, kernel_size=3, stride=4),
        #     nn.Tanh(),
        #     nn.MaxUnpool2d(kernel_size=2, stride=2), 
        #     nn.ConvTranspose2d(256, 512, kernel_size=3, stride=4),
        #     nn.Sigmoid()
        # )
        # Decoder
        self.unpool1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)
        self.act4 = nn.Tanh()
        # self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)  
        self.deconv1 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1,padding=1)  
        self.act5 = nn.Tanh()  
        self.unpool2 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2)
        self.act6 = nn.Tanh()
        # self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.maxunpool2 = nn.UpsamplingBilinear2d(scale_factor=2)   
        self.deconv2 = nn.ConvTranspose2d(512, 1, kernel_size=3, stride=4, padding=0)
        self.act7 = nn.Sigmoid()

    def forward(self, input):
        # x1 = input 
        # # encoded, indices = self.encoder(x)
        # x = self.encoder(input)
        # x = self.decoder(x)
        # return x, torch.nn.MSELoss()(x, input)
    # Encoder
        conv1_output = self.act1(self.conv1(input))
        maxpool1_output, maxpool1_indices = self.maxpool1(conv1_output)
        conv2_output = self.act2(self.conv2(maxpool1_output))
        maxpool2_output, maxpool2_indices = self.maxpool2(conv2_output)
        encoded = self.act3(self.conv3(maxpool2_output))

        # Decoder
        unpool1_output = self.act4(self.unpool1(encoded))
        # unpool1_output = self.maxunpool1(deconv1_output, maxpool2_indices)
        deconv1_output = self.act5(self.deconv1(unpool1_output))
        unpool2_output = self.act6(self.unpool2(deconv1_output))
        # unpool2_output = self.maxunpool2(deconv2_output, maxpool1_indices)
        decoded = self.act7(self.deconv2(unpool2_output))

        # decoded = self.act6(self.deconv3(unpool2_output))
        

        return decoded, torch.nn.MSELoss()(decoded, input)
        # return decoded
    # def forward(self, input):
    #     x, indices1, indices2, indices3 = self.encoder(input)  # Return indices from the encoder
    #     x = self.decoder(x, (indices3, indices2, indices1))  # Pass indices to the decoder
    #     return x, torch.nn.MSELoss()(x, input)

if __name__ == '__main__':
    device = "cuda"
    autoencoder = ConvAutoencoder().to(device).eval()
    img = torch.ones((10, 1, 192, 320)).to(device)  # Adjust input size as needed
    x = autoencoder(img)
    summary(autoencoder, input_size=(1, 192, 320))

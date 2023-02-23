import torch
from .spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)

        #self.batchnorm1 = torch.nn.BatchNorm2d(256)
        #self.batchnorm2 = torch.nn.BatchNorm2d(512)
        #self.batchnorm3 = torch.nn.BatchNorm2d(1024)
        #self.spectralnorm0 = SpectralNorm(self.conv1)
        self.spectralnorm1 = SpectralNorm(self.conv2)
        self.spectralnorm2 = SpectralNorm(self.conv3)
        self.spectralnorm3 = SpectralNorm(self.conv4)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        
        #x = self.leaky_relu(self.conv1(x))
        #x = self.leaky_relu(self.batchnorm1(self.conv2(x)))
        #x = self.leaky_relu(self.batchnorm2(self.conv3(x)))
        #x = self.leaky_relu(self.batchnorm3(self.conv4(x)))
        #x = self.conv5(x)
        
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.spectralnorm1(x))
        x = self.leaky_relu(self.spectralnorm2(x))
        x = self.leaky_relu(self.spectralnorm3(x))
        x = self.conv5(x)
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        self.deconv1 = torch.nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1, padding=0)
        self.deconv2 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv5 = torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

        self.batchnorm1 = torch.nn.BatchNorm2d(1024)
        self.batchnorm2 = torch.nn.BatchNorm2d(512)
        self.batchnorm3 = torch.nn.BatchNorm2d(256)
        self.batchnorm4 = torch.nn.BatchNorm2d(128)
        
        self.relu = torch.nn.ReLU(True)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):

        x = self.relu(self.batchnorm1(self.deconv1(x)))
        x = self.relu(self.batchnorm2(self.deconv2(x)))
        x = self.relu(self.batchnorm3(self.deconv3(x)))
        x = self.relu(self.batchnorm4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))

        return x
    
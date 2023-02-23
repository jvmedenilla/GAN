import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchvision.transforms as T

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    (1) In the discriminator step, you should zero gradients in the discriminator 
    and sample noise to generate a fake data batch using the generator. Calculate 
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.
    
    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
    
    You will need to reshape the fake image tensor outputted by the generator to 
    be dimensions (batch_size x input_channels x img_size x img_size).
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.
    
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    transforms = torch.nn.Sequential(T.ColorJitter(brightness=.5, hue=.3), T.RandomCrop(14))
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            #transforms(real_images)
            
            #print(real_images.shape, transformed_images.shape)
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None

            #  Train Discriminator
            
            D_solver.zero_grad()
            # forward pass real images through D (need to flatten the matrix first)
            real_logits = D(real_images).view(-1)

            # Sample noise as generator input
            noise = sample_noise(batch_size, 28).cuda()
            noise = torch.reshape(noise, (batch_size, 28, 1,1)).cuda()
            #print(noise.shape)
            # Generate a batch of images
            fake_images = G(noise).cuda()
            #fake_images = torch.reshape(fake_images,(batch_size, input_channels, img_size, img_size))
            # forward pass generated images through D (need to flatten the matrix first)
            fake_logits = D(fake_images.detach()).view(-1)
            # calculate discriminator loss
            d_error = discriminator_loss(real_logits, fake_logits) 
            # calulcate gradients for D in backward pass
            d_error.backward()
            # update D
            D_solver.step()
            
            # train generator:
            
            # zero generator gradients 
            G_solver.zero_grad()
            # generate new set of fake image (cant reuse the one used in D step)
            noiseG = sample_noise(batch_size, noise_size).cuda()
            #noiseG = torch.reshape(noiseG, (batch_size, noise_size, 1,1)).cuda()
            fake_images = G(noiseG).cuda()
            fake_logits = D(fake_images).view(-1)
            # calculate generator loss
            g_error = generator_loss(fake_logits)
            # calulcate gradients for G in backward pass
            g_error.backward()
            # update G
            G_solver.step()       


            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1
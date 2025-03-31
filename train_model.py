import os
import argparse
import torch
import torch.nn as n
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import Dataset, DataLoader
from dataset import get_training_data_RF 
from torch import einsum
import torch.nn.functional as F
import pickle
from models import Generator, Discriminator

parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--train_dir', type=str, required=True, help='Path to the training directory')

parser.add_argument('--path_train_data_fname', type=str, required=True, help='Filename of the training data')

parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')

args = parser.parse_args()

train_dir = args.train_dir

path_train_data_fname = args.path_train_data_fname

model_dir = args.model_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19(pretrained=True).to(device)
          
class Losses():
    def __init__(self):
        super().__init__()
        self.disc_losss = n.BCEWithLogitsLoss()
        self.gen_losss = n.BCEWithLogitsLoss()
        self.vgg_loss = n.MSELoss()
        self.mse_loss = n.L1Loss()
        self.lamda = 0.5
        self.eeta = 100
        self.wtPerceptual = 0.05
        
    def calculateLoss(self,discriminator, generator,LR_image, HR_image,i):
        
        d_loss = 0
        
        disc_optimizer.zero_grad()
        
        generated_output = generator(LR_image.to(device).float())
        
        fake_data = generated_output.clone()
        
        fake_label = discriminator(fake_data)
  
        HR_image_tensor = HR_image.to(device).float()
        
        real_data = HR_image_tensor.clone()
        
        real_label = discriminator(real_data)
        
        labels_real = (torch.ones(real_label.shape[0],1) - 0.2*torch.rand(real_label.shape[0], 1)).cuda()
        
        labels_fake = (0.2*torch.rand(fake_label.shape[0],1)).cuda()
        
        relativistic_d1_loss = self.disc_losss((real_label - torch.mean(fake_label)), labels_real)
        
        relativistic_d2_loss = self.disc_losss((fake_label - torch.mean(real_label)), labels_fake) 
             
        if i%3 == 0:
            d_loss = (relativistic_d1_loss + relativistic_d2_loss) / 2

            d_loss.backward(retain_graph = True)
            
            disc_optimizer.step()

        fake_label_ = discriminator(generated_output)
        
        real_label_ = discriminator(real_data)
                        
        gen_optimizer.zero_grad()
        
        labels_real = (torch.ones(fake_label_.shape[0],1) - 0.2*torch.rand(target.shape[0], 1)).cuda()      
        
        labels_fake = (0.2*torch.rand(real_label_.shape[0],1)).cuda()
        
        g_real_loss = self.gen_losss((fake_label_ - torch.mean(real_label_)), labels_real)
        
        g_fake_loss = self.gen_losss((real_label_ - torch.mean(fake_label_)), labels_fake)
        
        g_loss = (g_real_loss + g_fake_loss) / 2
                
        input2vgg_gen = generated_output.repeat(1, 3, 1, 1)
        
        input2vgg_gen = (input2vgg_gen-torch.mean(input2vgg_gen))/torch.std(input2vgg_gen)
        
        input2vgg_real = real_data.repeat(1, 3, 1, 1)
 
        input2vgg_real = (input2vgg_real-torch.mean(input2vgg_real))/torch.std(input2vgg_real)
        
        v_loss = self.vgg_loss(vgg.features[:15](input2vgg_gen),vgg.features[:15](input2vgg_real))
        
        m_loss = self.mse_loss(generated_output,real_data)
        
        generator_loss = self.lamda * g_loss + self.wtPerceptual * v_loss + self.eeta * m_loss
 
        generator_loss.backward()
        
        gen_optimizer.step()

        return d_loss, generator_loss, g_loss, v_loss, m_loss

with open(path_train_data_fname, 'rb') as z:
    
    train_data_fname = pickle.load(z)
    

train_dataset = get_training_data_RF(train_dir,train_data_fname)

BATCH_SIZE = 32

train_loader = DataLoader(dataset=train_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True
                      )

gen = Generator()

gen= n.DataParallel(gen)

gen.to(device)

disc = Discriminator()

disc= n.DataParallel(disc)

disc.to(device) 

gen_optimizer = optim.Adam(gen.parameters(),lr=0.0002,betas=(0.9, 0.999))

disc_optimizer = optim.Adam(disc.parameters(),lr=0.0002,betas=(0.9, 0.999))

for epoch in range (50):
       
    dloss_list=[]   
    gloss_list=[]
    
    for i, data in enumerate(train_loader, 0):


        target = data[0].cuda()
        input_ = data[1].cuda()
        

        disc_loss, gen_loss, g_loss, v_loss, m_loss = Losses().calculateLoss(disc, gen, input_, target,i)

        torch.cuda.empty_cache()
        
        print(">%d, d[%.3f]  g_loss[%.3f] vloss[%.3f] mloss[%.3f]  g[%.3f]"
            % (epoch, disc_loss, 0.5*g_loss,0.05*v_loss,100*m_loss, gen_loss))
    
torch.save({'epoch': epoch, 
            'state_dict': gen.state_dict(),
            'optimizer' : gen_optimizer.state_dict()
            }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))
        



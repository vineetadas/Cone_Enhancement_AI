import os
import torch
import numpy as np
import scipy.io
from collections import OrderedDict
from models import Generator

PATH_TO_TEST_DATA = "./data/test_images/input"

PATH_TO_RESULTS = "./data/test_images/output"

PATH_TO_TRAINED_MODEL = "./data/trained_model/model_epoch_49.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
model_g = Generator()

model_g.to(device)

checkpoint = torch.load(PATH_TO_TRAINED_MODEL)

try:
    model_g.load_state_dict(checkpoint["state_dict"])
except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model_g.load_state_dict(new_state_dict)
        
        
model_g.eval()

for filename in os.listdir(PATH_TO_TEST_DATA):
    
    print(os.path.join(PATH_TO_TEST_DATA,filename))
    
    mat_file = scipy.io.loadmat(os.path.join(PATH_TO_TEST_DATA,filename))
    
    input_vol  = mat_file['arr']
    
    cropped_vol1 = np.transpose(input_vol,[2,0,1])

    with torch.no_grad(): 
        cropped_vol1 = np.float32(np.expand_dims(cropped_vol1, 1))
        input_img = (cropped_vol1 - 127.5) / 127.5
        output_vol =[]
        
        for i in range(len(input_img)):
            
            test_img = np.expand_dims(input_img[i],0)

            test_img = torch.from_numpy(test_img)
            test_img = test_img.cuda()
            sr = model_g(test_img) 

            sr1 = (sr.cpu().detach().numpy().squeeze()+1)/2    
            sr2 = sr1*255
            sr2 = sr2.astype(np.uint8)
            output_vol.append(sr2)
        
        output_vol = np.array(output_vol)
        output_vol =  np.transpose(output_vol,[1,2,0])
        scipy.io.savemat(os.path.join(PATH_TO_RESULTS, filename), mdict={'arr': output_vol})

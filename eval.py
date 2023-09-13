import numpy as np
import torch
import random
import os
from opts.attack_opts import parse_opts
from spatial_transforms import spatial_Compose, Normalize, ToTensor
from model import generate_model_c3d
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        

if __name__ == '__main__':
    torch.set_printoptions(precision=6)
    opt = parse_opts()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.arch = '{}-{}'.format(opt.model_type, opt.model_depth)
    model = generate_model_c3d(opt)
    model.eval()
    if opt.model == 'c3d':
        norm_method = Normalize([101.2198 / opt.norm_value, 97.5751 / opt.norm_value, 89.530 / opt.norm_value] , [1, 1, 1])
        spatial_transform = spatial_Compose(
            [ToTensor(opt.norm_value), norm_method]) 
     
    video=[]
    path=opt.test_path
    for t in range(len(os.listdir(path))):
            frame=Image.open(os.path.join(path,str(t)+'.png'))
            frame=spatial_transform(frame)
            video.append(frame.unsqueeze(0))
    video=torch.cat(video,dim=0).permute(1,0,2,3).unsqueeze(0)
    video=video.cuda()
    type=path.split('/')[-2]
    label=int(path.split('/')[-1].split('to')[0])
    target=int(path.split('/')[-1].split('to')[-1].split('_')[0])
    if type== 'target' and  model(video).max(1)[1].item() == target:
        print('targeted attacks success')
    if type== 'untarget' and  model(video).max(1)[1].item() != label:
        print('untargeted attacks success')
         
    

import numpy as np
import torch
import random
import os
from tqdm import tqdm, trange
from opts.attack_opts import parse_opts
from spatial_transforms import spatial_Compose, Normalize, ToTensor
from model import generate_model_c3d
from attack_method import STDE
import torch.utils.data as data
from PIL import Image
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Data(data.Dataset):
    def __init__(self,video_path,spatial_transform):
        source_path=os.path.join(video_path,'clean')
        target_path=os.path.join(video_path,'target')
        with open(video_path+'/classInd.json', 'r', encoding="utf-8") as f:
            self.class_int=json.load(f)
        self.data=[os.path.join(source_path,p) for p in self.class_int['source_video']]
        self.target_data=[os.path.join(target_path,p) for p in self.class_int['target_video']]
       
        self.classtoindex={self.class_int['labels'][i]:i for i in range(len(self.class_int['labels']))}
        self.labels=[self.classtoindex[p] for p in self.class_int['source_video']]
        self.targets=[self.classtoindex[p] for p in self.class_int['target_video']]
        self.spatial_transform=spatial_transform
    def __getitem__(self, index):
        data=self.data[index]
        target_data=self.target_data[index]
        data=self.load_video(data)
        target_data=self.load_video(target_data)
        labels=self.labels[index]
        targets=self.targets[index]
        labels=torch.tensor(labels)
        targets=torch.tensor(targets)
        return data,labels,target_data,targets

    def load_video(self,path):
        video=[]
        for t in range(len(os.listdir(path))):
            frame=Image.open(os.path.join(path,str(t)+'.png'))
            frame=self.spatial_transform(frame)
            video.append(frame.unsqueeze(0))
        video=torch.cat(video,dim=0).permute(1,0,2,3)
        return video
    def __len__(self):
        return len(self.data)
        



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
    adversary = STDE(model=model, model_name=opt.model, pop_size=opt.pop_size, init_rate=opt.init_rate,
                     mutation_rate=opt.mutation_rate, steps=opt.steps, targeted=opt.targeted, device='cuda', seed=opt.seed,cf_rate=opt.cf_rate,time_mua=opt.time_mua,result_path=opt.result_path)
    correct = 0
    correct1=0
    test_data=Data(opt.video_path,spatial_transform)
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, num_workers=0)
    with tqdm(total=len(test_data), unit='img') as pbar:
        for batch_idx, (inputs, labels, target_video, target_label) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            targets = target_label.to(device)
            start_imgs = target_video.to(device)
            with torch.no_grad(): 
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                outputs = model(start_imgs)
                _, predicted = outputs.max(1)
                correct1 += predicted.eq(targets).sum().item()
                pbar.set_postfix(**{'clean acc': correct / len(test_data),'target acc': correct1 / len(test_data)})
                pbar.update(inputs.shape[0])
    correct = 0
    with tqdm(total=len(test_data), unit='img') as pbar:
            for batch_idx, (inputs, labels, target_video, target_label) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                targets = target_label.to(device)
                start_imgs = target_video.to(device)
                with torch.no_grad():
                    x_adv = adversary.perturb(inputs, labels, start_imgs, targets)
                    outputs = model(x_adv)
                    _, predicted = outputs.max(1)
                    if opt.targeted:
                        correct += predicted.eq(targets).sum().item()
                    else:
                        correct += (~predicted.eq(labels)).sum().item()
                    pbar.set_postfix(**{'FR': correct / len(test_data)})
                    pbar.update(inputs.shape[0])
    

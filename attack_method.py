import cv2
import torch
import torchvision.transforms as transforms
import random
import os
from tqdm import tqdm
from spatial_transforms import Normalize
import numpy as np
from spatial_transforms import spatial_Compose, Normalize, ToTensor


class STDE():
    def __init__(self, model, model_name, pop_size=15, init_rate=0.4, mutation_rate=1, steps=10000, targeted=False,
                 device='cuda', seed=0,cf_rate=0.7, time_mua=2,weight=1.0,result_path=None):
        self.model = model
        self.model_name = model_name
        self.pop_size = pop_size
        self.init_rate = init_rate
        self.mutation_rate = mutation_rate
        self.steps = steps
        self.now_queries = 0
        self.targeted = targeted
        self.device = device
        self.result_dir = result_path
        self.seed = seed
        self.cf_rate = cf_rate
        self.time_mua = time_mua
        self.weight=weight
        if targeted:
            self.result_dir = os.path.join(self.result_dir,'target')
        else:
            self.result_dir = os.path.join(self.result_dir, 'untarget')
        self.path=self.result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
       

    def initialise_population(self, inputs, source_labels, starting_imgs, target_labels):
        print('Initialise Population...')
        bs, c, t, h, w = inputs.shape
        self.t=t
        self.now_queries = 0
        L = [0 for i in range(10)]
        for i in range(int(self.cf_rate * 10)):
            L[i] = 1
        V = torch.zeros(bs, self.pop_size, t, 4).long().to(self.device)
        G = torch.ones(bs, self.pop_size).to(self.device) * torch.Tensor([float('Inf')]).to(self.device)
        C = torch.ones(bs, self.pop_size).to(self.device)
        F = torch.zeros(bs, self.pop_size,self.t).long().to(self.device)
        d=0
        for i in range(self.pop_size):
            h_margin = int(h * self.init_rate)
            w_margin = int(w * self.init_rate)
            cnt = 0
            while True:
                v_tmp = torch.zeros(bs, t, 4).long().to(self.device)
                v_tmp[:, :, 0] = torch.randint(0, h_margin, size=(bs, t)).to(self.device)  # x1
                v_tmp[:, :, 1] = torch.randint(0, w_margin, size=(bs, t)).to(self.device)  # y1
                v_tmp[:, :, 2] = torch.randint(h - h_margin + 1, h + 1, size=(bs, t)).to(self.device)  # x2
                v_tmp[:, :, 3] = torch.randint(w - w_margin + 1, w + 1, size=(bs, t)).to(self.device)  # y2
                f_tmp = np.random.choice(L, size=(bs, t))
                f_tmp = torch.from_numpy(f_tmp).int().long().to(self.device)
                v = (v_tmp * f_tmp[:, :, None]).long()
                v_mask = self.coord2mask(v)  # (bs,t,h,w)
                x_tmp = v_mask[:, None, :] * starting_imgs + (~v_mask[:, None, :]) * inputs
                flag = self.predict(x_tmp, source_labels, target_labels)
                fitness = self.get_fitness(x_tmp, inputs, v_mask)
                idx = ((G[:, i] > fitness) & flag & torch.isinf(G[:, i]))
                G[:, i][idx] = fitness[idx]
                V[:, i][idx] = v_tmp[idx].clone()
                F[:, i][idx] = f_tmp[idx].clone()
                if torch.isinf(G[:, i]).sum() == 0:
                    break
                cnt += 1
                if cnt > 10:
                    h_margin = w_margin = 1
                if cnt > 11:
                    L = [1 for i in range(10)]
        C = C * self.now_queries
        return V, G, C, F

    def coord2mask(self, v):
        b, t, _ = v.shape
        mask = torch.zeros((v.shape[0], v.shape[1], self.h, self.w),device=self.device).bool()
        mask = mask.view(v.shape[0] * v.shape[1], self.h, self.w)
        v = v.view(mask.shape[0], 4)
        for i in range(v.shape[0]):
            mask[i, v[i][0]:v[i][2], v[i][1]:v[i][3]] = True
        mask = mask.view(b, t, self.h, self.w)

        return mask
    
    def bound_handle(self, v_tmp):
        b, t, _ = v_tmp.shape
        v_tmp = v_tmp.view(b * t, 4)
        idx = (v_tmp[:, 0] > v_tmp[:, 2])
        v_tmp[:, 0][idx], v_tmp[:, 2][idx] = v_tmp[:, 2][idx], v_tmp[:, 0][idx]
        idx = (v_tmp[:, 1] > v_tmp[:, 3])
        v_tmp[:, 1][idx], v_tmp[:, 3][idx] = v_tmp[:, 3][idx], v_tmp[:, 1][idx]
        idx = (v_tmp[:, 3] >= self.w)
        v_tmp[:, 3][idx] = self.w - 1
        idx = (v_tmp[:, 2] >= self.h)
        v_tmp[:, 2][idx] = self.h - 1
        idx = (v_tmp[:, 1] < 0)
        v_tmp[:, 1][idx] = 0
        idx = (v_tmp[:, 0] < 0)
        v_tmp[:, 0][idx] = 0
        v_tmp = v_tmp.view(b, t, 4)
        return v_tmp

    def mutation(self, v_best, v_j, v_q, f_best, f_j, f_q):
        sub = (v_j - v_q) * self.mutation_rate
        sub = sub.long()
        v_tmp = v_best.clone()
        v_tmp = v_tmp + sub
        v_tmp = self.bound_handle(v_tmp)
        f = f_j | f_q
        f_tmp = f_best.clone()
        f_tmp = f_tmp & f
        return v_tmp.long(), f_tmp

    def crossover(self, v, f_tmp):
        mua_index = torch.randint(0, self.t, size=(f_tmp.shape[0], self.time_mua))
        mua = torch.randint(0, 2, size=(f_tmp.shape[0], self.time_mua))
        for t in range(f_tmp.shape[0]):
            for i in range(self.time_mua):
                f_tmp[t][mua_index[t][i]] = mua[t][i]
        f = f_tmp
        noise = torch.randint(-self.mutation_rate, self.mutation_rate + 1, (self.bs, self.t, 4)).to(self.device)
        noise = (noise * f[:, :, None]).int()
        v_tmp = v.clone() + noise
        v_tmp = self.bound_handle(v_tmp)
        return v_tmp.long(), f_tmp

    def get_fitness(self, inputs, x_adv, mask):
        m = mask.permute(0, 2, 3, 1)
        m = m.sum(dim=3).float()
        m = m.view(inputs.shape[0], -1)
        m[m == 1] = 0
        return torch.norm(inputs - x_adv, 0, dim=(1, 2, 3, 4)) - self.weight*m.sum(dim=1)

    def predict(self, inputs, source_labels, target_labels):
        self.now_queries += 1
        outputs = self.model(inputs)
        _, predicted = outputs.max(1)
        if self.targeted:
            flag = predicted.eq(target_labels)
        else:
            flag = (~predicted.eq(source_labels))
            
        return flag

    def perturb(self, inputs, source_labels, starting_imgs, target_labels):
        bs, c, t, h, w = inputs.shape
        self.h = h
        self.w = w
        self.bs = bs
        self.t = t
       
        V, G, C, F = self.initialise_population(inputs, source_labels, starting_imgs, target_labels)


        k_worst = G.argmax(dim=1)  # (bs,1)
        k_best = G.argmin(dim=1)  # (bs,1)

        bs_list = torch.arange(self.bs).long().to(self.device)
        with tqdm(total=self.steps, unit='iter',ncols=100) as pbar:
            for i in range(self.steps - self.now_queries):
                k = torch.randint(0, self.pop_size, size=(self.bs, 2)).to(self.device)  # (bs,2)
                cnt = random.randint(1, self.pop_size - 1)
                idx = (k[:, 0] == k[:, 1])
                k[:, 1][idx] = (k[:, 1][idx] + 1) % self.pop_size
                idx = (k[:, 0] == k_best)
                if k[:, 0][idx].shape[0] != 0:
                    k[:, 0][idx] = (k[:, 0][idx] + cnt) % self.pop_size
                assert (k[:, 0] == k_best).sum() == 0, 'j is not best'
                idx = (k[:, 1] == k_best)
                if k[:, 1][idx].shape[0] != 0:
                    k[:, 1][idx] = (k[:, 1][idx] + cnt) % self.pop_size
                assert (k[:, 1] == k_best).sum() == 0, 'q is not best'

                j, q = k[:, 0], k[:, 1]
                v_r, f_r = self.mutation(V[range(self.bs), k_best], V[range(self.bs), j], V[range(self.bs), q],
                                          F[range(self.bs), k_best], F[range(self.bs), j], F[range(self.bs), q])
                v_m, f_r = self.crossover(v_r, f_r)
                v = (v_m * f_r[:, :, None]).int()
                v_m_mask = self.coord2mask(v)
                x_adv = (~v_m_mask[:, None, :]) * inputs + v_m_mask[:, None, :] * starting_imgs
                flag = self.predict(x_adv, source_labels, target_labels)
                fitness = self.get_fitness(x_adv, inputs, v_m_mask)

                update_idx = (G[range(self.bs), k_worst] * 1000 > fitness * 1000) & flag

                if update_idx.sum() != 0:
                    G[bs_list[update_idx], k_worst[update_idx]] = fitness[update_idx]
                    V[bs_list[update_idx], k_worst[update_idx]] = v_m.clone()[update_idx]
                    C[bs_list[update_idx], k_worst[update_idx]] = self.now_queries
                    F[bs_list[update_idx], k_worst[update_idx]] = f_r.clone()[update_idx]

                k_worst = G.argmax(dim=1)
                k_best = G.argmin(dim=1)
                v_area = V[range(bs), k_best]
                ff = F[range(bs), k_best]
                v_mask = self.coord2mask((v_area * ff[:, :, None]).int())
                ttt_area = torch.true_divide(v_mask.sum(dim=(1,2,3)), self.w * self.h * t)
                pbar.set_postfix(**{'AOA': ttt_area.mean().item()})
                pbar.update(1)

        v_area = V[range(bs), k_best]
        ff = F[range(bs), k_best]
        v_mask = self.coord2mask((v_area * ff[:, :, None]).int())
        ttt_area = torch.true_divide(v_mask.sum(dim=(1,2,3)), self.w * self.h * t)
        V_ = (V[range(bs), k_best] * F[range(bs), k_best][:, :, None]).int()
        x_adv = (~self.coord2mask(V_))[:, None, :] * inputs + self.coord2mask(
            V_)[:, None, :] * starting_imgs
        print('Finish Attack!')
        self.record_video(x_adv,source_labels,target_labels,ttt_area)
        return x_adv


    def record_video(self, x_adv, source_label,target_labels,re):
        for bs in range(self.bs):
            if not os.path.exists(os.path.join(self.result_dir, str(source_label[bs].item())+'to'+str(target_labels[bs].item())+'_'+str(re[bs].item()))):
                os.makedirs(os.path.join(self.result_dir, str(source_label[bs].item())+'to'+str(target_labels[bs].item())+'_'+str(re[bs].item())))
            for t in range(self.t):
                imgs = self.tensor_to_pil(x_adv[bs, :, t, :].cpu())
                imgs.save(os.path.join(self.result_dir, str(source_label[bs].item())+'to'+str(target_labels[bs].item())+'_'+str(re[bs].item()), str(t) + '.png'))

    def tensor_to_pil(self, x_adv):
        x_adv = x_adv.cpu()
        mean = [-101.2198, -97.5751, -89.5303]
        std = [255, 255, 255]
        inv_transform = transforms.Compose([Normalize(mean, std), transforms.ToPILImage('RGB')])
        inputs = inv_transform(x_adv)
        return inputs


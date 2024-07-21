import random

import torch.utils.data
import torch.nn.init
import numpy as np

import matplotlib.pyplot as plt
from utils.utils import t_to_np

def plt_ori_vs_gen(ori_data, gen_data, run):
    for j, (o, g) in enumerate(zip(ori_data, gen_data)):
        f, ax = plt.subplots(1)
        plt.plot(o.squeeze(), label='ori')
        plt.plot(g.squeeze(), label='gen')
        plt.legend()
        if run:
            run['test/ori_gen'].log(f)
        else:
            plt.savefig(f'./figures/ori_vs_gen_{j}.png', dpi=300)
            plt.show()
        if j == 10:
            break

def plt_ori_vs_rec(ori_data, rec_data, run):
    for j, (o, r) in enumerate(zip(ori_data, rec_data)):
        f, ax = plt.subplots(1)
        plt.plot(o.squeeze(), label='ori')
        plt.plot(r.squeeze(), label='rec')
        plt.legend()
        if run:
            run['test/ori_rec'].log(f)
        else:
            plt.savefig(f'./figures/ori_vs_rec_{j}.png', dpi=300)
            plt.show()
        if j == 10:
            break

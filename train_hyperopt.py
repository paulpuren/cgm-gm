'''
Train function
'''

import argparse
import time
import neptune.new as neptune

import os, sys
import math

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dvae import *
from get_data import load_data, min_max_norm, get_phase_mag

from utils.utils_vis import plt_ori_vs_gen, plt_ori_vs_rec
from metrics.discrimanitive import discriminative_score_metrics
import logging
import torchaudio

# hyperopt dependencies
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def loss(X, X_hat, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar, beta, norm_dict, time_spec_converter, true_phase, time_domain_wfs, log_reg=True, alpha=1.):

    # X and X_hat are spectrogram in log space, with shape of [b,t,f]
    batch_size = X.shape[0]
    wfs_min, wfs_max = norm_dict['log_wfs']
    
    # amplitude reconstruction loss
    MSE_loss = nn.MSELoss(reduction='sum')
    recons_loss = MSE_loss(X_hat, X) # log norm of amplitude
    
    # waveform loss
    wfs_hat = min_max_norm(X_hat, wfs_min, wfs_max, '[0,1]', 'add')
    eps = 1e-10
    wfs_hat = (torch.pow(10, wfs_hat) - eps).permute(0, 2, 1)
    _, magnitude_hat = get_phase_mag(wfs_hat)
    time_domain_wfs_hat = time_spec_converter.spec_to_time((magnitude_hat * torch.exp(1j * true_phase.to(magnitude_hat.device))))
    min_len = min(time_domain_wfs_hat.squeeze().shape[-1], time_domain_wfs.squeeze().shape[-1])
    time_domain_loss = MSE_loss(time_domain_wfs_hat.squeeze()[..., :min_len], time_domain_wfs.to(magnitude_hat.device).squeeze()[..., :min_len])

    # if normal distribution + rnn
    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    recons_loss = recons_loss / batch_size
    kld_z = kld_z / batch_size
    time_domain_loss = time_domain_loss / batch_size

    total_loss = recons_loss + beta * kld_z + alpha * time_domain_loss

    return total_loss, recons_loss, time_domain_loss, kld_z


def train(args, model, train_loader, run, norm_dict, time_spec_converter):
    best_loss = np.inf
    train_loss_list = []
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    # beta_np_inc = frange_cycle_cosine(0.0, 1.0, args.epochs, 10)  # 4 is just the number of cycles

    print('========================')
    print('Training model...')
    print('========================\n')

    start = time.time()
    beta = args.beta
    log_reg = args.log_reg
    alpha = args.alpha
    
    for epoch in range(args.epochs):
        run['train/epoch'].log(epoch+1)
        model.train()
        train_loss, n = 0.0, 0
        rec_loss_avg, kl_div_avg, time_loss_avg = 0.0, 0.0, 0.0

        for batch_idx, (wfs, cond_var, true_phase, time_domain_wfs) in enumerate(train_loader):
            # [b,c,l]
            wfs = wfs.to(args.device).float().squeeze()
            # print(wfs.shape)

            cond_var = cond_var.to(args.device).float()
            batch_size = wfs.shape[0]

            # get the reconstruction and posterior
            wfs_hat, z_post_mean, z_post_logvar = model(wfs, cond_var)
            z_prior_mean, z_prior_logvar, _ = model.module.sample_prior(
                y=cond_var,
                n_sample=batch_size,
                seq_len=wfs.shape[1],
                random_sampling=True,
                device=args.device)

            l, rec_loss, time_domain_l, kl_div, pgv_reg = loss(wfs, wfs_hat, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar, beta, norm_dict, time_spec_converter, true_phase, time_domain_wfs, log_reg, alpha)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # record the loss
            train_loss += l.cpu().item()
            rec_loss_avg += rec_loss.cpu().item()
            kl_div_avg += kl_div.cpu().item()
            time_loss_avg += time_domain_l.cpu().item()
            n += batch_size

        scheduler.step()
        train_loss /= n
        rec_loss_avg /= n
        kl_div_avg /= n
        time_loss_avg /= n
        train_loss_list.append([train_loss, rec_loss_avg, kl_div_avg, time_loss_avg])
        if train_loss < best_loss:
            save_checkpoint(args.log_path, dict(model=model))
            best_loss = train_loss

        # print train loss
        print("Epoch: %s/%s, total train loss: %.10f, reconstruction loss: %.10f, time domain loss: %.10f, KL divergence: %.10f" 
                                                                                                        % (epoch + 1,
                                                                                                           args.epochs,
                                                                                                           train_loss,
                                                                                                           rec_loss_avg,
                                                                                                           time_loss_avg,
                                                                                                           kl_div_avg))
        if math.isnan(train_loss):
            return [[1e8]*5]


        run['train/total'].log(train_loss)
        run['train/rec'].log(rec_loss_avg)
        run['train/kl'].log(kl_div_avg)
        run['train/tdl'].log(time_loss_avg)

    end = time.time()
    print('The training time is: ', (end - start))
    print('check train loss list:', train_loss_list[0])

    return train_loss_list


def eval_metrics(args, model, test_loader, all_set, all_loader, run, time_spec_converter, norm_dict, SEQ_LEN):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model.eval()
    wfs_min, wfs_max = norm_dict['log_wfs']
    
    with torch.no_grad():
        real_wfs_list, pred_wfs_list = [], []

        real_wfs_list, pred_wfs_list = collect_real_and_gen(args, model, all_loader, pred_wfs_list, real_wfs_list, SEQ_LEN)
        true_phase_list = all_set[2]

        real_wfs_list = np.transpose(np.concatenate(tuple(real_wfs_list), axis=0), (0, 2, 1))
        pred_wfs_list = np.transpose(np.concatenate(tuple(pred_wfs_list), axis=0), (0, 2, 1))

        from metrics.visualization_metrics import visualization

        real_wfs_list = time_spec_converter.time_to_spec(torch.Tensor(real_wfs_list).squeeze().to(args.device)).permute(0, 2, 1).detach().cpu()
        _, real_wfs_list = get_phase_mag(real_wfs_list)
        pred_wfs_list = np.transpose(pred_wfs_list, (0, 2, 1))

        # get the reconstruction and posterior
        test_iter = iter(test_loader)
        wfs, cond_var, true_phase , _  = next(test_iter)

        cond_var = cond_var.to(args.device).float()
        wfs_hat, _, _ = model(wfs, cond_var)

        wfs = min_max_norm(wfs, wfs_min, wfs_max, '[0,1]', 'add')
        wfs_hat = min_max_norm(wfs_hat, wfs_min, wfs_max, '[0,1]', 'add')

        eps = 1e-10
        
        # transform amplitude to original waveforms
        wfs_hat = torch.pow(10, wfs_hat) - eps
        wfs = torch.pow(10, wfs) - eps
        
        conv_input_orig = wfs.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase.to(args.device))
        conv_input_hat  = wfs_hat.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase.to(args.device))

        wfs_orig = time_spec_converter.spec_to_time(conv_input_orig).unsqueeze(dim=-1).detach().cpu()
        wfs_hat = time_spec_converter.spec_to_time(conv_input_hat).unsqueeze(dim=-1).detach().cpu()

        pred_wfs_list = torch.Tensor(pred_wfs_list)
        pred_wfs_list = min_max_norm(pred_wfs_list, wfs_min, wfs_max, '[0,1]', 'add')
        pred_wfs_list = torch.pow(10, pred_wfs_list) - eps
        
        wfs_gen = time_spec_converter.spec_to_time(pred_wfs_list.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase_list.to(args.device))).unsqueeze(dim=-1).detach().cpu()
        wfs_all_orig = time_spec_converter.spec_to_time(real_wfs_list.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase_list.to(args.device))).unsqueeze(dim=-1).detach().cpu()
        
        visualization(real_wfs_list, pred_wfs_list, 'tsne', args, run)
        plt_ori_vs_rec(wfs_orig, wfs_hat, run)

        import matplotlib.pyplot as plt
        test_loader_iter = iter(test_loader)
        _, cond_var, true_phase, wfs = next(test_loader_iter)
        
        # [b,c,h,w]
        wfs = wfs.to(args.device).float()
        cond_var = cond_var.to(args.device).float()

        pred_wfs = model.module.generate(cond_var, SEQ_LEN)  # [679,3,6000]
        pred_wfs = min_max_norm(pred_wfs, wfs_min, wfs_max, '[0,1]', 'add')

        # transform amplitude to original signal
        pred_wfs = torch.pow(10, pred_wfs) - eps
        pred_wfs = time_spec_converter.griffinlim(pred_wfs.permute(0, 2, 1)).detach().cpu()
        # pred_wfs = time_spec_converter.spec_to_time(pred_wfs.permute(0, 2, 1)*torch.exp(1j * true_phase.to(args.device))).unsqueeze(dim=-1).detach().cpu()

        # scale wfs back to original magnitude
        wfs, pred_wfs = wfs.cpu().numpy(), pred_wfs.cpu().numpy()
        for i in range(wfs.shape[0]):
            f, ax = plt.subplots(1)
            plt.plot(wfs[i].squeeze(), label='original')
            plt.plot(pred_wfs[i].squeeze(), label='generated')
            plt.legend()
            run['test/gen_con_var'].log(f)
            # plt.savefig('./wfs{}.pdf'.format(i), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(f)

            if i == 20:
                break


    discriminative_score = list()
    for _ in range(10):
        disc_score = discriminative_score_metrics(wfs_all_orig, wfs_gen, args)
        discriminative_score.append(disc_score)
        print(disc_score)

    print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)) +
          ', std: ' + str(np.round(np.std(discriminative_score), 4)))

    return np.round(np.mean(discriminative_score), 4), np.round(np.std(discriminative_score), 4)


def collect_real_and_gen(args, model, loader, pred_wfs_list, real_wfs_list, SEQ_LEN):
    for batch_idx, (_, cond_var, _, wfs) in enumerate(loader):

        # [b,c,h,w]
        wfs = wfs.to(args.device).float()
        cond_var = cond_var.to(args.device).float()

        pred_wfs = model.module.generate(cond_var, SEQ_LEN)  # [679,3,6000]

        # scale wfs back to original magnitude
        wfs, pred_wfs = wfs.cpu().numpy(), pred_wfs.cpu().numpy()

        real_wfs_list.append(wfs)
        pred_wfs_list.append(pred_wfs)

    return real_wfs_list, pred_wfs_list


def save_checkpoint(ckpt_dir, state):
    import torch
    saved_state = {
    'model': state['model'].state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def restore_checkpoint(ckpt_dir, state, device='cuda:0'):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        return state


class TimeSpecConverter:
    def __init__(self, n_fft, w_len, h_len, power, device, n_iter=50):
        self.n_fft = n_fft
        self.w_len = w_len
        self.h_len = h_len
        self.power = power
        self.n_iter = n_iter
        self.device = device
        self.griffinlim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, n_iter=1000, win_length=self.w_len, 
                                                           hop_length=self.h_len, power=self.power).to(self.device)
        
    def time_to_spec(self, wfs):
        return torch.stft(wfs, n_fft=self.n_fft, hop_length=self.h_len, win_length=self.w_len, return_complex=True)
    
    def spec_to_time(self, wfs):
        return torch.istft(wfs, n_fft=self.n_fft, hop_length=self.h_len, win_length=self.w_len)
    
    def griffinlim(self, wfs):
        return self.griffinlim(wfs)
        

def main(args, mc):

    log_dir  = args.log_dir

    if mc:
        args.lr = mc['lr']
        args.batch_size = mc['batch_size']
        args.weight_decay = mc['weight_decay']

        args.z_dim = mc['z_dim']
        args.z_rnn_dim = mc['z_rnn_dim']

        if args.z_dim > args.z_rnn_dim:
            args.z_rnn_dim = args.z_dim

        args.beta = mc['beta']
        args.log_reg = mc['log_reg']
        args.alpha = mc['alpha']
        args.ncond = mc['ncond']

    name = 'GM_V2_VAE_data5_dist-{}_bs={}-rnn_size={}-z_dim={}-lr={}' \
           '-weight:kl={}-log_reg={}-w_decay={}-w_len={}-h_len={}-ncond={}-tcondvar={}-seed={}'.format(
        args.epochs, args.batch_size, args.z_rnn_dim, args.z_dim, args.lr,
        args.beta, str(args.log_reg), args.weight_decay, args.w_len, args.h_len, args.ncond, args.tcondvar, args.seed)

    os.makedirs(log_dir, exist_ok=True) 
    log_path = os.path.join(log_dir, name)  
    args.log_path = log_path

    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    time_spec_converter = TimeSpecConverter(n_fft=args.fft_size, w_len=args.w_len, h_len=args.h_len, power=args.power, device=args.device)
    
    # Get datasave_checkpoint
    print('========================')
    print('Loading data...')
    print('========================\n')
    
    train_set, test_set, all_set, train_loader, test_loader, all_loader, norm_dict, time_serie_len = load_data(args.path, time_spec_converter=time_spec_converter, train_bs=args.batch_size, tcondvar=args.tcondvar)
    SEQ_LEN = time_serie_len//args.h_len + 1

    # set-up neptune
    run = neptune.init_run(
        project=YOUR_PROJECT_NAME, # your project name
        api_token=YOUR_TOKEN, # use your own token
        tags=[args.tag],
        mode=args.neptune
    )  # your credentials
    run['config/hyperparameters'] = vars(args)

    # setup the model
    model = cVAE(in_dim=args.fft_size, z_dim=args.z_dim, ncond=args.ncond, z_rnn_dim=args.z_rnn_dim, in_size=len(norm_dict)-1).to(args.device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    state = dict(model=model)

    # train the model
    train_loss_list = train(args, model, train_loader, run, norm_dict, time_spec_converter)
    
    if train_loss_list[-1][0] < 1e8:
        disc_mean, disc_std = eval_metrics(args, model, test_loader, all_set, all_loader, run, time_spec_converter, norm_dict, SEQ_LEN)
        print('Discriminative score: {:.4f} +- {:.4f}'.format(disc_mean, disc_std))
        run['test/disc_mean'].log(disc_mean)
        run['test/disc_std'].log(disc_std)

        # if disc_mean < 0.25:
        save_checkpoint(log_path, state)
    else:
        disc_mean = 1e8

    run.stop()
    
    return {'loss': disc_mean,
            'args': args,
            'status': STATUS_OK}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--path', type=str, default='/scratch/gm/data/',
                        help='data directory')  
    parser.add_argument('--log_dir', type=str, default='/scratch/gm/logs',
                        help='model saving directory')  
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--z_dim', type=int, default=16, help='number of channels of latent space z')
    parser.add_argument('--z_rnn_dim', type=int, default=128, help='number of channels of latent space z using rnn')
    parser.add_argument('--ncond', type=int, default=16, help='number of channels of conditional variables')
    parser.add_argument('--beta', type=float, default=1.0, help='penalty coefficient for reconstruction loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='penalty coefficient for time domain loss') 
    parser.add_argument('--neptune', default='async')
    parser.add_argument('--tag', default='gm vae reg hlen')
    parser.add_argument('--log_reg', action='store_true')
    parser.add_argument('--tcondvar', default=2, type=int, help='loaded conditional variables: 0 - mag&dist&angle; 1 - mag&dist&angle&depth; 2 - mag&src-sta-coords&depth.')

    # audio args
    parser.add_argument('--w_len', type=int, default=160, help='window length')
    parser.add_argument('--h_len', type=int, default=46, help='hop length') 
    parser.add_argument('--power', type=int, default=1, help='power of the spectrogram')
    parser.add_argument('--fft_size', type=int, default=160, help='fft size')

    args = parser.parse_args()

    def get_experiment_space():
        space = {  # Architecture parameters
            'model': 'vae',
            'lr': hp.choice('lr', [8e-4, 7e-4, 6e-4]),
            'z_rnn_dim': hp.choice('z_rnn_dim', [16, 32]),
            'z_dim': hp.choice('z_dim', [8, 16, 32]),
            'beta': hp.choice('beta', [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2]),
            'weight_decay': hp.choice('weight_decay', [5e-6, 1e-5, 1e-6]),
            'alpha': hp.choice('alpha', [0.01, 0.05, 0.1]),
            'ncond': hp.choice('ncond', [16, 32]),
            'log_reg': True, # hp.choice('log_reg', [True, False]),

            # Data parameters
            'batch_size': hp.choice('batch_size', [128, 256])}

        return space

    rstate = np.random.default_rng(args.seed)
    print(f'rstate: {rstate}')
    trials = Trials()
    fmin_objective = functools.partial(main, args)
    space = get_experiment_space()
    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials, verbose=True, rstate=rstate)
    
    # opt_params = {  # Architecture parameters
    #        'model': 'vae',
    #        'lr': 7e-4,
    #        'z_rnn_dim': 32,
    #        'z_dim': 16,
    #        'beta': 0.05,
    #        'weight_decay': 5e-6,

    #        # Data parameters
    #        'batch_size': 256}

    opt_params = None
    main(args, opt_params)
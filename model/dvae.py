'''
Dynamic VAE module with GRUs
'''

import torch
import torch.nn as nn
from collections import OrderedDict


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                # q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)


class label_embedding(nn.Module):
    '''
    Label embedding for conditional variables
    '''
    
    def __init__(self, in_size=3, hid_len=100, ncond=128):
        super(label_embedding, self).__init__()

        # self.hid_len = hid_len
        self.mlp = MLP([in_size, 32, 32, ncond], last_activation=True) # [b,128]

    def forward(self, x, h_len):
        '''
        Args:
        -----
        x: conditional variables, shape: [b,c], [b,3]
        '''
        
        x = self.mlp(x) # [b,128]
        x = x.view(x.shape[0], 1, x.shape[1]).repeat(1, h_len, 1) # [b,1,128] -> [b,100,128]
        
        return x


class BatchLinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(BatchLinearUnit, self).__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.nrm = nn.BatchNorm1d(out_features)
        self.non_lin = nonlinearity

    def forward(self, x):
        x_lin = self.lin(x)
        x_nrm = self.nrm(x_lin.permute(0, 2, 1))
        return self.non_lin(x_nrm.permute(0, 2, 1))


class Encoder(nn.Module):
    def __init__(self, fft_size, nhid=16, ncond=0):
        super(Encoder, self).__init__()

        # tensor should be [b,l,c]
        self.rnn = nn.GRU(input_size=(fft_size // 2 + 1 + ncond), hidden_size=128+ncond, bidirectional=False,
                          num_layers=1, batch_first=True)
        self.b_norm = nn.BatchNorm1d((128+ncond))

        self.calc_mean = MLP([(128+ncond), 64, nhid], last_activation = False)
        self.calc_logvar = MLP([(128+ncond), 64, nhid], last_activation = False)

    def forward(self, x, y):

        # concatenate conditional variables
        # y = y.permute(0,2,1) # [b,l,c] -> [b,c,l]
        x = torch.cat((x, y), dim=-1) # [b,128+128,100]

        # rnn
        z, _ = self.rnn(x)
        z = self.b_norm(torch.permute(z, (0, 2, 1)))
        z = torch.permute(z, (0, 2, 1))  # permute back to b x s x c

        # get mean and std
        mean = self.calc_mean(z) # [b,100,128+128]
        std = self.calc_logvar(z)

        return mean, std


class Decoder(nn.Module):
    def __init__(self, fft_size, nhid=16, ncond=16):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(MLP([nhid+ncond, 128, 128, (fft_size // 2 + 1)], last_activation=False)) # [b,100,128]

    def forward(self, z, y):
        out = torch.cat((z, y), dim=-1)  # [b,l,c], [b,100,128+nhid]
        out = self.fc(out)

        return out


class cVAE(nn.Module):
    def __init__(self, in_dim, z_dim = 16, ncond = 16, z_rnn_dim=32, in_size=3):
        super(cVAE, self).__init__()

        self.z_dim = z_dim
        self.ncond = ncond
        self.encoder = Encoder(in_dim, z_dim, ncond = ncond)
        self.decoder = Decoder(in_dim, z_dim, ncond = ncond)

        self.hidden_dim = z_rnn_dim
        self.z_prior_gru = nn.GRUCell(self.z_dim + ncond, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # define physical embedding
        self.label_embedding = label_embedding(in_size=in_size, hid_len=100, ncond=ncond)

    def reparameterize(self, mean, logvar, random_sampling=True):
        eps = torch.randn(mean.shape).to(mean.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def sample_prior(self, y, n_sample, seq_len=100, random_sampling=True, device='cpu'):

        batch_size = n_sample

        # z_out = None  # This will ultimately store all z_s in the format [batch_size, seq_len, z_dim]
        z_out = torch.zeros(batch_size, seq_len, self.z_dim).to(device) # [b,l,c]
        z_means = torch.zeros(batch_size, seq_len, self.z_dim).to(device)
        z_logvars = torch.zeros(batch_size, seq_len, self.z_dim).to(device)

        # initialize arbitrary input (zeros) and hidden states.
        z_t = torch.randn(batch_size, self.z_dim).to(device)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).to(device)

        # concatenate conditional variables
        y = self.label_embedding(y, seq_len) # [b, l, c]

        for i in range(seq_len):

            # concatenate conditional variables
            z_t = torch.cat((z_t, y[:,i]) , dim=-1) # [b, ncond + z_rnn_dim]

            h_t_ly1 = self.z_prior_gru(z_t, h_t_ly1)

            z_mean_t = self.z_prior_mean(h_t_ly1)
            z_logvar_t = self.z_prior_logvar(h_t_ly1)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)

            z_out[:, i] = z_t
            z_means[:, i] = z_mean_t
            z_logvars[:, i] = z_logvar_t

        return z_means, z_logvars, z_out


    def forward(self, x, y):
        y = self.label_embedding(y, x.shape[1]) # [b, l, c]

        # [b,l,c], c=16 
        mean, logvar = self.encoder(x, y)
        z = self.reparameterize(mean, logvar)

        return self.decoder(z, y), mean, logvar
    
    def generate(self, phy_labels, seq_len):
        
        # phy_labels: conditional variables
        batch_size = phy_labels.shape[0]

        # do sampling and rnn
        _, _, z = self.sample_prior(
            y = phy_labels,
            n_sample = batch_size, 
            seq_len = seq_len,
            random_sampling = True,
            device = phy_labels.device)
        
        y = self.label_embedding(phy_labels, seq_len)

        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res
    


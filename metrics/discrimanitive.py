import torch
import numpy as np
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score
from utils.utils import train_test_divide, batch_generator, extract_time


def discriminative_score_metrics(ori_data, generated_data, args):
    # Basic Parameters

    ## Builde a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = 16
    iterations = 2000
    batch_size = 128

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

    class Discriminator(nn.Module):
        def __init__(self, fft_size, hidden_dim):
            super(Discriminator, self).__init__()

            # the input dim: [batch,channel,length]
            # self.enc_net = nn.Sequential(
            #     BatchLinearUnit(fft_size // 2 + 1, fft_size // 2 + 1, nonlinearity=nn.Tanh()))

            # tensor should be [b,l,c]
            self.rnn = nn.GRU(input_size=(fft_size // 2 + 1), hidden_size=hidden_dim, bidirectional=False,
                               num_layers=1, batch_first=True)

            self.linear = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, last_hidden_state = self.rnn(x)
            y_hat_logit = self.linear(last_hidden_state)
            y_hat = nn.functional.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat


    model = Discriminator(args.fft_size, hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)

    if vars(args).get("h_len_ratio") is not None:
        hop_length = int(args.w_len*args.h_len_ratio)
    elif vars(args).get("h_len") is not None:
        hop_length = args.h_len
    else:
        raise NotImplementedError
    spectrogram = torchaudio.transforms.Spectrogram(win_length=args.w_len, hop_length=hop_length,
                                                    power=args.power, n_fft=args.fft_size).to(args.device)

    train_loss = 0.0

    model.train()
    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb = torch.stack(batch_generator(train_x, batch_size)).to(args.device)
        X_hat_mb = torch.stack(batch_generator(train_x_hat, batch_size)).to(args.device)

        X_mb = log_norm_spectogram(X_mb, spectrogram)
        X_hat_mb = log_norm_spectogram(X_hat_mb, spectrogram)

        y_logit_real, y_pred_real = model(X_mb.float())
        y_logit_fake, y_pred_fake = model(X_hat_mb.float())

        real_labels = torch.ones_like(y_logit_real)
        fake_labels = torch.zeros_like(y_logit_fake)

        d_loss_real = nn.functional.binary_cross_entropy_with_logits(y_logit_real, real_labels).mean()
        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(y_logit_fake, fake_labels).mean()

        d_loss = d_loss_real + d_loss_fake

        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()

        train_loss += d_loss.cpu().item()

        if itt % 100 == 0:
            # record the loss
            print('{}: train loss: {:.4f}'.format(itt, d_loss.cpu().item()))


    print('final train loss: {:.4f}'.format(train_loss / iterations))

    model.eval()
    with (torch.no_grad()):

        test_x = torch.stack(test_x).to(args.device)
        test_x_hat = torch.stack(test_x_hat).to(args.device)
        test_x, test_x_hat = log_norm_spectogram(test_x, spectrogram), log_norm_spectogram(test_x_hat, spectrogram)
        _, y_pred_real_curr = model(test_x.float())
        _, y_pred_fake_curr = model(test_x_hat.float())

        y_pred_real_curr = y_pred_real_curr.detach().cpu().numpy()
        y_pred_fake_curr = y_pred_fake_curr.detach().cpu().numpy()

        y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
        y_label_final = np.concatenate((np.ones([y_pred_real_curr.shape[1], ]), np.zeros([y_pred_fake_curr.shape[1], ])),
                                       axis=0)

        # Compute the accuracy
        acc = accuracy_score(y_label_final, (y_pred_final > 0.5).reshape(-1))
        discriminative_score = np.abs(0.5 - acc)

    return discriminative_score


def log_norm_spectogram(X_mb, spectrogram):
    X_mb = spectrogram(X_mb.permute(0, 2, 1)).squeeze()
    eps = 1e-10
    X_mb = X_mb + eps
    X_mb = torch.log10(X_mb)
    # X dimensions = b x f x t where f is the
    # transfer x to be - b x t x f
    X_mb = X_mb.permute(0, 2, 1)
    return X_mb


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat


def batch_generator(data, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)

    return X_mb
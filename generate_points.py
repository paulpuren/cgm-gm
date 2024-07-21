'''
Generate waveforms from a 100x100 grid (a sub-region in the SFBA) for simulating FAS maps
'''

import torch
from model.dvae import *
from argparse import Namespace
from train_hyperopt import restore_checkpoint, TimeSpecConverter
import numpy as np
from itertools import product
import pandas as pd
import torchaudio
import re
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from geopy import distance


def min_max_norm(x, x_min, x_max, range='[0,1]', mode='add'):

    if range == '[0,1]':
        if mode == 'sub':
            return (x - x_min) / (x_max - x_min)
        elif mode == 'add':
            return x * (x_max - x_min) + x_min
        else:
            raise NotImplementedError

    elif range == '[-1,1]':
        if mode == 'sub':
            return 2.0 * ((x - x_min) / (x_max - x_min)) - 1.0
        elif mode == 'add':
            x = (x + 1.0) / 2.0
            return x * (x_max - x_min) + x_min        
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    
    
def extract_parameter_values(input_string, parameter_names):
    parameter_values = {}

    for param in parameter_names:
        pattern = re.compile(f"{param}=(\d+)")
        match = pattern.search(input_string)

        if match:
            parameter_values[param] = int(match.group(1))

    return parameter_values


def get_GMMs(M, Z, Rrup):
    
    '''
    Ergodic GMM developed for the SFBA.
    '''
    
    # At 2Hz:
    Med_FAS_2 = -16.2739 + 3.0407*M  -1.4842*np.log(Rrup) + 0.0060 * Z
    Sigma_FAS_2 = 0.72 #(for all M, Rrup and Z values included)

    # At 5Hz:
    Med_FAS_5 = -13.5466 + 2.6166*M  -1.7052*np.log(Rrup) + 0.0178 * Z
    Sigma_FAS_5 = 0.77 #(for all M, Rrup and Z values included)

    # At 10Hz:
    Med_FAS_10 = -13.4538 + 2.2958*M  -1.9021*np.log(Rrup) + 0.0339 * Z
    Sigma_FAS_10 = 0.81 #(for all M, Rrup and Z values included)

    # At 15Hz:
    Med_FAS_15 = -14.4238 + 2.1082*M  -2.0292*np.log(Rrup) + 0.0461 * Z
    Sigma_FAS_15 = 0.87 #(for all M, Rrup and Z values included)
    
    return [Med_FAS_2, Med_FAS_5, Med_FAS_10, Med_FAS_15], [Sigma_FAS_2, Sigma_FAS_5, Sigma_FAS_10, Sigma_FAS_15]


def calculate_rupture_distance(r_src_lat, r_src_lon, src_depth, sample_lat, sample_lon):
    # calculate the source-site distance
    dist = []
    for slon, slat in list(product(sample_lon, sample_lat)):
        dist_val = haversine_distances([(r_src_lat, r_src_lon), (radians(slat), radians(slon))])* 6371000/1000
        dist.append(dist_val[0][-1])
    dist = np.array(dist).reshape(len(dist),1)
    rrup = np.sqrt(dist**2 + src_depth**2)
    return rrup


def KO98_smoothing(freq, y, bexp):
    nx = len(freq)
    ysmooth = np.zeros_like(y)
    max_ratio = 10 ** (3 / bexp)
    min_ratio = 1 / max_ratio
    for ix in range(nx):
        fc = freq[ix]
        if fc < 1e-6:
            ysmooth[ix] = 0
        total = 0
        window_total = 0
        for jj in range(nx):
            frat = freq[jj] / fc
            if freq[jj] < 1e-6 or frat > max_ratio or frat < min_ratio:
                continue  # skip
            elif abs(freq[jj] - fc) < 1e-6:
                window = 1
            else:
                x = bexp * np.log10(frat)
                window = (np.sin(x) / x) ** 4

            total += window * y[jj]
            window_total += window

        if window_total > 0:
            ysmooth[ix] = total / window_total
        else:
            ysmooth[ix] = 0

    return ysmooth


def compute_angle(info):

    '''
    Calculate angles between source centers and station locations
    '''

    # source location: (lat, long)
    src_lat = info['src_lat'].to_numpy() 
    src_lon = info['src_lon'].to_numpy()

    # station location: (lat, long)
    station_lat = info['sta_lat'].to_numpy() 
    station_lon = info['sta_lon'].to_numpy()

    # calculate the source-site angle
    angle = []
    for i in range(len(src_lat)):
        src_coord = (src_lat[i], src_lon[i])
        station_coord = (station_lat[i], station_lon[i])

        # Calculate the vector from the source center to the station location
        vector = np.array(station_coord) - np.array(src_coord)

        # Calculate the angle between the vectors and the x-axis (east direction)
        angle_rad = np.arctan2(vector[1], vector[0])  # Angle in radians

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        angle.append(angle_deg)
        
    angle = np.array(angle).reshape(len(angle),1) # [1471,1]

    return angle

    
data = pd.read_csv('/scratch/gm/data/Time_Series_Data_v5_EW.csv', header=None)

# assign names for the first 13 columns
names = ["eid",    #"Event ID"
        "src_lat", #"Source Lat"
        "src_lon", #"Source Lon"
        "sta_lat", #"Station Lat"
        "sta_lon", #"Station Lon"
        "depth",   #"Depth (km)"
        "mag",     #"Magnitude"
        "Sampling Frequency",
        "Number of Time Steps",
        "Delta t",
        "Min Frequency",
        "Max Frequency",
        "Number of Frequency Steps",
        "rup"]     #"rupture distance"

# define the information for the recorded data
info = data.iloc[:, :14]
info.columns = names

info['r-slat'] = info['src_lat'].apply(lambda x: radians(x))
info['r-slon'] = info['src_lon'].apply(lambda x: radians(x))
info['angle']  = compute_angle(info)
cond_var_dict = {0: ['mag', 'rup', 'angle'],
                 1: ['mag', 'rup', 'angle', 'depth'],
                 2: ['mag', 'src_lat', 'src_lon', 'sta_lat', 'sta_lon', 'depth'],
                 3: ['mag', 'depth'],
                 4: ['mag', 'rup', 'depth']}

log_path = '/scratch/gm/logs/GM_V2_VAE_data5_dist-5000_bs=128-rnn_size=32-z_dim=32-lr=0.0008-weight:kl=0.2-log_reg=True-w_decay=1e-05-w_len=160-h_len=46-ncond=16-tcondvar=2-seed=3407'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parameter_names = ["rnn_size", "z_dim", "w_len", "h_len", "tcondvar", "ncond"]
parameter_values = extract_parameter_values(log_path, parameter_names)
    
fft_size = parameter_values['w_len']
w_len = parameter_values['w_len']
h_len = parameter_values['h_len']
tcondvar = parameter_values['tcondvar']

z_dim = parameter_values['z_dim']
z_rnn_dim = parameter_values['rnn_size']
ncond = parameter_values['ncond'] if 'ncond' in parameter_values else 32

time_spec_converter = TimeSpecConverter(n_fft=fft_size, w_len=w_len, h_len=h_len, power=1, device=device)

SEQ_LEN = 6000//h_len + 1

# get the time series data
wfs = data.iloc[:, 14:].to_numpy() # [5108, 7000]

# cut waveforms to 6000, delete the first 1000 samples
wfs = wfs[:, np.newaxis, (1000):] # [5108, 1, 6000]
time_serie_len = wfs.shape[-1]

cond_vars = []
norm_dict = {}

for cvar in cond_var_dict[tcondvar]:
    cv = info[cvar].to_numpy()
    cv = cv.reshape(cv.shape[0], 1)
    cv_min, cv_max = cv.min(), cv.max()
    norm_dict[cvar] = [cv_min, cv_max]
    
    # normalized by min and max to [0, 1]
    cv = min_max_norm(cv, cv_min, cv_max, '[0,1]', 'sub')
    cond_vars.append(cv)

cond_var = np.concatenate(cond_vars, axis=1)

orig_wfs = torch.from_numpy(wfs).float()
wfs = torch.abs(time_spec_converter.time_to_spec(orig_wfs.squeeze()))

wfs += 1e-10
wfs = torch.log10(wfs)
wfs = wfs.permute(0, 2, 1)

wfs_min, wfs_max = wfs.min(), wfs.max()
norm_dict['log_wfs'] = [wfs_min, wfs_max]

# load model
model = cVAE(in_dim=fft_size, z_dim=z_dim, ncond=ncond, z_rnn_dim=z_rnn_dim, in_size=len(norm_dict)-1).to(device)

model = torch.nn.DataParallel(model, device_ids=[0])
state = dict(model=model)
best_model_state = restore_checkpoint(ckpt_dir=log_path, state=state, device=device)
best_model = best_model_state['model']
best_model.eval()


print(norm_dict)

# define a region
region = [ -122.503491204439, -121.8, 37.2661695985203, 38.066565827788]
non_ergs = pd.read_csv(f'./Non_Ergodic_Prediction_Matrix_Map_2Hz.csv', header=None) # load the non-ergodic GMM file
sample_lat = non_ergs[0]
sample_lon = non_ergs[1]
slat_idx = sample_lat[sample_lat.between(region[2], region[3])]
slon_idx = sample_lon[sample_lon.between(region[0], region[1])]

station_lat, station_lon = [], []
for slon, slat in list(zip(sample_lon.tolist(), sample_lat.tolist())):
    station_lon.append(min_max_norm(slon, norm_dict['sta_lon'][0], norm_dict['sta_lon'][-1], '[0,1]', 'sub'))
    station_lat.append(min_max_norm(slat, norm_dict['sta_lat'][0], norm_dict['sta_lat'][-1], '[0,1]', 'sub'))

good_idx = list(set(slat_idx.index.tolist()).intersection(slon_idx.index.tolist()))

# scenario 1
# chosen_lat, chosen_lon = 37.86083, -122.25667 #37.77333, -122.1898 #37.4910, -121.8290 #37.3757, -121.8261
# m, z = 3.84, 7.94
# r = 8.147416

# scenario 2
# chosen_lat, chosen_lon = 37.5777, -121.9740
# m, z = 3.98, 8.37
# r = 8.147416

chosen_lat, chosen_lon = 37.97817, -122.05350
m, z = 2.93, 15.28
# r = 8.147416

# scenario 3
# chosen_lat, chosen_lon = 37.3965, -121.7468 #37.77333, -122.1898 #37.4910, -121.8290 #37.3757, -121.8261
# m, z = 3.9000, 10.5500
# r = 8.147416


# tcond 2
norm_lat, norm_lon, norm_m, norm_z = [min_max_norm(d, norm_dict[nd][0], norm_dict[nd][-1], '[0,1]', 'sub') for d, nd in zip([chosen_lat, chosen_lon, m, z], ['src_lat', 'src_lon', 'mag', 'depth'])]
condvar = np.asarray([[norm_m, norm_lat, norm_lon, -1, -1, norm_z]])
rand_select = np.repeat(condvar, repeats=len(station_lat), axis=0)
rand_select[:, 3] = station_lat
rand_select[:, 4] = station_lon

# tcond 1
# norm_a, norm_m, norm_z, norm_r = [min_max_norm(d, norm_dict[nd][0], norm_dict[nd][-1], '[0,1]', 'sub') for d, nd in zip([angle, m, z, dist], ['angle', 'mag', 'depth', 'rup'])]
# condvar = np.asarray([[norm_m, -1, -1, norm_z]])
# rand_select = np.repeat(condvar, repeats=len(norm_a), axis=0)
# rand_select[:, 2] = norm_a
# rand_select[:, 1] = norm_r

# tcond 3
# norm_m, norm_z = [min_max_norm(d, norm_dict[nd][0], norm_dict[nd][-1], '[0,1]', 'sub') for d, nd in zip([m, z], ['mag', 'depth'])]
# condvar = np.asarray([[norm_m, norm_z]])
# rand_select = np.repeat(condvar, repeats=len(norm_m), axis=0)

# tcond 4
# norm_m, norm_r, norm_z = [min_max_norm(d, norm_dict[nd][0], norm_dict[nd][-1], '[0,1]', 'sub') for d, nd in zip([m, dist, z], ['mag', 'rup', 'depth'])]
# condvar = np.asarray([[norm_m, -1, norm_z]])
# rand_select = np.repeat(condvar, repeats=len(norm_r), axis=0)
# rand_select[:, 1] = norm_r
# print(norm_r)


rand_select = torch.from_numpy(rand_select).float().to(device)
rand_select = rand_select[good_idx]
length = rand_select.shape[0]
# print(length)

pred_wfss_sources = []
from tqdm import tqdm

for ngen in tqdm(range(1)):
    pred_wfss = []
    batch_size = 50
    batch_ranges = length//batch_size+1
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=fft_size, n_iter=500, win_length=w_len, hop_length=h_len, power=1).to(device)
    for idx in range(batch_ranges):
        with torch.no_grad():
            pred_wfs = best_model.module.generate(rand_select[idx*batch_size:(idx+1)*batch_size].to(device), SEQ_LEN)
            
            eps = 1e-10
            pred_wfs = pred_wfs.permute(0, 2, 1)
            pred_wfs = min_max_norm(pred_wfs, wfs_min, wfs_max, '[0,1]', 'add')
            pred_wfs = torch.pow(10, pred_wfs) - eps
            pred_wfss.append(griffinlim(pred_wfs).cpu().numpy())
            
    pred_wfss = np.concatenate(pred_wfss, axis=0)
    pred_wfss_sources.append(pred_wfss)

    num_data = pred_wfss.shape[0]
    L_signal = pred_wfss.shape[1]
    Fs = 100
    delta_t = 0.01
    frequency_vector = np.arange(-L_signal/2, L_signal/2) /L_signal * Fs
    indices_one_sided = np.argwhere(np.logical_and(frequency_vector>=2, frequency_vector<=15)).squeeze()
    hz_idx = [np.argwhere(frequency_vector[indices_one_sided] >= hz)[0].item() for hz in [2, 5, 10, 15]]
    fft_loop = np.fft.fft(pred_wfss)
    fft_loop_abs = np.fft.fftshift(np.abs(fft_loop), axes=-1)
    Fourier_Amplitude_Array_proper_pred = fft_loop_abs[:, indices_one_sided]*delta_t

    Amplitude_Signal_Smoothed_loop_pred = KO98_smoothing(frequency_vector[indices_one_sided], Fourier_Amplitude_Array_proper_pred.T, 20)
    np.save(f'./Gen_FAS_cond2.npy', Amplitude_Signal_Smoothed_loop_pred)
    exit()


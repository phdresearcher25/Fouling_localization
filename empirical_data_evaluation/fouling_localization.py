import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import shift
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#%%
# Settings

data_folder = './data/'

output_folder = './outputs/'


a_lamb_name = 'Steel1020_A_Lamb.txt'
s_lamb_name = 'Steel1020_S_Lamb.txt'


model_dir = './data/'

model_file_name = "model_name.pth"


device = 'cpu'  # cpu or gpu

max_helical_order =  10  
distance_tx = 0.5
pipe_diameter = 0.151

A0_velocity = 2940  

fs = 4e6
t0 = 1/fs
freq_range = (100e3, 800e3)

mean_range = (50, 4000)
corr_range = (200, 3000)

cwt_freq_range = (380e3-40e3, 380e3+40e3)
morlet_w = 9

shift_y = 5
Gx = 50
Gy = 48

#%% Visualization

def plot_dataframe(df, out_dir, f_name):
    df.plot(figsize=(10, 4))
    plt.savefig(os.path.join(out_dir, f_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cwt_stats(tx_pairs, cwt_df_max, ratio_df_max, signal_length, toa, out_dir):
    delta = 40
    for tx_pair in [tx_pairs[1], tx_pairs[2], tx_pairs[4], tx_pairs[5], tx_pairs[6], tx_pairs[7],
                    tx_pairs[9], tx_pairs[10]]:
        
        points = [int(tmp - delta) + cwt_df_max['Clean'][tx_pair][int(tmp - delta):int(tmp + delta)].argmax() for tmp in toa]
        
        plt.figure(figsize=(10, 5))
        scale = cwt_df_max['Clean'][tx_pair].max() / 1.5
        plt.plot(cwt_df_max['Clean'][tx_pair] / scale, label='Clean')
        plt.plot(cwt_df_max['Foul'][tx_pair] / scale, label='Foul')
        # plt.plot(ratio_df_max[tx_pair], label='ratio max')

        plt.hlines(1, 0, signal_length, linestyles='dashed')

        plt.scatter(points, np.ones(len(points)))
        plt.xlim(0, signal_length)
        plt.ylim(0, 1.5)
        plt.legend()
        plt.title(tx_pair)
        # plt.show()
        plt.savefig(os.path.join(out_dir, 'max_stats_{}_{}'.format(tx_pair[0], tx_pair[1])), dpi=300,
                    bbox_inches='tight')
        plt.close()
        

def plot_reconstructed_fouling(f_map, out_dir):
    plt.figure(figsize=(5, 5))
    plt.imshow(f_map, vmin=0, vmax=1)
    plt.colorbar(fraction=0.04)
    plt.title('Experimental fouling reconstructed with NNs')

    plt.savefig(os.path.join(out_dir, 'reconstructed_fouling.png'), dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_3d(f_map, distance_tx, grid, pipe_diameter, out_dir=None, file_name=None):
    """
    Non-interactive: no fig.show, no offline.plot, no write_image.
    Only writes a standalone HTML file using basic file I/O.
    """
    # Get cylinder coordinates for plotting in 3D
    x_2d = np.linspace(0, distance_tx, grid[1])
    y_2d = np.linspace(np.pi, 3 * np.pi, grid[0])  
    f_map = np.roll(f_map, int(grid[0] / 2), axis=0)

    phi, Z = np.meshgrid(y_2d, x_2d)
    X = pipe_diameter / 2 * np.cos(phi)
    Y = pipe_diameter / 2 * np.sin(phi)

    location_tx = np.array([
        [pipe_diameter / 2, 0, 0],
        [-pipe_diameter / 2, 0, 0],
        [0, pipe_diameter / 2, distance_tx],
        [0, -pipe_diameter / 2, distance_tx]
    ]) * 100

    # Build the Plotly figure (in memory only)
    fig = go.Figure(layout=go.Layout(height=900, width=1200))

    fig.add_trace(go.Surface(
        x=X.T * 100,
        y=Y.T * 100,
        z=Z.T * 100,
        surfacecolor=f_map,
        opacity=1,
        colorscale='Jet',
        cmin=0,
        cmax=1
    ))

    fig.add_trace(go.Scatter3d(
        x=location_tx[:, 0],
        y=location_tx[:, 1],
        z=location_tx[:, 2],
        mode="markers",
        marker=dict(size=10)
    ))

    fig.update_layout(
        scene_camera=dict(
            up=dict(x=1, y=0, z=0),
            eye=dict(x=0, y=3.0, z=0)
        )
    )

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save as a self-contained HTML file 
    html_path = os.path.join(out_dir, f"3D_fouling_{file_name[-12:]}.html")
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)     
        
        
#%% Functions

def load_data(root_dir, f_name):
    data_df = pd.read_csv(os.path.join(root_dir, f_name), delimiter='\t')
    data_df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    data_df.set_index('time', inplace=True)

    return data_df


def data_preproc(dict_names, dat_dir, out_dir, mean_range, corr_range):
    d = {}
    for condition in list(dict_names.keys()):
        for inx, file_name in enumerate(dict_names[condition]):
            data_df = load_data(root_dir=dat_dir, f_name=file_name)

            if condition == 'Clean' and inx == 0:
                repetitions = data_df.groupby('time').size()[0]
                signal_length = data_df.groupby('time').size().shape[0]
                data_df_corr, lags_df = correct_jitter(data_df, mean_range=mean_range, corr_range=corr_range,
                                                        repetitions=repetitions, signal_length=signal_length)


                if out_dir != None:
                    plot_dataframe(df=lags_df, out_dir=out_dir, f_name='lag_df.png')
                    plot_dataframe(df=lags_df, out_dir=out_dir, f_name='data_df_corr.png')

            tmp, _ = correct_jitter(data_df,
                                    mean_range=mean_range,
                                    corr_range=corr_range,
                                    repetitions=repetitions,
                                    signal_length=signal_length)
            d[condition, tmp.columns[0][:-1] + file_name[-5]] = pd.DataFrame(columns=tmp.columns, data=tmp)

    data_df_corr_all = pd.concat(d, axis=1)
    # logging.info('-- dataframe head:\n{}'.format(data_df_corr_all.to_string()))

    # Correlate clean and foul for accurate amp ratio retrival
    tx_pair_all = []
    print('\n-- (Tx, Rx) --> lag:')
    for tx_pair in data_df_corr_all['Clean'].columns:
        tx_pair_all.append(tx_pair)
        lag = lag_finder(data_df_corr_all['Clean'][tx_pair][corr_range[0]:corr_range[1]],
                         data_df_corr_all['Foul'][tx_pair][corr_range[0]:corr_range[1]])
        print('  -- {} --> {}'.format(tx_pair, lag))
        data_df_corr_all.loc[:, ('Foul', tx_pair[0], tx_pair[1])] = shift(data_df_corr_all['Foul'][tx_pair], -lag)

    return data_df_corr_all, signal_length


def get_cwt_stats(data_df, tx_pairs, cwt_freq_range, wavelet_w, sampling_rate):
    # Get all cwt and calculate foul/clean ratio based on mean or max
    # ratio_df_max = pd.DataFrame(0, index=data_df_corr_all['Clean'].index, columns=data_df_corr_all['Clean'].columns)
    ratio_df_max = data_df['Clean'].copy() * 0
    ratio_df_mean = data_df['Clean'].copy() * 0
    cwt_df_max = data_df.copy() * 0

    for tx_pair in tx_pairs:
        cwt_clean = make_cwt(data_df['Clean'][tx_pair], fs=sampling_rate, freq_range=cwt_freq_range, w=wavelet_w)
        cwt_foul = make_cwt(data_df['Foul'][tx_pair], fs=sampling_rate, freq_range=cwt_freq_range, w=wavelet_w)
        cwt_df_max.loc[:, ('Clean', tx_pair[0], tx_pair[1])] = np.abs(cwt_clean).max(0)
        cwt_df_max.loc[:, ('Foul', tx_pair[0], tx_pair[1])] = np.abs(cwt_foul).max(0)
        ratio_df_max[tx_pair] = np.abs(cwt_foul).max(0) / np.abs(cwt_clean).max(0)
        ratio_df_mean[tx_pair] = np.abs(cwt_foul).mean(0) / np.abs(cwt_clean).mean(0)

    return cwt_df_max, ratio_df_max, ratio_df_mean


def lag_finder(y1, y2, sr=1):
    n = len(y1)

    corr = np.correlate(y2, y1, mode='same') / np.sqrt(np.correlate(y1, y1, mode='same')[int(n/2)] * np.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)
    delay = delay_arr[np.argmax(corr)]
    
    return delay

def correct_jitter(data_df, mean_range, corr_range, repetitions, signal_length, align_col=False):
 
    lags_df = pd.DataFrame()
    data_df_corr = pd.DataFrame()
    for j, col in enumerate(data_df.columns):
        data_df_rep = pd.DataFrame()
        for i in range(repetitions):
            # reshape repetitions column into separate columns 
            data_df_rep[str(i+1)] = data_df[col][i*signal_length:(i+1)*signal_length]
 
            if i == 0: # fix if the first rep is not nan
                y1 = data_df_rep['1']
                y1 -= y1[mean_range[0]:mean_range[1]].mean()
                lags = [0]
            else:
                yi = data_df_rep[str(i+1)]
                yi -= yi[mean_range[0]:mean_range[1]].mean()
                if np.isnan(yi).any():
                    lag = 0
                else:
                    lag = lag_finder(y1[corr_range[0]:corr_range[1]], yi[corr_range[0]:corr_range[1]])
                lags.append(lag)
        lags -= np.mean(lags)    
        for i in range(repetitions):
            yi = data_df_rep[str(i+1)]
            yi -= yi[mean_range[0]:mean_range[1]].mean()
            data_df_rep[str(i+1)] = shift(yi, -lags[i])
        # excluding abnormal amplitude data (too different from mean)
        mask_outliers = np.abs(data_df_rep.max(axis=0) / data_df_rep.max(axis=0).mean() - 1)*100 < 20        
        if mask_outliers.sum() < repetitions:
            print('   *** Max signal variation > 20% from mean. Excluding those from mean...')
            print(((data_df_rep.max(axis=0) / data_df_rep.max(axis=0).mean() - 1)*100).values)
        lags_df[col] = lags
        data_df_corr[col] = data_df_rep.loc[:, mask_outliers].mean(1)
 
    return data_df_corr, lags_df


def get_path_length(max_order, distance_x, distance_y, pipe_dia):
    path_length = []
    pipe_circ = np.pi * pipe_dia
    for order in range(round(max_order/2)+1):
        if order == 0:
            path_length.append(np.sqrt(distance_x**2 + distance_y**2))
        else:
            path_length.append(np.sqrt(distance_x**2 + (distance_y - order * pipe_circ)**2))
            path_length.append(np.sqrt(distance_x**2 + (distance_y + order * pipe_circ)**2))
    return np.asarray(path_length)[:max_order]


def make_cwt(data_df, fs, freq_range=(1e3, 800e3), w=5, out_dir=None, plot_cwt=False, plot_dispertion=False, t0=None,
             path_length=None, A_lamb=None, S_lamb=None):

    t = data_df.index.values/fs

    freq = np.linspace(freq_range[0], freq_range[1],101)

    widths = fs * w / (freq * 2*np.pi)

    cwtm = signal.cwt(data_df, signal.morlet2, widths, w=w)

    if plot_cwt:
        plt.figure(figsize=(15, 7))
        plt.pcolormesh(t*1e3, freq/1e3, np.abs(cwtm)/np.abs(cwtm).max(), cmap='turbo', shading='auto')
        #plt.pcolormesh(t*1e3, freq/1e3, mask/2, cmap='turbo', shading='auto')
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [ms]')

        plt.colorbar()
        plt.xlim(0, max(t*1e3))
        plt.ylim(freq_range[0]/1e3, freq_range[1]/1e3)

        if plot_dispertion:
            for dist in path_length:
                plt.plot((t0 + dist/A_lamb[:, 2])*1e3, A_lamb[:, 0], c='w')
                plt.plot((t0 + dist/S_lamb[:, 2])*1e3, S_lamb[:, 0], c='y')
        plt.title(data_df.name)
        # plt.show()
        plt.savefig(os.path.join(out_dir, 'CWT.png'), dpi=300, bbox_inches='tight')
    return cwtm


def get_amp_ratio_at_max(tx_pairs, cwt_df_max, toa):
    # Get amp ratio at the maximum of the peaks
    delta = 40  # find peak max at toa-delta and toa+delta
    
    # transducer_pairs = [1, 2, 5, 6]
    transducer_pairs = [1,2,4,5] 
    
    ratio = []
    print(tx_pairs) 
    for mask, tx_pair in enumerate(tx_pairs):
        if mask in transducer_pairs:  
            for i, tmp in enumerate(toa):
                # point = cwt_df_max['Clean'][tx_pair][int(tmp - delta):int(tmp + delta)].argmax()
                point = int(tmp-delta) + cwt_df_max['Clean'][tx_pair][int(tmp - delta):int(tmp + delta)].argmax()
                tmp = cwt_df_max['Foul'][tx_pair][point] / cwt_df_max['Clean'][tx_pair][point]
                ratio.append(tmp)
    
    #print(ratio)
    return ratio

def get_neural_nets(model_name, device):
    import define_model
    # Dynamically get the model class by its name
    ModelClass = getattr(define_model, model_name)
    # Instantiate the model and move it to the specified device
    model = ModelClass().to(device)
    
    return model


def load_model(model, base_dir, model_f_name):
    model.load_state_dict(torch.load(os.path.join(base_dir, model_f_name), map_location=torch.device('cpu')))
    return model

#%%

# RAUSCU DATA
clean_file_name = 'RAUSmeasurement_2024-03-18T143239'
foul_file_name = 'RAUSmeasurement_2024-03-18T144324'  # F1

job_index = "F1"


output_dir = os.path.join(output_folder, job_index)
result_dir = os.path.join(output_dir, 'results')
debug_dir = os.path.join(output_dir, 'debug')


# Create the directory if it doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Create the directory if it doesn't exist
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)


clean_file_names = [x for x in os.listdir(data_folder) if clean_file_name in x]
foul_file_names = [x for x in os.listdir(data_folder) if foul_file_name in x]

clean_file_names = np.sort(clean_file_names)
foul_file_names = np.sort(foul_file_names)

# Load all data, correlate reps and return mean
d_name = {'Clean': clean_file_names, 'Foul': foul_file_names}

data_df_corr_all, signal_length = data_preproc(dict_names=d_name, dat_dir=data_folder, out_dir= debug_dir,
                                                mean_range=mean_range, corr_range=corr_range)

tx_pair_all = [x for x in data_df_corr_all['Clean'].columns]

# Calculate trajectory length for helical paths
path_length = get_path_length(max_order=max_helical_order, distance_x=distance_tx,
                              distance_y=pipe_diameter * np.pi / 4, pipe_dia=pipe_diameter)


# Multiplying time by the sampling frequency converts the time from seconds to sample indices
toa = (t0 + path_length / A0_velocity) * fs

cwt_df_max, ratio_df_max, ratio_df_mean = get_cwt_stats(data_df=data_df_corr_all, tx_pairs=tx_pair_all,
                                                            cwt_freq_range=cwt_freq_range,
                   
                                                            wavelet_w=morlet_w, sampling_rate=fs)

plot_cwt_stats(tx_pairs=tx_pair_all, cwt_df_max=cwt_df_max, ratio_df_max=ratio_df_max, signal_length=signal_length,
                toa=toa, out_dir=debug_dir)



exp_input = get_amp_ratio_at_max(tx_pair_all, cwt_df_max, toa)
exp_input = np.reshape(exp_input, (1, 40))


exp_input[exp_input > 1] = 1.
exp_input[exp_input < 0] = 0.
print("EXP INPUT: ", exp_input)


#%%
# Load trained NN model
model = get_neural_nets("BasicNetwork", device=device)
model = load_model(model, base_dir=model_dir, model_f_name=model_file_name)
model.eval()

# Generate fouling map
F_map = model(torch.tensor(exp_input, dtype=torch.float32)).detach().cpu().numpy()
F_map = np.flipud(np.roll(F_map.reshape(Gy, Gx), shift_y, axis=0))

plot_reconstructed_fouling(f_map=F_map, out_dir=result_dir)

plot_3d(F_map, distance_tx, grid=(Gy, Gx), pipe_diameter=pipe_diameter, out_dir=result_dir, file_name=foul_file_name)

#%%




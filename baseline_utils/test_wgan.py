import os, hdf5storage
from baseline_utils.wgan_helper import *
import torch
import numpy as np
import matplotlib.pyplot as plt


def loadValidateChannels(config, seed, mu_train, std_train):
    channels = np.array([], dtype='complex64')
    for scenario in config.scenario_list:
        fileName = './DeepMIMO-5GNR/DeepMIMO_dataset/%s_path%d_seed%d.mat' % (
            scenario, config.num_paths, seed)
        contents = hdf5storage.loadmat(fileName)
        channel_scenario = np.asarray(contents['channels'], dtype=np.complex64)
        if len(channels) < 1:
            channels = channel_scenario
        else:
            np.concatenate((channels, channel_scenario), 0)
    channels = np.transpose(channels, (1, 2, 0))

    # normalize
    # mu = np.zeros([1])
    # std = np.std(channels)
    channels = (channels - mu_train)/std_train


    return np.asarray(channels)


# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True


#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65
train_seed, val_seed = 1111, 2222

length = int(N_t/4)
breadth = int(N_r/4)

G_test = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, 128*length*breadth),
    torch.nn.ReLU(),
    View([1,128,length,breadth]),
    torch.nn.Upsample(scale_factor=2),
    Conv2d(128,128,4,bias=False),
    torch.nn.BatchNorm2d(128,momentum=0.8),
    torch.nn.ReLU(),
    torch.nn.Upsample(scale_factor=2),
    Conv2d(128,128,4,bias=False),
    torch.nn.BatchNorm2d(128,momentum=0.8),
    torch.nn.ReLU(),
    Conv2d(128,2,4,bias=False),
)
G_test = G_test.type(dtype)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--train', type=str, default='I2_28B')
parser.add_argument('--test', type=str, default='I2_28B')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--num_paths', type=int, default=10)
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
config = parser.parse_args()


dft_basis = sio.loadmat("../data/dft_basis.mat")
A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)

########################## Get statistics of training data ##########################
if not config.train == 'mixed':
    config.train_scenario_list = [config.train]
else:
    config.train_scenario_list = ['O1_28','O1_28B','I2_28B']
H_ori = None
for scenario in config.train_scenario_list:
    H_org = sio.loadmat("./DeepMIMO-5GNR/DeepMIMO_dataset/%s_path10_seed%d.mat" % (scenario, train_seed))
    if H_ori is None:
        H_ori = H_org['channels'].transpose((1,2,0))
    else:
        H_ori = np.concatenate((H_ori,H_org['channels'].transpose((1,2,0))),-1)
mu_train = np.zeros([1])
std_train = np.std(H_ori)

########################## Get normalized validation data ######################################
# H_org = sio.loadmat("./DeepMIMO-5GNR/DeepMIMO_dataset/%s_path10_seed%d.mat" % (config.test, val_seed))
# H_ori = H_org['channels'].transpose((1,2,0))
if not config.test == 'mixed':
    config.test_scenario_list = [config.train]
else:
    config.test_scenario_list = ['O1_28','O1_28B','I2_28B']
H_ori = None
for scenario in config.test_scenario_list:
    H_org = sio.loadmat("./DeepMIMO-5GNR/DeepMIMO_dataset/%s_path10_seed%d.mat" % (scenario, val_seed))
    if H_ori is None:
        H_ori = H_org['channels'][:100].transpose((1,2,0))
    else:
        H_ori = np.concatenate((H_ori,H_org['channels'][:100].transpose((1,2,0))),-1)
# normalize
H_ori = (H_ori-mu_train)/std_train

H_extracted = np.transpose(copy.deepcopy(H_ori),(2,1,0))
for i in range(H_ori.shape[2]):
    H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))
# H_extracted_real = np.real(H_extracted)
# H_extracted_imag = np.imag(H_extracted)
# H_extracted_real = (H_extracted_real - mu_real)/std_real # normalize
# H_extracted_imag = (H_extracted_imag - mu_imag)/std_imag

H_extracted_real = np.real(H_extracted)
H_extracted_imag = np.imag(H_extracted)


A_T_R = np.kron(A_T.conj(),A_R)
A_T_R_real = dtype(np.real(A_T_R))
A_T_R_imag = dtype(np.imag(A_T_R))

N_s = N_r
N_rx_rf = N_r
Nbit_t = 6
Nbit_r = 2
angles_t = np.linspace(0,2*np.pi,2**Nbit_t,endpoint=False)
angles_r = np.linspace(0,2*np.pi,2**Nbit_r,endpoint=False)
freq = 2000
model_vec = range(58000,60000,freq)

def training_precoder(N_t,N_s):
    angle_index = np.random.choice(len(angles_t),(N_t,N_s))
    return (1/np.sqrt(N_t))*np.exp(1j*angles_t[angle_index])

def training_combiner(N_r,N_rx_rf):
    angle_index = np.random.choice(len(angles_r),(N_r,N_rx_rf))
    W = (1/np.sqrt(N_r))*np.exp(1j*angles_r[angle_index])
    return np.matrix(W).getH()

ntest = 5
nrepeat = 100
SNR_vec = np.arange(-10,32.5,2.5)
# SNR_vec = np.array([30, 10, -10])
alpha = 0.6
nmse_all = np.zeros((len(SNR_vec),len(model_vec),nrepeat,ntest))
N_p = int(alpha*N_t)
qpsk_constellation = (1/np.sqrt(2))*np.array([1+1j,1-1j,-1+1j,-1-1j])

pilot_sequence_ind = np.random.randint(0,4,size=(N_s,N_p))
symbols = qpsk_constellation[pilot_sequence_ind]
precoder_training = training_precoder(N_t,N_s)
W = training_combiner(N_r,N_rx_rf)
A = np.kron(np.matmul(symbols.T,precoder_training.T),W)

A_real = dtype(np.real(A))
A_imag = dtype(np.imag(A))
identity = np.identity(N_r)
lambda_reg = 1e-3

for midx, model in enumerate(model_vec):
    G_test.load_state_dict(torch.load('models/wgan_gp/%s/generator%d.pt'%(config.train,model)))
    G_test.eval()
    for ind in range(nrepeat):
        for snr_idx, SNR in enumerate(SNR_vec):
            for i in range(ntest):
                vec_H_single = np.reshape(H_ori[:,:,ind].flatten('F'),[N_r*N_t,1])
                signal = np.matmul(H_ori[:,:,ind],np.matmul(precoder_training,symbols)) # [n_rx, n_pilot]
                E_s = np.multiply(signal,np.conj(signal)) #[n_r, n_pilot]
                noise_matrix = (1/np.sqrt(2))*(np.random.randn(N_r,N_p)+1j*np.random.randn(N_r,N_p))
                vec_y = np.zeros((N_rx_rf*N_p,1,1),dtype='complex64') #[n_rx*n_pilot, 1, 1]
                std_dev = (1/(10**(SNR/20)))*np.sqrt(E_s)
                rx_signal = signal + np.multiply(std_dev,noise_matrix)
                rx_signal = np.matmul(W,rx_signal)
                vec_y[:,0,0] = rx_signal.flatten('F') # [n_rx*n_pilot, 1]
                vec_y_real = dtype(np.real(vec_y[:,:,0]))
                vec_y_imag = dtype(np.imag(vec_y[:,:,0]))
                def gen_output(x):
                    pred = G_test(x)
                    # pred[0,0,:,:] = dtype(std_real)*pred[0,0,:,:] + dtype(mu_real)
                    # pred[0,1,:,:] = dtype(std_imag)*pred[0,1,:,:] + dtype(mu_imag)
                    #################################### check this part ###########################
                    pred_real = torch.mm(A_T_R_real,pred[0,0,:,:].view(N_t*N_r,-1)) - torch.mm(A_T_R_imag,pred[0,1,:,:].view(N_t*N_r,-1))
                    pred_imag = torch.mm(A_T_R_real,pred[0,1,:,:].view(N_t*N_r,-1)) + torch.mm(A_T_R_imag,pred[0,0,:,:].view(N_t*N_r,-1))
                    diff_real = vec_y_real - torch.mm(A_real,pred_real) + torch.mm(A_imag,pred_imag)
                    diff_imag = vec_y_imag - torch.mm(A_real,pred_imag) - torch.mm(A_imag,pred_real)
                    diff = torch.norm(diff_real) ** 2 + torch.norm(diff_imag) ** 2
                    # pred_complex = pred[0,0,:,:] + pred[0,1,:,:]*1j
                    # pred_H = torch.mm(torch.mm(torch.tensor(A_R, dtype=torch.complex64).conj().T.cuda(),pred_complex.T),torch.tensor(A_T,dtype=torch.complex64).cuda())
                    # non_noisy = torch.matmul(pred_H,torch.tensor(np.matmul(precoder_training,symbols),dtype=torch.complex64).cuda())
                    # diff = torch.tensor(rx_signal,dtype=torch.complex64).cuda() - torch.matmul(torch.tensor(W,dtype=torch.complex64).cuda(),non_noisy)
                    # diff = torch.tensor(H_ori[:,:,ind],dtype=torch.complex64).cuda() - pred_H
                    # diff = diff.real.abs().sum().square() + diff.imag.abs().sum().square()
                    return diff + lambda_reg*torch.norm(x)**2
                x = Variable(torch.randn(1, latent_dim)).type(dtype)
                x.requires_grad = True
                learning_rate = 0.02
                optimizer = torch.optim.Adam([x], lr=learning_rate)
                for a in range(200):
                    optimizer.zero_grad()
                    loss = gen_output(x)
                    loss.backward()
                    optimizer.step()
                gen_imgs = G_test(x).data.cpu().numpy()
                # gen_imgs[0,0,:,:] = std_train.real * gen_imgs[0,0,:,:] + mu_train.real
                # gen_imgs[0,1,:,:] = std_train.imag * gen_imgs[0,1,:,:] + mu_train.img
                gen_imgs_complex = gen_imgs[0,0,:,:] + 1j*gen_imgs[0,1,:,:]
                gen_imgs_complex = np.matmul(A_T_R,np.reshape(gen_imgs_complex,[N_t*N_r,1]))

                # print((np.linalg.norm(gen_imgs_complex - vec_H_single)/np.linalg.norm(vec_H_single))**2)
                val_nmse_all = (np.sum(np.square(np.abs(gen_imgs_complex - vec_H_single)))/ np.sum(np.square(np.abs(vec_H_single))))
                nmse_all[snr_idx, midx, ind, i] = val_nmse_all
                print(SNR, model, val_nmse_all)
        print(SNR, nmse_all[snr_idx, 0,:].min(-1).mean())

avg_nmse = nmse_all.min(axis=-1).mean(axis=-1)

result_dir = 'results/wgan_gp/train%s_test%s' % (config.train, config.test)
os.makedirs(result_dir, exist_ok=True)
plt.figure(figsize=(10, 10))
plt.plot(SNR_vec, 10 * np.log10(avg_nmse), linewidth=4, label='test scenario: %s' % config.test)
plt.grid()
plt.legend()
plt.title('WGAN based channel estimation')
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results.png'), dpi=300,
                bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(SNR_vec, avg_nmse,  linewidth=4, label='test scenario: %s' % config.test)
plt.grid()
plt.legend()
plt.title('WGAN based channel estimation')
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results_mse.png'), dpi=300,
                bbox_inches='tight')
plt.close()

save_dict = {'nmse_all': nmse_all,
                 'avg_nmse': avg_nmse,
                 'pilot_alpha': alpha,
                 'snr_range': SNR_vec,
                 'config': config,
                 }
torch.save(save_dict, os.path.join(result_dir, 'results.pt'))
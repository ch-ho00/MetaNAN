import numpy as np
import torch
import random 
from configs.local_setting_degae import EVAL_CONFIG, TRAIN_CONFIG
from configs.config import CustomArgumentParser
import sys
from nan.dataloaders.create_training_dataset import create_training_dataset
from nan.dataloaders.basic_dataset import Mode, de_linearize
from nan.dataloaders import dataset_dict
from degae.model import DegAE
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


sys.argv = sys.argv + ['--config', str(EVAL_CONFIG)]

# Create training args
parser = CustomArgumentParser.config_parser()
train_args = parser.parse_args(verbose=True)

for key, value in sorted(vars(train_args).items()):
    print(f"{key:<30}: {value}")

# Set distributed options
if train_args.distributed:
    torch.cuda.set_device(train_args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

val_dataset = dataset_dict["llff_test"](train_args, Mode.validation, scenes=['horns', 'trex', 'fern'])
val_dataset.set_noise_param_list()

load_ckpt = './out/srgan_woEmbedExtractClamp_1e-3_1e1_1e0_5e-3__NAN/model_115000.pth'
model = DegAE(train_args, train_scratch=False)
checkpoint = torch.load(load_ckpt, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["model"])


num_sample = len(val_dataset.all_combination)
all_combi = list(range(len(val_dataset.all_combination)))
sampled_idxs = random.choices(all_combi, k=num_sample)

embeddings = []
all_hparams = {}

skip_forward = True

if not skip_forward:
    for gain_level in [1, 20, 16, 8]: #,4,
        for idx in tqdm(sampled_idxs):
            rgb_d1, parsed_hparams = val_dataset.get_singleview_param(idx, eval_gain=gain_level)
            degrad_vec = model.degrep_extractor(rgb_d1.cuda(), parsed_hparams['white_level'].cuda())
            embeddings.append(degrad_vec)

            for k in parsed_hparams:
                if k not in all_hparams.keys():
                    all_hparams[k]  = [parsed_hparams[k]]
                else:
                    all_hparams[k] += [parsed_hparams[k]]
            
            if 'gain_level' not in all_hparams.keys():
                all_hparams['gain_level'] = [gain_level]            
            else:
                all_hparams['gain_level'] += [gain_level]
                
            if random.random() < 0.01:
                s1 = parsed_hparams['omega_c']
                s2 = parsed_hparams['kernel']
                s3 = parsed_hparams['blur_sigma']
                s4 = parsed_hparams['betag_range']
                s5 = parsed_hparams['betap_range']
                s6 = parsed_hparams['eval_gain']
                plt.imsave(f'./results/sampled_patches/{s1}_{s2}_{s3}_{s4}_{s5}_{s6}.png', de_linearize(rgb_d1[0].permute(1,2,0), parsed_hparams['white_level']).cpu().clamp(0,1).numpy())

    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, f'./results/embedding_{num_sample}.pt')
    for k in all_hparams.keys():
        if not isinstance(all_hparams[k][0], str):
            if isinstance(all_hparams[k][0], torch.Tensor):
                all_hparams[k] = torch.stack(all_hparams[k]).cpu().reshape(-1).numpy()            
            else:
                all_hparams[k] = np.stack(all_hparams[k])
            print(k, all_hparams[k].shape)
            
    with open(f'./results/hparams_{num_sample}.pkl', 'wb') as fp:
        pickle.dump(all_hparams, fp)

with open(f'./results/hparams_{num_sample}.pkl', 'rb') as fp:
    hparams = pickle.load(fp)

### 
with open(f'./results/hparams_{num_sample}.pkl', 'rb') as fp:
    hparams = pickle.load(fp)

embeddings = torch.load(f'./results/embedding_{num_sample}.pt')
import torch
from torch.nn.functional import pairwise_distance
K = 5
closest_hparams = {'retrieved_'+ (k) : [] for k in hparams.keys()}
dict_ = {'query_' +k : [] for k in hparams.keys()}
closest_hparams.update(dict_)

for gain_level in [1, 20, 16, 8]: #,4,
    for idx in tqdm(sampled_idxs):
        rgb_d1, parsed_hparams = val_dataset.get_singleview_param(idx, eval_gain=gain_level)
        query_vec = model.degrep_extractor(rgb_d1.cuda(), parsed_hparams['white_level'].cuda())

        distances = pairwise_distance(query_vec, embeddings)
        _, indices = torch.topk(distances, k=K, largest=False)
        for k in hparams.keys():
            if k  == 'gain_level':
                continue
            for indice in indices.cpu().int().numpy():            
                closest_hparams['retrieved_' + k] += [hparams[k][indice]]
                closest_hparams['query_' + k] += [parsed_hparams[k]]

prefix = 'firstRun'

val2idx = {round(val,3) : idx for idx, val in enumerate(np.linspace(0,1, 4))}
val2idx[0.666] = 2
val2idx[0.334] = 1
kernel2idx = {
    kernel : idx for idx, kernel in enumerate( ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]) 
}

for k in hparams.keys():        
    print(k)
    if k == 'white_level':
        continue
    if k == 'kernel':
        retrieved = [kernel2idx[val] for val in closest_hparams['retrieved_'+ k]]
        query =  [kernel2idx[val] for val in closest_hparams['query_'+ k] ]
    elif k == 'eval_gain':
        retrieved = [val for val in closest_hparams['retrieved_'+ k]]
        query =  [val for val in closest_hparams['query_'+ k] ]
    else:
        retrieved = [val2idx[val] for val in closest_hparams['retrieved_'+ k]]
        query = [val2idx[val] for val in closest_hparams['query_'+ k] ]
        
    cm = confusion_matrix(query, retrieved)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'./results/{prefix}_{num_sample}_{k}_confusion.png')
    plt.clf()
print()        

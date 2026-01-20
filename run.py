import argparse
import os
os.environ['WANDB_MODE'] = 'disabled'      
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from exp.exp_forecast import Exp_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_anomaly_detection_classification import Exp_Anomaly_Detection_classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_train_CPRFL import Exp_train_CPRFL
from utils.tools import HiddenPrints



def main(dataset, d, a):
    parser = argparse.ArgumentParser(description='Large Time Series Model')      
    parser.add_argument('--task_name', type=str, default='Timer_xl_Exp_Anomaly_Detection',
                        help='task name, options:[forecast, imputation, anomaly_detection,'
                             'Exp_Anomaly_Detection,Exp_Anomaly_Detection_classification,'
                             'Timer_xl_Exp_Anomaly_Detection,CPRFL_train]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model', type=str, default='model_CPRFL',
                        help='model name, options: [Timer TrmEncoder Timer_XL MSDNN model_CPRFL'
                             'Tim _LLM,TimeMixer, Autoformer,FEDformer, Crossformer, ETSformer, Informer, iTransformer, PatchTST, TimesNet]'
                        "AGSX,AICTRCD,ECGTransForm,EffNet,CNN,densenet,resnet,MS_Mamba,Net_1d(ECGFounder)"
                        )

    parser.add_argument('--seed', type=int, default=0, help='random seed')      

    parser.add_argument('--weather_use_pretrain_model', type=bool, default=True, help='True,False')
    parser.add_argument('--loss_name', type=str, default='BCELoss',
                        help='BCELoss,ASLloss,onlyCLIP,CLIP+ASLloss,CLIP+BCELoss')
    parser.add_argument('--dataset', type=str, default=dataset, help='MIMIC,PTBXL,SPH,G12EC')
    parser.add_argument('--pretrain_dataset', type=str, default='SPH', help='MIMIC,PTBXL,SPH')
    parser.add_argument('--root_path', type=str, default='/.../PTBXL_diagnostic',
                        help='root path of the data file')
    parser.add_argument('--backbone', type=str, default='MSDNN',
                        help='model name, options: [MSDNN,Autoformer,Densenet,resnet]')
    parser.add_argument('--llm_model', type=str, default='BERT',
                       help='model name, options: [LLAMA, GPT2, BERT')
    parser.add_argument('--num_classes', type=int, default=44, help='num_classes')      
    parser.add_argument('--a', type=float, default=a, help='损失函数中的超参数a')
    parser.add_argument('--d', type=float, default=d, help='可学习参数维度d,128,256,512,1024')


    parser.add_argument('--data_path', type=str, default='X_smooth_100Hz.npy', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)      
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=12, help='output size')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')      
    parser.add_argument('--seq_len', type=int, default=768, help='input sequence length')
    parser.add_argument('--input_token_len', type=int, default=96, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='output token length')
    parser.add_argument('--patch_len', type=int, default=96, help='input sequence length')


    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')      
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=120, help='train epochs')##有用的epoch
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learcn ing rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)      
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=3, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--ckpt_path', type=str, default='/.../Timer_anomaly_detection_1.0.ckpt', help='ckpt file')      
    parser.add_argument('--finetune_rate', type=float, default=0.1, help='finetune ratio')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--use_norm', action='store_true', help='use norm', default=True)
    parser.add_argument('--flash_attention', action='store_true', help='flash attention', default=False)
    parser.add_argument('--covariate', action='store_true', help='use cov', default=False)


    parser.add_argument('--subset_rand_ratio', type=float, default=0.01, help='mask ratio')
    parser.add_argument('--data_type', type=str, default='custom', help='data_type')

    parser.add_argument('--decay_fac', type=float, default=0.75)      
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)      
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)      
    parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=False)
    parser.add_argument('--output_len', type=int, default=96, help='output len')
    parser.add_argument('--output_len_list', type=int, nargs="+", help="output_len_list")      
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--is_finetuning', type=bool, default=True, help='status：True,finturn; False:test')      
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--llm_layers', type=int, default=6)


    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))      
        rank = int(os.environ.get("RANK", "0"))      
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()      
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)


    if args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'Timer_xl_Exp_Anomaly_Detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'CPRFL_train':
        Exp = Exp_train_CPRFL
    elif args.task_name == 'Exp_Anomaly_Detection_classification':
        Exp = Exp_Anomaly_Detection_classification
    elif args.task_name in ['anomaly_detection', "Exp_Anomaly_Detection"]:
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'forecast':
        Exp = Exp_Forecast

    else:
        raise ValueError('task name not found')

    if args.dataset == 'PTBXL':
        args.num_classes = 44
    elif args.dataset == 'MIMIC':
        args.num_classes = 102
    elif args.dataset == 'SPH':
        args.num_classes = 44
    elif args.dataset == 'G12EC':
        args.num_classes = 26
        args.learning_rate =0.0005


    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        if args.is_finetuning:
            args.ckpt_path = "/.../timer_xl_checkpoint.pth"
            args.save_path = "..._"+str(args.model)+"a="+str(args.a)+"d=+"+str(args.d)+args.backbone+"_pretrain_"+args.llm_model+"_"+args.dataset+"_"+str(args.num_classes)+"class/"      
            setting = '{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.dataset,
                )
            setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")

            exp = Exp(args)      
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.finetune(setting)


        else:
            args.ckpt_path = "/.../Timer_anomaly_detection_1.0.ckpt"
            setting = 'test_plot_{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
            )

            setting += datetime.now().strftime("%y-%m-%d")
            exp = Exp(args)      

if __name__ == '__main__':
    dataset_all = ["PTBXL","SPH", "G12EC"]      
      
    d =[256]
    for dataset in dataset_all:      
        if dataset == 'PTBXL':
            a = 1
            print("PTBXL")
        elif dataset == 'SPH':
            a = 0.4
        elif dataset == 'G12EC':
            a = 1
        for d_ in d:
            main(dataset, d_, a)



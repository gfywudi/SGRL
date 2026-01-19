import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch

      
from data_provider.data_loader_benchmark import CIDatasetBenchmark, \
    CIAutoRegressionDatasetBenchmark


def collate_fn(batch):
    input_ids = [sample[2]['input_ids'] for sample in batch]
    attention_masks = [sample[2]['attention_mask'] for sample in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)        
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return input_ids, attention_masks


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ['test',"valid"]:
        shuffle_flag = False
        drop_last = True
        batch_size = 128       
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size        
        freq = args.freq

    if args.task_name == 'forecast':
        if args.use_ims:
            data_set = CIAutoRegressionDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.output_len if flag == 'test' else args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        else:
            data_set = CIDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                pred_len=args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False)
        return data_set, data_loader
    elif args.task_name in ['Exp_Anomaly_Detection_classification' ,'Exp_Anomaly_Detection','Timer_xl_Exp_Anomaly_Detection',"CPRFL_train"]:
        if args.task_name == 'Exp_Anomaly_Detection_classification':
            from data_provider.data_loader import PTBXL,PTBXL_all_cycle
            data_set = PTBXL_all_cycle(
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag,)
        elif args.task_name in ['anomaly_detection', "Exp_Anomaly_Detection","Timer_xl_Exp_Anomaly_Detection","CPRFL_train"]:
                  
            if args.dataset == "MIMIC":
                from data_provider.data_loader_old import MIMIC
                data_set = MIMIC(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag=flag)
            elif args.dataset == "PTBXL":
                from data_provider.data_loader_old import PTBXL
                data_set = PTBXL(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag=flag, )
            elif args.dataset == "SPH":
                from data_provider.data_loader_old import SPH
                data_set = SPH(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag=flag, )
            elif args.dataset == "G12EC":
                from data_provider.data_loader_old import G12EC
                data_set = G12EC(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag=flag, )
        drop_last = False
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
                  
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name in ['Exp_train_time_llm']:
              
        if args.dataset == "MIMIC":
            from data_provider.data_loader_time_llm import MIMIC
            data_set = MIMIC(
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag, )
        elif args.dataset == "PTBXL":
            from data_provider.data_loader_time_llm import PTBXL
            data_set = PTBXL(
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag, )
        elif args.dataset == "SPH":
            from data_provider.data_loader_time_llm import SPH
            data_set = SPH(
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag, )
        elif args.dataset == "G12EC":
            from data_provider.data_loader_time_llm import G12EC
            data_set = G12EC(
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag, )
        drop_last = False
        print(flag, len(data_set))
        data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                pin_memory=True)
        return data_set, data_loader
    else:
        raise NotImplementedError

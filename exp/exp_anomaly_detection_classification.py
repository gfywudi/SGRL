import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual
from exp.sdtw_cuda_loss import SoftDTW
from scipy import interpolate

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import wandb
import logging

warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from models import TrmEncoder, Timer_my_change
from sklearn.metrics import roc_auc_score,f1_score

      
from huggingface_hub import snapshot_download
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

def calculate_auc_by_sample_count( true_labels, pred_probs,head_threshold=100, tail_threshold=10):

    num_labels = pred_probs.shape[1]

          
    label_sample_counts = np.sum(true_labels, axis=0)        

          
    top_labels = []
    middle_labels = []
    bottom_labels = []

    for label_idx in range(num_labels):
        count = label_sample_counts[label_idx]
        if count > head_threshold:
            top_labels.append(label_idx)
        elif count < tail_threshold:
            bottom_labels.append(label_idx)
        else:
            middle_labels.append(label_idx)

          
    def compute_auc_for_labels(labels):
        aucs = []
        for label in labels:
            auc = roc_auc_score(true_labels[:, label], pred_probs[:, label])
            aucs.append(auc)
        return np.mean(aucs)

    top_auc = compute_auc_for_labels(top_labels)
    middle_auc = compute_auc_for_labels(middle_labels)
    bottom_auc = compute_auc_for_labels(bottom_labels)

    return top_auc, middle_auc, bottom_auc

class FinetuningConfig:



  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  distributed: bool = False
  master_port: str = "12358"
  master_addr: str = "localhost"
  use_wandb: bool = True
  wandb_project: str = "Timeser"


class MetricsLogger(ABC):


  @abstractmethod
  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:

    pass

  @abstractmethod
  def close(self) -> None:
    pass


class WandBLogger(MetricsLogger):


  def __init__(self, project: str, config: Dict[str, Any], rank: int = 0):
    self.rank = rank
    if rank == 0:
      wandb.init(project=project, config=config)

  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:

    wandb.log(metrics, step=step)

  def close(self) -> None:

    if self.rank == 0:
      wandb.finish()

class Exp_Anomaly_Detection_classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_classification, self).__init__(args)
        self.use_wandb: bool = True
        self.wandb_project: str = "Timer_anomaly_detection"      
        finetuning_config = FinetuningConfig()
        self.logger = logging.getLogger(__name__)
        if self.use_wandb:
            self.metrics_logger = WandBLogger(self.wandb_project, finetuning_config.__dict__,
                                            rank=0)

    def get_model(self,load_weights: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo_id = "/.../torch_model.ckpt"
        hparams = TimesFmHparams(
            backend=device,
            per_core_batch_size=32,
            horizon_len=960,
            num_layers=50,
            use_positional_embedding=False,
            context_len=
            960,        
        )
        tfm = TimesFm(hparams=hparams,
                      checkpoint=TimesFmCheckpoint(path=repo_id))
        model = PatchedTimeSeriesDecoder(tfm._model_config)
        if load_weights:
            checkpoint_path = "/...torch_model.ckpt"
                  
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(loaded_checkpoint)
        return model, hparams, tfm._model_config

    def _build_model(self):
        model_name = "timer"
        if model_name =="timesfm":
            model,_,_ = self.get_model(load_weights=True)
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
                  
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def gradient_penalty(self,x, x_recon):
        differences = x_recon - x
        grad = torch.autograd.grad(
            outputs=differences.mean(),
            inputs=x,
            create_graph=True
        )[0]
        return grad.pow(2).mean()

    def mse_distance_matrix(self,X,X_recon):

              
        diff = X.unsqueeze(1) - X_recon.unsqueeze(0)        
        squared_diff = diff ** 2

              
        D = torch.mean(squared_diff, dim=-1)        
        gamma = 1.0
        S = torch.exp(-gamma * D)
        return S

    def supervised_contrastive_loss(self,features,features_recon, labels, temperature=0.1, base_temperature=0.07):
        import torch.nn.functional as F
        batch_size = features.shape[0]

        similarity_matrix = self.mse_distance_matrix(features,features_recon)
              

              
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
              

              
        sim_div_temp = similarity_matrix / temperature
        logits_max, _ = torch.max(sim_div_temp, dim=1, keepdim=True)
        logits = sim_div_temp - logits_max.detach()        
        exp_logits = torch.exp(logits)

              
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

              
        loss = - (temperature / base_temperature) * mean_log_prob_pos.mean()
        return loss


    def _select_criterion(self,reconstructed, reconstructed_12,input,input_12,classification,batch_y,batch_whether):
              
        criterion = torch.nn.HuberLoss(reduction='mean', delta=2.0)
        mse = nn.MSELoss()(input, reconstructed)
        HuberLoss  = criterion(input, reconstructed)
        HuberLoss_12  = criterion(input_12, reconstructed_12)
        error = torch.abs(input - reconstructed)
        exponential_loss = torch.mean(torch.exp(1 * error) - 1)
              
              
        per_sample_mse = torch.mean(torch.abs(input.unsqueeze(dim = 1)* batch_whether.view(-1, 1, 1,1)- reconstructed* batch_whether.view(-1, 1, 1) )+1e-6, dim=[1, 2])        

        per_sample_mse_12 = torch.mean(torch.abs(input_12*batch_whether.view(-1, 1, 1) - reconstructed_12* batch_whether.view(-1, 1, 1)) +1e-6, dim=[1, 2])        


        loss_fn = nn.CrossEntropyLoss()

        weighted_mse = (per_sample_mse * batch_whether).mean()        
              

        weighted_mse_12 = (per_sample_mse_12 * batch_whether).mean()        

        return weighted_mse+0.1*mse

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_correct = 0
        total_samples = 0
        all_preds = []        
        all_labels = []        
        all_preds_one= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_whether,batch_x_12) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_12 = batch_x_12.float().to(self.device)
                batch_y = batch_y.float().to(self.device).squeeze()
                batch_whether = batch_whether.float().to(self.device).squeeze()
                if self.args.use_ims:
                          
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                          
                    outputs_all = self.model(batch_x,batch_x_12, None, None, None)
                outputs, outputs_12,classification = outputs_all

                loss = criterion(outputs,outputs_12,batch_x.squeeze(dim=1), batch_x_12.squeeze(dim=1),classification,batch_y,batch_whether)

                pred_probs = torch.sigmoid(classification)
                predictions = (pred_probs >= 0.5).float()
                correct = (predictions == batch_y).all(dim=1).sum().item()
                total_correct += correct
                total_samples += batch_y.size(0)

                total_loss.append(loss.item())

                pred_probs = torch.sigmoid(classification).detach().cpu().numpy()        
                true_labels = batch_y.detach().cpu().numpy()
                all_preds.append(pred_probs)
                all_labels.append(true_labels)
                all_preds_one.append(predictions.detach().cpu().numpy())

        accuracy = total_correct / total_samples
        total_loss = np.average(total_loss)
        all_preds = np.concatenate(all_preds, axis=0)        
        all_labels = np.concatenate(all_labels, axis=0)        
        all_preds_one = np.concatenate(all_preds_one, axis=0)        
              
        macro_auc = roc_auc_score(all_labels, all_preds, average="macro", multi_class="ovr")
        macro_f1  = f1_score(all_labels, all_preds_one, average='macro')
        top_auc, middle_auc, bottom_auc = calculate_auc_by_sample_count(all_labels, all_preds_one)

              
        return total_loss,accuracy,macro_auc,macro_f1,top_auc, middle_auc, bottom_auc

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
              
        test_data, test_loader = self._get_data(flag='valid')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

              

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion
        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        self.logger.info(
            f"Starting training for {self.args.train_epochs} epochs...")
        self.logger.info(f"Training samples: {train_steps}")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
                  
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x,batch_y,batch_whether,batch_x_12 = batch
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_12 = batch_x_12.float().to(self.device)
                batch_y = batch_y.float().to(self.device).squeeze()
                batch_whether = batch_whether.float().to(self.device)
                if self.args.use_ims:
                          
                    outputs_all = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None,batch_y)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                          
                    outputs_all = self.model(batch_x,batch_x_12, None, None, None)
                outputs,outputs_12,classification= outputs_all
                loss = criterion(outputs,outputs_12,batch_x.squeeze(dim=1), batch_x_12.squeeze(dim=1),classification,batch_y,batch_whether)


                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())


            train_loss = np.average(train_loss)
                  
            if self.args.train_test:
                test_loss,acc, macro_auc,macro_f1,top_auc, middle_auc, bottom_auc  = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f} "
                      "Test acc: {4:.7f} Test auc: {5:.7f} Test f1: {6:.7f} "
                      "Test top_auc: {4:.7f} Test middle_auc: {5:.7f} Test bottom_auc: {6:.7f}  ".format(
                    epoch + 1, train_steps, train_loss, test_loss,acc,macro_auc,macro_f1
                ,top_auc, middle_auc, bottom_auc))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))


            metrics = {
                "train_loss": train_loss,
                "val_loss": test_loss,
                "val_acc": acc,
                "val_macro_auc": macro_auc,
                "val_f1": macro_f1,
                "learning_rate": model_optim.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }
            if self.use_wandb:
                self.metrics_logger.log_metrics(metrics)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(test_loss)
            history["learning_rate"].append( model_optim.param_groups[0]["lr"])

            self.logger.info(
                    f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}"
                )
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(test_loss, self.model, self.args.save_path)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if self.use_wandb:
            self.metrics_logger.close()
              
        return self.model

    def find_border(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border1_str = parts[-2]
        border2_str = parts[-1]
        if '.' in border2_str:
            border2_str = border2_str[:border2_str.find('.')]

        try:
            border1 = int(border1_str)
            border2 = int(border2_str)
            return border1, border2
        except ValueError:
            return None

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        score_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        border_start = self.find_border_number(self.args.data_path)
        border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        if self.args.use_ims:
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        else:
            rec_token_count = self.args.seq_len // self.args.patch_len

        input_list = []
        output_list = []
        with torch.no_grad():
            for i, batch_x in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                      
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = self.model(batch_x, None, None, None)

                input_list.append(batch_x[0, :, -1].detach().cpu().numpy())
                output_list.append(outputs[0, :, -1].detach().cpu().numpy())
                for j in range(rec_token_count):
                          
                    token_start = j * self.args.patch_len
                    token_end = token_start + self.args.patch_len
                    score = torch.mean(self.anomaly_criterion(batch_x[:, token_start:token_end, :],
                                                              outputs[:, token_start:token_end, :]), dim=-1)
                    score = score.detach().cpu().numpy()
                    score = np.mean(score)
                    score_list.append((token_count, score))
                    token_count += 1

        input = np.concatenate(input_list, axis=0).reshape(-1)
        output = np.concatenate(output_list, axis=0).reshape(-1)
        half_patch_len = self.args.patch_len // 2
        input = input[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        output = output[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        data_path = os.path.join('./test_results/UCR/', setting, self.args.data_path[:self.args.data_path.find('.')])
        file_path = data_path + '.pdf'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        visual(input, output, file_path)

        score_list.sort(key=lambda x: x[1], reverse=True)

        def is_overlap(index):
            start = index * self.args.patch_len + border_start
            end = start + self.args.patch_len
            if border1 <= start <= border2 or border1 <= end <= border2 or start <= border1 and end >= border2:
                return True
            else:
                return False

              
        topk = 0
        for i, (index, score) in enumerate(score_list):
            if is_overlap(index):
                topk = i
                break
        print("score_list: ", score_list)
        print('topk:', topk + 1)
        filename = 'ucr768_' + self.args.model + '.csv'
        results = [self.args.data_path, topk + 1, len(score_list)] + score_list
        with open(filename, 'a') as f:
            f.write(','.join([str(result) for result in results]) + '\n')

        return

    def Scale(self,ecg_1d,target_length):
        ecg_flat = ecg_1d.squeeze()
              
        x_original = np.linspace(0, 1, len(ecg_flat))
        x_target = np.linspace(0, 1, target_length)
              
              
        f = interpolate.interp1d(
            x_original,
            ecg_flat,
            kind=self.interp_method,
            fill_value='cubic'
        )
        scaled = f(x_target)
              
        return np.expand_dims(scaled, axis=0)

    def cat_12lead(self, patients_ECG_head_and_tail,predictions_vals):      
        keys = list(patients_ECG_head_and_tail.keys())
        ECG_12_lead = []
        for i in range(12):
            list = []
            for j in range(len(predictions_vals[:,0])):      
                key_i = keys[i]
                value_i = patients_ECG_head_and_tail[key_i]
                target_length = value_i[1]-value_i[0]
                scale_ECG = self.Scale(predictions_vals[j,120*i+0:120*i+96],target_length)
                list.append(scale_ECG)      
            expanded_result = np.concatenate(list, axis=0)
            ECG_12_lead.append(np.unsqueeze(expanded_result, axis=1))
        ECG_12_lead_array = np.concatenate(ECG_12_lead, axis=1)
        return ECG_12_lead_array



    def plot(self, setting):
        import matplotlib.pyplot as plt
        self.model =  self.model.to("cpu")
        self.model.eval()
        test_data, test_loader = self._get_data(flag='test')
        self.Y_train = np.load('/.../Y_origin.npy', allow_pickle=True)
        self.test_index = np.load("/.../"+str(9 + 1) + "fold_index.npy")
        self.Y_test =  self.Y_train[self.test_index]


        for id in range(200):
            x_context,patients_ECG_head_and_tail,y,_,batch_x_12 = test_data[id]
            y_test_name =  self.Y_test[id]
            x_context = torch.from_numpy(x_context)       
            batch_x_12 = torch.from_numpy(batch_x_12)        
            batch_x_12 = batch_x_12.expand(x_context.size(0), -1, -1)
            y = torch.from_numpy(y)
                  

            with torch.no_grad():
                predictions,outputs_12,_ = self.model(x_context.float(),batch_x_12.float(), None, None, None)
            if x_context.shape[0] > 1:
                fig, axes = plt.subplots(x_context.shape[0], 1, figsize=(30, 30))        
                fig.suptitle(f"TimesFM Predictions vs Ground Truth ({y_test_name})", y=1.05)        
                for i in range(x_context.shape[0]):
                    ax = axes[i]
                    context_vals = x_context[i, :, 0].cpu().numpy()
                    predictions_vals = predictions[i, :, 0].cpu().numpy()
                          
                          
                    abs_diff = (context_vals - predictions_vals) ** 2
                          
                    threshold = 0.5        
                    diff_mask = abs_diff > threshold
                          
                    ax.plot(context_vals, label="True", color="blue", linewidth=1)
                    ax.plot(predictions_vals, label="Pred", color="green", linestyle="--", linewidth=1)
                    ax2 = ax.twinx()
                    ax2.plot(abs_diff, label="error", color="pink", linewidth=1)
                    ax.fill_between(np.linspace(0, 1440, 1440), abs_diff, color='pink', alpha=0.4,
                                    label='Area under curve')
                    ax.set_title(f"number_beat {str(i)} disease ({y_test_name})")
                    ax.set_xlabel("Time Step")
                    if i == 0:        
                        ax.set_ylabel("Value")
                    ax.grid(True)
            else:
                if x_context.shape[2] == 12:
                    fig, axes = plt.subplots(x_context.shape[2], 1, figsize=(30, 30))        
                    fig.suptitle(f"TimesFM Predictions vs Ground Truth ({y_test_name})", y=1.05)        
                    lead = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2', 'V3' ,'V4', 'V5' ,'V6']
                elif x_context.shape[2] == 1:
                    fig, axes = plt.subplots(1, 1, figsize=(30, 4))        
                    fig.suptitle(f"TimesFM Predictions vs Ground Truth ({y_test_name})", y=1.05)        
                    lead = ["1"]
                for i in range(x_context.shape[2]):
                    if x_context.shape[2] > 1:
                        ax = axes[i]
                    elif x_context.shape[2] == 1:
                        ax = axes
                    context_vals = x_context[0,:,i].cpu().numpy()
                    predictions_vals = predictions[0,:,i].cpu().numpy()
                    abs_diff = (context_vals - predictions_vals) ** 2
                          
                    threshold = 0.5        
                    diff_mask = abs_diff > threshold
                          

                    ax.plot(context_vals, label="True", color="blue", linewidth=1)
                    ax.plot(predictions_vals, label="Pred", color="green", linestyle="--", linewidth=1)
                    ax2 = ax.twinx()
                    ax2.plot(abs_diff, label="error", color="pink", linewidth=1)
                    ax.fill_between(np.linspace(0, 1440, 1440),abs_diff, color='pink', alpha=0.4, label='Area under curve')
                          
                    ax.set_title(f"lead {lead[i]} disease ({y_test_name})")
                    ax.set_xlabel("Time Step")
                    if i == 0:        
                        ax.set_ylabel("Value")
                    ax.grid(True)

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            safe_name = [s.replace('/', '*') for s in y_test_name]
            save_path = path+"/"+str(safe_name)+str(id)+"predictions.png"
            plt.show()
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

    def plot_12(self, setting):
        import matplotlib.pyplot as plt
        self.model =  self.model.to("cpu")
        self.model.eval()
        test_data, test_loader = self._get_data(flag='test')
        self.Y_train = np.load('/.../Y_origin.npy', allow_pickle=True)
        self.test_index = np.load("/.../"+str(9 + 1) + "fold_index.npy")
        self.Y_test =  self.Y_train[self.test_index]


        for id in range(200):
            x_context,y,_,batch_x_12 = test_data[id]
            y_test_name =  self.Y_test[id]
                  
            print("y_test_name", self.Y_test.shape)

            x_context = torch.from_numpy(x_context)       
            print("x_context",x_context.shape)
            batch_x_12 = torch.from_numpy(batch_x_12)
            print("x_context",batch_x_12.shape)

                  

            with torch.no_grad():
                predictions,outputs_12,classification = self.model(x_context.float(),batch_x_12.float(), None, None, None)
            if batch_x_12.shape[2] > 1:
                fig, axes = plt.subplots(batch_x_12.shape[2] , 1, figsize=(30, 30))        
                fig.suptitle(f"TimesFM Predictions vs Ground Truth ({y_test_name})", y=1.05)        
                lead = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2', 'V3' ,'V4', 'V5' ,'V6']
            elif batch_x_12.shape[2] == 1:
                fig, axes = plt.subplots(1, 1, figsize=(30, 4))        
                fig.suptitle(f"TimesFM Predictions vs Ground Truth ({y_test_name})", y=1.05)        
                lead = ["1"]
            for i in range(batch_x_12.shape[2]):
                if batch_x_12.shape[2] > 1:
                    ax = axes[i]
                elif batch_x_12.shape[2] == 1:
                    ax = axes
                context_vals = batch_x_12[0,:,i].cpu().numpy()
                predictions_vals = outputs_12[0,:,i].cpu().numpy()

                      
                abs_diff = np.abs(context_vals - predictions_vals)
                      
                threshold = 0.5        
                diff_mask = abs_diff > threshold

                ax.plot(context_vals, label="True", color="blue", linewidth=1)
                ax.plot(predictions_vals, label="Pred", color="green", linestyle="--", linewidth=1)
                ax2 = ax.twinx()
                ax2.plot(abs_diff, label="error", color="pink", linewidth=1)
                ax.fill_between(np.linspace(0, 767, 768), abs_diff, color='pink', alpha=0.4, label='Area under curve')
                ax.set_title(f"lead {lead[i]} disease ({y_test_name})")
                ax.set_xlabel("Time Step")
                if i == 0:        
                    ax.set_ylabel("Value")
                ax.grid(True)

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            safe_name = [s.replace('/', '*') for s in y_test_name]
            save_path = path+"/"+str(safe_name)+str(id)+"predictions_12lead.png"
            plt.show()
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")


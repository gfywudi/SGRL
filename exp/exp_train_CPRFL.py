import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score
from tqdm import tqdm

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
import json

warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from exp.exp_tool_loss import FocalLoss,AsymmetricLoss
from models import TrmEncoder, Timer_my_change




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

class Exp_train_CPRFL(Exp_Basic):
    def __init__(self, args):
        super(Exp_train_CPRFL, self).__init__(args)
        self.use_wandb: bool = True
        self.wandb_project: str = "Timer_anomaly_detection"
        finetuning_config = FinetuningConfig()
        self.logger = logging.getLogger(__name__)
        if self.use_wandb:
            self.metrics_logger = WandBLogger(self.wandb_project, finetuning_config.__dict__,
                                            rank=0)

        if self.args.dataset == "MIMIC":
            with open(
                    '/.../label_vocab.json',
                    'r', encoding='utf-8') as file:
                text_list = json.load(file)

        elif self.args.dataset =="PTBXL":
            self.label_list = np.loadtxt('/..../Y_unique_list.txt',
                                         delimiter=',', dtype=str)
            with open('/..../CKEPE_prompt.json', 'r', encoding='utf-8') as file:
                data_dict = json.load(file)
            text_list = []
            result = [s.replace('_', '') for s in self.label_list]
            for i in result:
                a =data_dict[i]
                text_list.append(a)

        elif self.args.dataset =="SPH":
            with open('/..../Y_unique_list.json', 'r', encoding='utf-8') as file:
                self.label_list = json.load(file)


    def _build_model(self):
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

    def _select_criterion(self,reconstructed,input,batch_y,batch_whether):

              
        fn = nn.BCEWithLogitsLoss()
        loss_fn = fn(reconstructed, batch_y)
        return loss_fn
              
              

    def get_pred_labels_with_best_threshold(self, pred_probs, true_labels, thresholds=np.arange(0.0, 1.1, 0.01)):

        num_labels = pred_probs.shape[1]
        best_thresholds = np.zeros(num_labels)        
        pred_labels = np.zeros_like(pred_probs)        
              
        for i in range(num_labels):
            best_f1 = 0
            best_threshold = 0

                  
            for threshold in thresholds:
                      
                temp_pred_labels = (pred_probs[:, i] >= threshold).astype(int)
                      
                f1 = f1_score(true_labels[:, i], temp_pred_labels)

                      
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

                  
            pred_labels[:, i] = (pred_probs[:, i] >= best_threshold).astype(int)
            best_thresholds[i] = best_threshold

                  
                  

        return pred_labels, best_thresholds

    def calculate_auc_by_sample_count_MIMIC(self, true_labels, pred_probs):

        num_labels = pred_probs.shape[1]

              
        label_sample_counts = np.sum(true_labels, axis=0)        

              
        sorted_labels = np.argsort(label_sample_counts)        

              
        total_labels = len(label_sample_counts)
        lower_bound = total_labels // 4        
        upper_bound = 3 * total_labels // 4        

              
        bottom_labels = sorted_labels[:lower_bound]        
        middle_labels = sorted_labels[lower_bound:upper_bound]        
        top_labels = sorted_labels[upper_bound:]        

        pred_labels,best_thresholds = self.get_pred_labels_with_best_threshold(pred_probs, true_labels, thresholds=np.arange(0.0, 1.1, 0.01))

              
        def compute_auc_for_labels(labels):
            aucs = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            metrics_dict = {}
            for label in labels:
                auc = roc_auc_score(true_labels[:, label], pred_probs[:, label])
                accuracy = accuracy_score(true_labels[:, label], pred_labels[:, label])
                precision = precision_score(true_labels[:, label], pred_labels[:, label])
                recall = recall_score(true_labels[:, label], pred_labels[:, label])
                f1 = f1_score(true_labels[:, label], pred_labels[:, label])
                thred = best_thresholds[label]
                count = label_sample_counts[label]        
                metrics_dict[label] = {
                    "AUC": auc,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "f1": f1,
                    "Sample Count": count,
                    "best_thresholds": thred
                }
                aucs.append(auc)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            macro_auc = np.mean(aucs)
            macro_accuracy = np.mean(accuracies)
            macro_precision = np.mean(precisions)
            macro_recall = np.mean(recalls)
            macro_f1 = np.mean(f1_scores)
            return metrics_dict, macro_auc, macro_accuracy, macro_precision, macro_recall, macro_f1

        top_metrics, top_macro_auc, top_macro_accuracy, top_macro_precision, top_macro_recall, top_macro_f1 = compute_auc_for_labels(
            top_labels)
              
        middle_metrics, middle_macro_auc, middle_macro_accuracy, middle_macro_precision, middle_macro_recall, middle_macro_f1 = compute_auc_for_labels(
            middle_labels)
              
        bottom_metrics, bottom_macro_auc, bottom_macro_accuracy, bottom_macro_precision, bottom_macro_recall, bottom_macro_f1 = compute_auc_for_labels(
            bottom_labels)
        all_labels = sorted_labels
        all_metrics, all_macro_auc, all_macro_accuracy, all_macro_precision, all_macro_recall, all_macro_f1 = compute_auc_for_labels(
            all_labels)
        return {
            "Top": {
                "Metrics": top_metrics,
                "Macro AUC": top_macro_auc,
                "Macro Accuracy": top_macro_accuracy,
                "Macro Precision": top_macro_precision,
                "Macro Recall": top_macro_recall,
                "Macro F1": top_macro_f1
            },
            "Middle": {
                "Metrics": middle_metrics,
                "Macro AUC": middle_macro_auc,
                "Macro Accuracy": middle_macro_accuracy,
                "Macro Precision": middle_macro_precision,
                "Macro Recall": middle_macro_recall,
                "Macro F1": middle_macro_f1
            },
            "Bottom": {
                "Metrics": bottom_metrics,
                "Macro AUC": bottom_macro_auc,
                "Macro Accuracy": bottom_macro_accuracy,
                "Macro Precision": bottom_macro_precision,
                "Macro Recall": bottom_macro_recall,
                "Macro F1": bottom_macro_f1
            },
            "All": {
                "Metrics": all_metrics,
                "Macro AUC": all_macro_auc,
                "Macro Accuracy": all_macro_accuracy,
                "Macro Precision": all_macro_precision,
                "Macro Recall": all_macro_recall,
                "Macro F1": all_macro_f1
            }
        }

    def calculate_auc_by_sample_count(self, train_labels, true_labels, pred_probs, head_threshold=1000,
                                      tail_threshold=100):

        num_labels = pred_probs.shape[1]

              
        label_sample_counts = np.sum(train_labels, axis=0)        

              
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

        pred_labels, best_thresholds = self.get_pred_labels_with_best_threshold(pred_probs, true_labels,
                                                                                thresholds=np.arange(0.0, 1.1, 0.01))

              
        def compute_auc_for_labels(labels):
            aucs = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            metrics_dict = {}
            for label in labels:
                auc = roc_auc_score(true_labels[:, label], pred_probs[:, label])
                accuracy = accuracy_score(true_labels[:, label], pred_labels[:, label])
                precision = precision_score(true_labels[:, label], pred_labels[:, label])
                recall = recall_score(true_labels[:, label], pred_labels[:, label])
                f1 = f1_score(true_labels[:, label], pred_labels[:, label])
                count = label_sample_counts[label]        
                thred = best_thresholds[label]
                metrics_dict[self.label_list[label]] = {
                    "AUC": auc,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "f1": f1,
                    "Sample Count": count,
                    "best_thresholds": thred
                }
                aucs.append(auc)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            macro_auc = np.mean(aucs)
            macro_accuracy = np.mean(accuracies)
            macro_precision = np.mean(precisions)
            macro_recall = np.mean(recalls)
            macro_f1 = np.mean(f1_scores)
            return metrics_dict, macro_auc, macro_accuracy, macro_precision, macro_recall, macro_f1

        top_metrics, top_macro_auc, top_macro_accuracy, top_macro_precision, top_macro_recall, top_macro_f1 = compute_auc_for_labels(
            top_labels)
              
        middle_metrics, middle_macro_auc, middle_macro_accuracy, middle_macro_precision, middle_macro_recall, middle_macro_f1 = compute_auc_for_labels(
            middle_labels)
              
        bottom_metrics, bottom_macro_auc, bottom_macro_accuracy, bottom_macro_precision, bottom_macro_recall, bottom_macro_f1 = compute_auc_for_labels(
            bottom_labels)
        all_labels = top_labels + middle_labels + bottom_labels
        all_metrics, all_macro_auc, all_macro_accuracy, all_macro_precision, all_macro_recall, all_macro_f1 = compute_auc_for_labels(
            all_labels)
        return {
            "Top": {
                "Metrics": top_metrics,
                "Macro AUC": top_macro_auc,
                "Macro Accuracy": top_macro_accuracy,
                "Macro Precision": top_macro_precision,
                "Macro Recall": top_macro_recall,
                "Macro F1": top_macro_f1
            },
            "Middle": {
                "Metrics": middle_metrics,
                "Macro AUC": middle_macro_auc,
                "Macro Accuracy": middle_macro_accuracy,
                "Macro Precision": middle_macro_precision,
                "Macro Recall": middle_macro_recall,
                "Macro F1": middle_macro_f1
            },
            "Bottom": {
                "Metrics": bottom_metrics,
                "Macro AUC": bottom_macro_auc,
                "Macro Accuracy": bottom_macro_accuracy,
                "Macro Precision": bottom_macro_precision,
                "Macro Recall": bottom_macro_recall,
                "Macro F1": bottom_macro_f1
            },
            "All": {
                "Metrics": all_metrics,
                "Macro AUC": all_macro_auc,
                "Macro Accuracy": all_macro_accuracy,
                "Macro Precision": all_macro_precision,
                "Macro Recall": all_macro_recall,
                "Macro F1": all_macro_f1
            }
        }

    def vali(self, all_train_label, vali_loader, criterion):
        total_loss = []
        total_loss = []
        total_correct = 0
        total_samples = 0
        all_preds = []        
        all_labels = []        
        all_preds_one = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_whether) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_whether = batch_whether.float().to(self.device)
                if self.args.use_ims:
                          
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                          
                    outputs,_ = self.model(batch_x)

                loss = criterion(outputs, batch_x,batch_y,batch_whether)
                total_loss.append(loss.item())

                classification = outputs
                pred_probs = torch.sigmoid(classification)
                predictions = (pred_probs >= 0.5).float()
                correct = (predictions == batch_y).all(dim=1).sum().item()
                total_correct += correct
                total_samples += batch_y.size(0)

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
                  
                  

            all_preds = np.nan_to_num(all_preds, nan=0.0)
            all_labels = np.nan_to_num(all_labels, nan=0.0)
                  
            macro_auc = roc_auc_score(all_labels, all_preds, average="macro", multi_class="ovr")
            macro_f1 = f1_score(all_labels, all_preds_one, average='macro')

            if self.args.dataset =="MIMIC":
                all_dict = self.calculate_auc_by_sample_count_MIMIC(all_labels, all_preds)
            elif self.args.dataset == "SPH":
                all_dict = self.calculate_auc_by_sample_count(all_train_label,all_labels, all_preds,head_threshold=1000, tail_threshold=50)
            elif self.args.dataset == "PTBXL":
                all_dict = self.calculate_auc_by_sample_count(all_train_label,all_labels, all_preds,head_threshold=1000, tail_threshold=100)

              
        total_loss = np.average(total_loss)
        return total_loss, accuracy, all_dict["All"]["Macro AUC"], all_dict["All"]["Macro F1"], all_dict["Top"]["Macro AUC"],all_dict["Middle"]["Macro AUC"], all_dict["Bottom"]["Macro AUC"],all_labels, all_preds,all_dict
              

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
              
        test_data, test_loader = self._get_data(flag='valid')
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion
        history = {"train_loss": [], "val_loss": [], "learning_rate": []}
        self.logger.info(
            f"Starting training for {self.args.train_epochs} epochs...")
        self.logger.info(f"Training samples: {train_steps}")
        for epoch in range(self.args.train_epochs):
            all_labels = []
            iter_count = 0
            train_loss = []
                  
            epoch_time = time.time()
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            scaler = torch.cuda.amp.GradScaler()
            for i, (batch_x, batch_y, batch_whether) in loop:
                  
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_whether = batch_whether.float().to(self.device)
                      
                with torch.cuda.amp.autocast():
                    outputs,_ = self.model(batch_x)
                    true_labels = batch_y.detach().cpu().numpy()
                    loss = criterion(outputs, batch_x,batch_y,batch_whether)
                all_labels.append(true_labels)

                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
                      
                train_loss.append(loss.item())

            all_labels = np.concatenate(all_labels, axis=0)
            train_loss = np.average(train_loss)
            if self.args.train_test:
                      
                test_loss, acc, macro_auc, macro_f1,top_auc, middle_auc, bottom_auc,all_labels, all_preds, all_dict = self.vali(all_labels, test_loader, criterion)
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}"
                    " Test acc: {4:.7f} Test auc: {5:.7f} Test f1: {6:.7f} "
                    "Test top_auc: {7:.7f} Test middle_auc: {8:.7f} Test bottom_auc: {9:.7f}"
                    "".format(
                        epoch + 1, train_steps, train_loss, test_loss, acc, macro_auc, macro_f1,
                    top_auc, middle_auc, bottom_auc))

            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))


            metrics = {
                "train_loss": train_loss,
                "val_loss": test_loss,
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

            early_stopping(macro_auc, self.model, self.args.save_path,all_labels, all_preds,all_dict)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if early_stopping.early_stop:
                print("Early stopping")
                break


        if self.use_wandb:
            self.metrics_logger.close()
              
        return self.model






    def save_model(self, save_path,epoch,model_optim):
        """
        保存模型为 .ckpt 格式（兼容 PyTorch Lightning 的检查点格式）
        :param save_path: 保存路径，应以 .ckpt 结尾
        """
        if not save_path.endswith('.ckpt'):
            raise ValueError("保存路径应以 .ckpt 结尾")

        checkpoint = {
            'epoch': epoch,        
                  
            'state_dict': {k: v for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': model_optim.state_dict() if hasattr(self, 'optimizer') else None,
            'loss': self.best_loss if hasattr(self, 'best_loss') else None,
                  
        }
        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")
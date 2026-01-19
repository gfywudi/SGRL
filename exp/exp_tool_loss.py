import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy



try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            temperature=0.07
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temperature = temperature
              
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T        
        logits_per_text = logit_scale * text_features @ image_features.T        

              
        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)        
        pred_1 = F.log_softmax(logits_per_image, dim=-1)        
        pred_2 = F.log_softmax(logits_per_text, dim=-1)
        loss_a = F.kl_div(pred_1, labels, reduction='sum') / num_logits        
        loss_b = F.kl_div(pred_2, labels, reduction='sum') / num_logits
        total_loss = (loss_a + loss_b) / 2
        return total_loss


class UniCL(nn.Module):
    def __init__(self,
                 local_loss=False,
                 gather_with_grad=False,
                 cache_labels=False,
                 rank=0,
                 world_size=1,
                 use_horovod=False,
                 temperature=0.05,
                 uniCl_type="increase_dimension"):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temperature = temperature
        self.uniCl_type = uniCl_type
              
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, labels):        
        device = image_features.device
        batch_size, label_nums = labels.shape

        y = torch.ones((labels.shape[1], labels.shape[0], labels.shape[0]), device=device)        
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 0:
                    y[j, i, :] = 0
                    y[j, :, i] = 0
        logits_per_image = image_features @ text_features.T        
        logits_per_text = text_features @ image_features.T        
        num_logits = logits_per_image.shape[1]

        if self.uniCl_type == "increase_dimension":        
            logit_scale = nn.Parameter(F.normalize(torch.ones([label_nums]), p=2, dim=0) * np.log(1 / self.temperature))
                  
                  

            logits_per_image = logits_per_image.unsqueeze(0).repeat(label_nums, 1, 1)        
            logits_per_text = logits_per_text.unsqueeze(0).repeat(label_nums, 1, 1)        
            for i in range(label_nums):
                logits_per_image[i] = logit_scale[i] * logits_per_image[i].clone()
                logits_per_text[i] = logit_scale[i] * logits_per_text[i].clone()

            total_loss = torch.tensor(0.0, device=device)
            for image, text, label in zip(logits_per_image, logits_per_text, y):
                pred_1 = F.log_softmax(image, dim=-1)
                pred_2 = F.log_softmax(text, dim=-1)
                loss_a = F.kl_div(pred_1, label, reduction='sum') / num_logits
                loss_b = F.kl_div(pred_2, label, reduction='sum') / num_logits
                loss = (loss_a + loss_b) / 2
                total_loss = torch.add(total_loss, loss)

        else:        
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
            logits_per_image = logit_scale * logits_per_image
            logits_per_text = logit_scale * logits_per_text
            uni_labels = torch.max(y, dim=0).values
            pred_1 = F.log_softmax(logits_per_image, dim=-1)        
            pred_2 = F.log_softmax(logits_per_text, dim=-1)
            loss_a = F.kl_div(pred_1, uni_labels, reduction='sum') / num_logits        
            loss_b = F.kl_div(pred_2, uni_labels, reduction='sum') / num_logits
            total_loss = (loss_a + loss_b) / 2
        return total_loss / label_nums

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def clip_loss(x, y, temperature=0.07, device=None):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels)
    loss_i = F.cross_entropy(sim.T, labels)

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss实现
        :param alpha: 平衡因子，默认为 0.25
        :param gamma: 调节因子，默认为 2.0
        :param reduction: 损失的计算方式，'mean' 或 'sum'，默认为 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的预测值，形状为 (N, C)，其中 N 是样本数量，C 是类别数
        :param targets: 真实标签，已经是 one-hot 编码，形状为 (N, C)
        :return: 计算得到的 Focal Loss
        """
              
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

              
        p_t = (inputs * targets).sum(dim=1)

              
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

              
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class AsymmetricLoss(nn.Module):      
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

              
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

              
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

              
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

              
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)        
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
              
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device        
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def label_decorrelation_loss(H_sem, y, eps=1e-5):
          
          
    B, C, D = H_sem.shape

    present = (y.sum(dim=0) > 0)              
    H = H_sem[:, present, :]                  
    C_ = H.shape[1]
    if C_ <= 1:
        return H_sem.new_tensor(0.0)

          
    mean = H.mean(dim=0, keepdim=True)        
    var  = H.var(dim=0, unbiased=False, keepdim=True)
    Hn   = (H - mean) / torch.sqrt(var + eps)         

          
          
    Hn = Hn.permute(1, 0, 2).reshape(C_, B*D)         
    corr = (Hn @ Hn.t()) / (B*D)                      

          
    off = corr - torch.eye(C_, device=corr.device)
    loss = (off**2).sum() / (C_*(C_-1))
    return loss


if __name__ == '__main__':
          
          
          
    test = UniCL()
    result = test(torch.randn((3, 5)), torch.randn((3, 5)), torch.randint(0, 2, (3, 4)))




from math import sqrt
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.TF import Transformer
from models.MSDNN_use_to_cat import MSDNN
from model_to_cat.Autoformer_use_to_cat import Autoformer_use_to_cat
from model_to_cat.densenet_use_to_cat import DenseNet
from model_to_cat.resnet_use_to_cat import ECGResNet

from torch.nn.functional import normalize
import json
import numpy as np



class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
  
    def __init__(self, configs):
        super(Model, self).__init__()

        self.no_feature_grad = False

        self.patch_len = configs.input_token_len
        self.stride = configs.stride
        self.backbone_model_name = configs.backbone
        self.weather_use_pretrain_model = configs.weather_use_pretrain_model
        self.pretrain_dataset = configs.pretrain_dataset
        self.dataset = configs.dataset
        if configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{configs.gpu}')
        else:
            self.device = torch.device('cpu')

        self.d = configs.d

        self.gap = nn.AdaptiveAvgPool1d(1)
        self._get_model_and_tokenizer(configs.llm_model, configs.llm_layers)


        ctx_vectors = nn.Parameter(torch.empty(configs.num_classes, self.d).float().to(self.device))
        vocab_size = len(self.tokenizer)
        nn.init.normal_(ctx_vectors, mean=0, std=1)
        ctx_vectors_ = ctx_vectors.round().clamp(min=0, max=vocab_size - 1).long()

        self.ctx_vectors = nn.Parameter(ctx_vectors_.float())   
        #
         
         
         
 

        self.semantic_weight  = nn.Parameter(torch.ones(configs.num_classes))
        self.ecg_weight = torch.ones(1, device=self.device)



        if configs.llm_model == 'LLAMA':
            self.d_llm = 4096
        elif configs.llm_model == 'GPT2':
            self.d_llm = 768
        elif configs.llm_model == 'BERT':
            self.d_llm = 768
        else:
            raise Exception('LLM model is not defined')

         
         
         
         
         
         
         

        self.proj_t = nn.Sequential(
            nn.Linear(self.d_llm, 1024),
            nn.GELU(),
            nn.Linear(1024, self.d),
        )

        self.fc = nn.Linear(self.d, configs.num_classes)   
        self.fc_xiaorong = nn.Linear(13568, configs.num_classes)   
        self.fc_mix =  nn.Linear(configs.num_classes*2, configs.num_classes)

        self.modulation_net = nn.Sequential(
            nn.Linear(256, 256 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(256 // 4, 256),
            nn.Sigmoid()   
        )

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        self.top_k = 5   


         
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, self.stride,
                                              configs.dropout)

         
        self.word_embeddings = self.llm_model.get_input_embeddings().weight   
        self.vocab_size = self.word_embeddings.shape[0]


        self.attention =  nn.Sequential(
        Transformer(
            dim=256,
            depth=1,
            heads=4,
            dim_head=64,
            mlp_dim=128,
            dropout=0.5
        ))

        self.input_ids, self.attention_mask = self.All_label_text_get()

        if self.backbone_model_name == "MSDNN":
            self.proj_e_no_llm = nn.Sequential(
                nn.Linear(176, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.d),
                nn.BatchNorm1d(self.d),
            )
            self.preMSDNN = MSDNN()
            if self.weather_use_pretrain_model:
                if self.pretrain_dataset == "PTBXL":
                    self.ckpt_path = "/.../result/long_tail/model/ECG_no_pretrain_MSDNN_LLM+_onlyCLIP_PTBXL_44class/model_2025-10-15:15.pth"
                elif self.pretrain_dataset == "SPH":
                    self.ckpt_path = "/.../result/long_tail/model/ECG_no_pretrain_MSDNN_LLM+_onlyCLIP_SPH_44class/model_2025-10-20:15.pth"
                elif self.pretrain_dataset == "MIMIC":
                    self.ckpt_path = "/.../result/long_tail/model/ECG_no_pretrain_MSDNN_LLM+_onlyCLIP_MIMIC_102class/model_2025-10-21:00.pth"
                elif self.pretrain_dataset == "G12EC":
                    self.ckpt_path = "/.../result/long_tail/model/ECG_have_pretrain_MSDNN_LLM+_onlyCLIP_G12EC_26class/model_2025-11-06:19.pth"
            else:
                self.ckpt_path = ""
            if self.ckpt_path != '':
                if self.ckpt_path == 'random':
                    print('loading model randomly')
                else:
                    print('loading model: ', self.ckpt_path)
                    if self.ckpt_path.endswith('.pth'):
                        sd = torch.load(self.ckpt_path, map_location="cpu")
                        self.preMSDNN.load_state_dict(sd, strict=False)
                        self.proj_e_no_llm.load_state_dict(sd, strict=False)
                        print("load proj_e_no_llm")
                        self.proj_t.load_state_dict(sd, strict=False)
                        print("load proj_t")
                    elif self.ckpt_path.endswith('.ckpt'):
                        sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
                        self.preMSDNN.load_state_dict(sd, strict=False)
                    else:
                        raise NotImplementedError

        elif self.backbone_model_name == "Autoformer":
            self.preMSDNN = Autoformer_use_to_cat(configs)

        elif self.backbone_model_name == "Densenet":
            self.proj_e_no_llm = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.d),
                nn.BatchNorm1d(self.d),
            )
            self.preMSDNN = DenseNet()

            if self.weather_use_pretrain_model:
                if self.pretrain_dataset == "PTBXL":
                    self.ckpt_path = "/.../result/long_tail/model/ECG_no_pretrain_Densenet_LLM+_onlyCLIP_PTBXL_44class/model_2025-10-23:21.pth"
                elif self.pretrain_dataset == "SPH":
                    self.ckpt_path = ""
                elif self.pretrain_dataset == "MIMIC":
                    self.ckpt_path = ""
            else:
                self.ckpt_path = ""

            if self.ckpt_path != '':
                if self.ckpt_path == 'random':
                    print('loading model randomly')
                else:
                    print('loading model: ', self.ckpt_path)
                    if self.ckpt_path.endswith('.pth'):
                        sd = torch.load(self.ckpt_path, map_location="cpu")
                        self.preMSDNN.load_state_dict(sd, strict=False)
                        if 'proj_e_no_llm' in sd:
                            self.proj_e_no_llm.load_state_dict(sd['proj_e_no_llm'], strict=False)
                    elif self.ckpt_path.endswith('.ckpt'):
                        sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
                        self.preMSDNN.load_state_dict(sd, strict=False)
                    else:
                        raise NotImplementedError

        elif self.backbone_model_name == "resnet":
            self.proj_e_no_llm = nn.Sequential(
                nn.Linear(744, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.d),
                nn.BatchNorm1d(self.d),
            )
            self.preMSDNN = ECGResNet()
            self.ckpt_path = ""

    def _get_model_and_tokenizer(self, model_name, layers):
        print("> loading model: ", model_name)
        local_path = "/.../large_model/"
        if model_name == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained(local_path + 'huggyllama-llama-7b')
            self.llama_config.num_hidden_layers = layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(local_path + 'huggyllama-llama-7b', config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained(local_path + 'huggyllama-llama-7b')
        elif model_name == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(local_path + 'openai-community-gpt2')
            self.gpt2_config.num_hidden_layers = layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(local_path + 'openai-community-gpt2', config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(local_path + 'openai-community-gpt2')
        elif model_name == 'BERT':
            self.bert_config = BertConfig.from_pretrained(local_path + 'bert-base-uncased')
            self.bert_config.num_hidden_layers = layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            self.llm_model = BertModel.from_pretrained(local_path + 'bert-base-uncased',
                                                       config=self.bert_config, attn_implementation="eager")
            self.tokenizer = BertTokenizer.from_pretrained(local_path + 'bert-base-uncased')
        else:
            raise Exception('LLM model is not defined')
        print("> loading model done")

    def All_label_text_get(self):
        if self.dataset == "MIMIC":

            with open('/../data/MIMIC/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/label_json_X_and_Y/Y_unique_list.json', 'r', encoding='utf-8') as file:
                self.label_list = json.load(file)
            with open('/.../code/mamba-main/CKEPE_prompt_MIMIC.json', 'r',
                      encoding='utf-8') as file:
                data_dict = json.load(file)
            text_list = []
            result = [s.replace('_', '') for s in self.label_list]
            for i in result:
                a = data_dict[i]
                text_list.append(a)

        elif self.dataset =="PTBXL":
            self.label_list = np.loadtxt('/...../Y_unique_list.txt',
                                         delimiter=',', dtype=str)
            with open('/....../CKEPE_prompt.json', 'r', encoding='utf-8') as file:
                data_dict = json.load(file)
            text_list = []
            result = [s.replace('_', '') for s in self.label_list]
            for i in result:
                a =data_dict[i]
                text_list.append(a)

        elif self.dataset =="SPH":
            with open('/...../Y_unique_list.json', 'r', encoding='utf-8') as file:
                self.label_list = json.load(file)
            with open('/......./CKEPE_prompt.json', 'r',
                      encoding='utf-8') as file:
                data_dict = json.load(file)
            text_list = []
            result = [s.replace('_', '') for s in self.label_list]
            for i in result:
                a = data_dict[i]
                text_list.append(a)

        elif self.dataset == "G12EC":
            self.label_list = np.loadtxt('/../unique_list_filtered.txt',
                                         delimiter=',', dtype=str)
            with open('.../CKEPE_prompt.json', 'r',
                      encoding='utf-8') as file:
                data_dict = json.load(file)
            text_list = []
            result = [s.replace('_', '') for s in self.label_list]
            for i in result:
                a = data_dict[i]
                text_list.append(a)

        self._get_model_and_tokenizer('BERT', 6)
        prompt = self.tokenizer(text_list, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=self.d, return_attention_mask=True)
        input_ids = prompt.input_ids
        attention_mask  = prompt.attention_mask

        return input_ids.to(self.device), attention_mask.to(self.device)


    def ext_ecg_emb(self, ecg):
        ecg_emb = self.preMSDNN(ecg)
        return ecg_emb

    def semantic_consistency_loss(self,PH, PG):
        PH = PH.float()
        PG = PG.float()


        return torch.norm(PH - PG, p=1)   



    def forecast(self, x_enc):
        batch_size = x_enc.size(0)

        enc_out  = self.ext_ecg_emb(x_enc)
        enc_out = torch.squeeze(enc_out)
        dec_out = self.proj_e_no_llm(enc_out)
        feature_maps = dec_out

        ones_attention_mask = torch.ones_like(self.ctx_vectors).to(self.device)
        input_ids = torch.cat([self.ctx_vectors.long(),self.input_ids], dim=1)
        attention_mask = torch.cat([ones_attention_mask, self.attention_mask], dim=1)
        text_features = self.get_text_emb(input_ids,attention_mask)



        sc_loss = self.semantic_consistency_loss(self.input_ids, self.input_ids+self.ctx_vectors.long())



        text_features = text_features.float()
        attr = text_features.unsqueeze(0).expand(batch_size, text_features.size(0), self.d)#torch.Size([128, 44, 256])
        ###
        alpha = self.modulation_net(feature_maps)
        alpha = alpha.unsqueeze(1)
        attr = attr * (1.0 + alpha)

        feature = torch.cat((feature_maps.unsqueeze(1), attr), 1)
         

        feature = self.attention(feature)#([128, 45, 256])

        feature_ = feature.detach()
        decorrelation_feature = feature_[:, -attr.size(1):, :]
         

        classify_feature = feature[:, 0:1, :]#([128, 44, 256])?
        sim = F.cosine_similarity(feature_[:, -attr.size(1):, :], decorrelation_feature, dim=-1)   
         
        score = self.fc(classify_feature.squeeze()+dec_out)
         
        score = score.squeeze()

        return score+sim,decorrelation_feature, sc_loss


    def get_text_emb(self, input_ids, attention_mask):
        if input_ids.dim() > 2:
            input_ids = input_ids.squeeze()
             
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.squeeze()

        text_emb = self.llm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_emb = self.proj_t(text_emb)
         
         
        return text_emb

    def forward(self, x_enc):
        score,decorrelation_feature ,sc_loss = self.forecast(x_enc)
        return score,decorrelation_feature, sc_loss





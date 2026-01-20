import os

import torch

from models import (TrmEncoder, Timer_my_change, Timer_XL, time_llm, MSDNN, time_llm_ECG,
                    Autoformer, Crossformer, ETSformer, Informer, iTransformer, PatchTST, TimesNet
                    , model_CPRFL, FEDformer,TimeMixer,)

from ECG_model import (AGSX,AICTRCD,ECGTransForm,EffNet,CNN,densenet_,resnet,Net_1d)



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TrmEncoder': TrmEncoder,
            'Timer': Timer_my_change,
            "Timer_XL": Timer_XL,
            "Time_LLM": time_llm,
            "MSDNN": MSDNN,
            "time_llm_ECG": time_llm_ECG,
            "Autoformer": Autoformer,
            "Crossformer":Crossformer,
            "ETSformer":ETSformer,
            "Informer":Informer,
            "iTransformer":iTransformer,
            "PatchTST":PatchTST,
            "TimesNet": TimesNet,
            "AGSX":AGSX,
            "AICTRCD":AICTRCD,
            "ECGTransForm":ECGTransForm,
            "EffNet":EffNet,
            "CNN":CNN,
            "densenet":densenet_,
            "resnet":resnet,
            "Net_1d":Net_1d,
            "FEDformer":FEDformer,
            "TimeMixer":TimeMixer,
                  
            "model_CPRFL":model_CPRFL

        }

        if self.args.use_multi_gpu:
            self.model = self._build_model()
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

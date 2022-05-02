from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import datasets

class TrainerWithTrainingLoss(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        train_dataset = np.random.choice(self.train_dataset, size=len(self.train_dataset) // 10, replace=False)
        # train_result = super().evaluate(train_dataset, ignore_keys, 'train')

        return eval_result
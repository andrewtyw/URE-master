import random
import torch
import numpy as np
import transformers


def set_global_random_seed(seed):
    """to fix the random seed

    Args: seed:int
    """
    print("set seed:",seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

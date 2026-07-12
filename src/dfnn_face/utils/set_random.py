# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
import numpy as np
import random


def set_random_seed(random_seed):

    np.seed = random_seed
    np.random.seed(random_seed)
    random.seed = random_seed
    torch.manual_seed(random_seed)
    random_generator = torch.Generator()
    random_generator.manual_seed(random_seed)


import torch
import numpy as np
from math import inf
from PIL import Image

import torch.nn.functional as F

class DependentLabelGenerator:
    def __init__(self, num_classes, feature_size, transform):
        self.W = torch.FloatTensor(np.random.randn(num_classes, feature_size, num_classes))
        self.num_classes = num_classes
        self.transform = transform
    def generate_dependent_labels(self, data, target):
        # 1*m *  m*10 = 1*10
        img = Image.fromarray(data)
        img = self.transform(img)
        A = img.view(1, -1).mm(self.W[target]).squeeze(0)
        A[target] = -inf
        A = F.softmax(A, dim=0)

        new_label = int(np.random.choice(list(range(self.num_classes)), p=A.cpu().numpy()))
        return new_label
import numpy as np
import torch
from torchmetrics import JaccardIndex

##################################################################

def compute(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    if union > 0:
        iou = intersection / union
    else:
        iou = 1
    return {"intersection": intersection, "union": union, "iou": iou}

jaccard = JaccardIndex(num_classes=2, threshold=0.5, average=None, absent_score=-1)

##################################################################


pred = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

mask = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

compute(pred, mask)

compute(1-pred, 1-mask)

jaccard(torch.tensor(pred), torch.tensor(mask))

##################################################################


pred = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

mask = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

compute(pred, mask)

compute(1-pred, 1-mask)

jaccard(torch.tensor(pred), torch.tensor(mask))


##################################################################


pred = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

mask = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

compute(pred, mask)

compute(1-pred, 1-mask)

jaccard(torch.tensor(pred), torch.tensor(mask))

##################################################################


pred = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

mask = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

compute(pred, mask)

compute(1-pred, 1-mask)

jaccard(torch.tensor(pred), torch.tensor(mask))

# if the union is zero, then it has to be an OK image AND prediction is an OK image !

##################################################################



pred = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [0, 0, 1]
])

mask = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
])

compute(pred, mask)

compute(1-pred, 1-mask)

jaccard(torch.tensor(pred), torch.tensor(mask))

#{'intersection': 2, 'union': 4, 'iou': 0.5}

#{'intersection': 5, 'union': 7, 'iou': 0.7142857142857143}

# 5 = 9 - 4
# 7 = 9 - 2

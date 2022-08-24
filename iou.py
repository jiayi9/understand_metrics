import numpy as np

##################################################################

def compute(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    if union > 0:
        iou = intersection / union
    else:
        iou = 1
    return {"intersection": intersection, "union": union, "iou": iou}


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

# if the union is zero, then it has to be an OK image AND prediction is an OK image !

##################################################################


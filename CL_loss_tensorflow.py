import torch
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


from typing import Optional, Callable, Tuple, Any
from PIL import Image
import torchvision.models as models

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf


def nt_xent_loss(output1, output2, temperature):
    # concatenate v1 img and v2 img via the rows, stacking vertically
    out = torch.cat([output1, output2], dim=0)
    #print(out.size())
    n_samples = len(out)
    print(n_samples)

    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    # the diagonal of the matrix is the square of each vector element in the out vector, which shows the similarity between the same elements
    cov = torch.mm(out, out.t().contiguous())
    #print(cov.size())
    sim = torch.exp(cov/temperature)
    #print(sim)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    #print(mask)
    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    neg = sim.masked_select(mask).view(n_samples,-1).sum(dim=-1)
    #print((sim.masked_select(mask)).size())
    #print((sim.masked_select(mask).view(n_samples,-1)).size())
    #print(neg.size())

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    pos = torch.exp(torch.sum(output1 * output2, dim=-1)/temperature)
    print('pos: ',pos)
    # concatenate via the rows, stacking vertically
    pos = torch.cat([pos,pos], dim=0)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos/neg).mean()
    return loss



def test_loss(output1, output2, temperature):
    # concatenate v1 img and v2 img via the rows, stacking vertically
    print(output1.shape)
    out = tf.concat([output1, output2], 0)
    print(out.shape)
    n_samples = len(out)
    print(n_samples)

    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    # the diagonal of the matrix is the square of each vector element in the out vector, which shows the similarity between the same elements
    out_transpose = tf.transpose(out, conjugate=True)
    cov = tf.matmul(out, out_transpose)
    #print(cov)
    sim = tf.math.exp(cov/temperature)
    #print(sim)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    mask = ~tf.eye(n_samples, dtype=bool)
    #print(mask)
    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    true_values = tf.boolean_mask(sim, mask)
    # reshape to the original row but the masked value is removed
    reshaped_true_values = tf.reshape(true_values, shape=(n_samples,-1))
    # sum all rows
    neg = tf.math.reduce_sum(reshaped_true_values, axis=-1)
    #print(neg)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    pos = tf.math.exp(tf.math.reduce_sum(output1 * output2, axis=-1)/temperature)
    #print(pos)
    # concatenate via the rows, stacking vertically
    pos = tf.concat([pos,pos], 0)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = tf.math.reduce_mean(-tf.math.log(pos/neg))
    return loss


x1 = torch.randn(2, 128)
#print(x1)
#print(x1.size())

x2 = torch.randn(2, 128)


loss = nt_xent_loss(output1=x1, output2=x2, temperature=0.5)
print(loss)

loss_2 = test_loss(output1=x1, output2=x2, temperature=0.5)
print(loss_2)















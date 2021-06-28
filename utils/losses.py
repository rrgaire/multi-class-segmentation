
import torch
import torch.nn as nn


device = torch.device('cuda:1')
# class_wt = torch.Tensor([ 0.29500824,  1.87575569, 28.17118693, 24.01050374]).to(device)
#weight=class_wt
ce_loss = nn.CrossEntropyLoss()

def muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = ce_loss(d0,labels_v.type(dtype=torch.long))
	loss1 = ce_loss(d1,labels_v.type(dtype=torch.long))
	loss2 = ce_loss(d2,labels_v.type(dtype=torch.long))
	loss3 = ce_loss(d3,labels_v.type(dtype=torch.long))
	loss4 = ce_loss(d4,labels_v.type(dtype=torch.long))
	loss5 = ce_loss(d5,labels_v.type(dtype=torch.long))
	loss6 = ce_loss(d6,labels_v.type(dtype=torch.long))

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss0, loss
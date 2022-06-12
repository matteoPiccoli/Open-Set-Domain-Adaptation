import torch
from torch import nn
from center_loss import CenterLoss
from optimizer_helper import get_optim_and_scheduler
from matplotlib import pyplot as plt
import numpy as np


# Implement Step1

def do_epoch(args, feature_extractor, rot_cls, obj_cls, source_loader, optimizer, device):
    
    # Initialize the loss and the network
    rot_classes = 4 *(1 + args.is_multi * (args.n_classes_known - 1))
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=rot_classes, feat_dim=1024) # Add the center loss

    if args.weight_center_loss > 0:
        optimizer.add_param_group({'params':criterion_cent.parameters()})

    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    # Network to GPU
    net = feature_extractor.to(device)

    # Variables initialization (accuracy computation)
    correct_cls = 0
    correct_rot = 0

    #-----------------------------------#
    #             Training              #  
    #-----------------------------------#
    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):
      
        data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

        # zero_grad() clears old gradients from the last step
        # (otherwise you accumulate the gradients from all loss.backward() calls)
        optimizer.zero_grad()

        # Forward pass object classifier
        out_net = net.forward(data)
        out_obj = obj_cls.forward(out_net)
        # Forward pass rotational classifier and concatenation
        # of original + rotated images
        out_rot = net.forward(data_rot)
        inp_rot = torch.cat((out_net, out_rot), 1)
        out_rot = rot_cls.forward(inp_rot)

        # Loss function evaluation
        class_loss = criterion_xent(out_obj, class_l)
        rot_loss = args.weight_RotTask_step1 * criterion_xent(out_rot, rot_l)
        cen_loss = args.weight_center_loss * criterion_cent(inp_rot, rot_l)
        loss = class_loss + rot_loss + cen_loss

        # loss.backward() computes the derivative of the loss w.r.t. the parameters
        # (or anything requiring gradients) using backpropagation
        loss.backward()

        # By doing so, weight_center_loss would not impact on the learning of centers
        if args.weight_center_loss > 0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_center_loss)
        
        # opt.step() causes the optimizer to take a step 
        # based on the gradients of the parameters
        optimizer.step()

        # Softmax pass among the predictions
        cls_soft = nn.Softmax(dim=1)(out_obj)
        rot_soft = nn.Softmax(dim=1)(out_rot)
        # Selection of the highest prediction
        cls_pred = cls_soft.argmax(dim=1)
        rot_pred = rot_soft.argmax(dim=1)       

        # Counting correctly classified samples
        for (i, j, k, l) in zip(cls_pred, class_l, rot_pred, rot_l):
          correct_cls += (i == j).item()
          correct_rot += (k == l).item()
        
    # Accuracy computation
    acc_cls = 100 * np.asarray(correct_cls, dtype="float32")/ np.asarray(source_loader.dataset.__len__(), dtype="float32")
    acc_rot = 100 * np.asarray(correct_rot, dtype="float32")/ np.asarray(source_loader.dataset.__len__(), dtype="float32")
        
    return class_loss, acc_cls, rot_loss, acc_rot, cen_loss


def step1(args, feature_extractor, rot_cls, obj_cls, source_loader, device):
  
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step1,
                                                   args.learning_rate, args.train_all)

    for epoch in range(args.epochs_step1):

        # Print for every epoch the loss/accuracy
        print('Epoch: ', epoch)
        class_loss, acc_cls, rot_loss, acc_rot, cen_loss = do_epoch(args, feature_extractor, rot_cls, obj_cls, source_loader,
                                                          optimizer, device)
        print("Class Loss %.4f, Class Accuracy %.4f, Rot Loss %.4f, Rot Accuracy %.4f, Center Loss %.4f" % (
        class_loss.item(), acc_cls, rot_loss.item(), acc_rot, cen_loss.item()))

        scheduler.step()

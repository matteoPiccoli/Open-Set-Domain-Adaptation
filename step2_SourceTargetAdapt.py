import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np


# Implement Step2

def _do_epoch(args, epoch, feature_extractor, rot_cls, obj_cls, source_loader, target_loader_train, target_loader_eval, optimizer, device):
    
    # Initialize the loss and the network
    criterion = nn.CrossEntropyLoss().to(device)
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    # Network to GPU
    net = feature_extractor.to(device)

    # Variables initialization (accuracy computation)
    total_cls = 0
    correct_cls = 0
    total_rot = 0
    correct_rot = 0
    
    target_loader_train = cycle(target_loader_train)

    #-----------------------------------#
    #             Training              #  
    #-----------------------------------#
    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):

        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        # Data from Ds + Dt_unknown
        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        # Data from Dt_known
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        # zero_grad() clears old gradients from the last step
        # (otherwise you accumulate the gradients from all loss.backward() calls)
        optimizer.zero_grad()

        # Forward pass object classifier
        # Ds + Dt_unknown
        out_net = net.forward(data_source)
        out_obj = obj_cls.forward(out_net)
        # Forward pass rotational classifier
        # Dt_known
        out_target = net.forward(data_target)
        out_target_rot = net.forward(data_target_rot)
        inp_rot = torch.cat((out_target, out_target_rot), 1)
        out_rot = rot_cls.forward(inp_rot)     

        # Loss function evaluation
        class_loss = criterion(out_obj, class_l_source)
        rot_loss = criterion(out_rot, rot_l_target)
        loss = class_loss + args.weight_RotTask_step2 * rot_loss     

        # loss.backward() computes the derivative of the loss w.r.t. the parameters
        # (or anything requiring gradients) using backpropagation
        loss.backward()

        # opt.step() causes the optimizer to take a step based on the gradients of the parameters
        optimizer.step()

        # Softmax pass among the predictions
        cls_soft = nn.Softmax(dim=1)(out_obj)
        rot_soft = nn.Softmax(dim=1)(out_rot)
        # Selection of the highest prediction
        cls_pred = cls_soft.argmax(dim=1)
        rot_pred = rot_soft.argmax(dim=1)       

        # Counting correctly classified samples
        for (i, j, k, l) in zip(cls_pred, class_l_source, rot_pred, rot_l_target):
          correct_cls += (i == j)
          correct_rot += (k == l)

        # Total samples (summing up a batch at a time)
        total_cls += class_l_source.size(0)
        total_rot += rot_l_target.size(0)

    # Accuracy computation
    correct_cls = correct_cls.cpu().numpy()
    correct_rot = correct_rot.cpu().numpy()
    acc_cls = 100 * correct_cls / total_cls
    acc_rot = 100 * correct_rot / total_rot


    #-----------------------------------#
    #            Evaluation             #  
    #-----------------------------------#

    if (epoch%10==0 or epoch==args.epochs_step2-1):
        # Switching to test (instead of training) 
        feature_extractor.eval()
        obj_cls.eval()
        
        # Variables initialization (for OS*, UNK, HOS computation)
        correct_known = 0
        correct_unknown = 0
        total_known = 0
        total_unknown = 0

        with torch.no_grad():
          for it, (data, class_l,_,_) in enumerate(target_loader_eval):
              data, class_l  = data.to(device), class_l.to(device)

              # Forward pass object classifier
              out_net = net.forward(data)
              out_obj = obj_cls.forward(out_net)

              # Softmax pass among the predictions
              cls_soft = nn.Softmax(dim=1)(out_obj)
              # Selection of the highest prediction
              cls_pred = cls_soft.argmax(dim=1)

              # Correctly classified known/unknown samples 
              # and total number of samples
              for (i, j) in zip(cls_pred, class_l):
                  if (class_l >= args.n_classes_known):
                      correct_unknown += (i == 45)
                      total_unknown += 1
                  else:
                      correct_known += (i == j)
                      total_known += 1

        # Computing OS*, UNK and HOS
        correct_known = correct_known.cpu().numpy()
        correct_unknown = correct_unknown.cpu().numpy()
        OS_star = correct_known / total_known
        UNK = correct_unknown / total_unknown
        HOS = 2 *(OS_star * UNK)/(OS_star + UNK)

        print("OS* = %.4f" % OS_star)
        print("UNK = %.4f" % UNK)
        print("HOS = %.4f\n" % HOS)
        print("Class Loss %.4f, Class Accuracy %.4f, Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))
    

def step2(args, feature_extractor, rot_cls, obj_cls, source_loader, target_loader_train, target_loader_eval, device):
  
  optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step2,
                                                 args.learning_rate, args.train_all)

  for epoch in range(args.epochs_step2):
    
    #print('Epoch: ', epoch)
    _do_epoch(args,epoch,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
    scheduler.step()

import torch
from torch import nn
from torch.nn.functional import softmax, log_softmax 
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import math
from statistics import mean


# Implement the evaluation on the target for the known/unknown separation

def entropy_loss(x, is_multi, n_classes_known):
    if is_multi:
        return torch.sum(-(softmax(x, dim=1) * log_softmax(x, dim=1)) / math.log(n_classes_known), 1).mean()
    else:
        return torch.sum(-softmax(x, dim=1) * log_softmax(x, dim=1), 1).mean()

    
def evaluation(args, feature_extractor, rot_cls, target_loader_eval, device):

    # Network to GPU
    net = feature_extractor.to(device)
        
    # Lists initialization (AUROC computation)
    ground_truth = []  
    normality_score = []
    rot_score = []
    ent_score = []


    #-----------------------------------#
    #     Target domain separation      #  
    #-----------------------------------#
    with torch.no_grad():
        for it, (data, class_l, _, _) in enumerate(target_loader_eval):
            data, class_l = data.to(device), class_l.to(device)
            
            # Initialization
            k_list = torch.zeros(args.n_classes_known).to(device)
            ent_rot = torch.cuda.FloatTensor()
            data_rot = data
            rot_l = 0
            rot_soft = 0
            
            # Pass to the network the rotated images            
            for i in range(0, 4):

                # Forward pass rotational classifier
                net.eval()
                out_net = net.forward(data)
                out_rot = net.forward(data_rot)
                net.train()
                inp_rot = torch.cat((out_net, out_rot), 1)
                inp_rot = inp_rot.to(device)
                rot_cls.eval()
                out_rot = rot_cls.forward(inp_rot)
                rot_cls.train()
                # Softmax
                soft = nn.Softmax(dim=1)(out_rot)

                # Increment the rotational score, adding the probability
                # of the current rotation
                if args.is_multi:
                    for k in range(args.n_classes_known):
                        k_list[k] += soft[0][(k * 4) + i]    
                else:          
                    rot_soft += soft[0][i]
                # Concatenate to entropy_rot, for the entropy score
                # computation
                ent_rot = torch.cat((ent_rot, out_rot), 0)
                                                                                          
                # Rotation of the image and incrementing the label
                rot_l += 1
                data_rot = torch.rot90(data, k=rot_l, dims=[2, 3])
            
            # Ground_truth list 
            # 1 --> known class
            # 0 --> unknown class
            if class_l < args.n_classes_known:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            # Score computation
            if args.is_multi:
                k_list = k_list / 4
                rot_soft = max(k_list)
                rot_score.append(rot_soft.item())
            else:
                rot_score.append(rot_soft.item() / 4)
            
            ent_score.append(entropy_loss(ent_rot, args.is_multi, args.n_classes_known).item())

    # Normalization of the entropy score
    raw = ent_score
    min_raw = min(raw)
    max_raw = max(raw)
    for index, item in enumerate(raw):
        ent_score[index] = ((item - min_raw) / (max_raw - min_raw))


    #-----------------------------------#
    #      Create new .txt files        #  
    #-----------------------------------#
    rand = random.randint(0, 100000)
    print('Generated random number is : ', rand)

    # This .txt files will have the names of the source images and the names of the target images selected as unknown
    # Ds + Dt_unknown
    target_unknown = open('drive/My Drive/Colab Notebooks/new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')
    # This .txt files will have the names of the target images selected as known
    # Dt_known
    target_known = open('drive/My Drive/Colab Notebooks/new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')
    
    # Read target domain .txt file
    f = open('drive/My Drive/Colab Notebooks/txt_list/' + args.target + '.txt', 'r')
    path = f.readlines()
    target_file_names = []
    # Splits every row in two parts and takes only the first part
    # the split character is ' '
    for row in path:
        row = row.split(' ')
        target_file_names.append(row[0])
        
    # Read source domain .txt file
    s = open('drive/My Drive/Colab Notebooks/txt_list/' + args.source + '_known.txt', 'r')
    path_s = s.readlines()
    
    # Write the known/unknown samples' paths in the two files
    # according to the threshold
    number_of_known_samples = 0
    number_of_unknown_samples = 0

    # Write the source domain list
    for count, line in enumerate(path_s):
        target_unknown.write(line)
    target_unknown.write('\n')

    for count in range(len(target_loader_eval)):

        # Normality score computation and predicted labels
        normality_score.append(max(rot_score[count], 1 - ent_score[count]))

        if normality_score[count] > args.threshold:
            target_known.write(path[count])
            number_of_known_samples += 1
        else:
            target_unknown.write(target_file_names[count] + ' ' + str(args.n_classes_known) + '\n')
            number_of_unknown_samples += 1
    
    # AUROC metric
    auroc = roc_auc_score(ground_truth, normality_score)
    print('AUROC %.4f' % auroc)
    print("Avg Normality: " + str(mean(normality_score)))  
       
    # Close files
    target_unknown.close()
    target_known.close()
    f.close()
    s.close()

    print('The number of target samples selected as known is: ', number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand


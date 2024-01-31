import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import byzantine1

def fltrust(gradients, net, lr, f, byz, score_list,value_remove,two_min_values1,idxes_malicious1,two_min_values2,idxes_malicious2):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    print("len(gradients)",len(gradients))
    print("len(gradients[0])",len(gradients[0]))
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    #weighted trust score
    print("score_list",score_list)
    print("max(score_list),min:",max(score_list),value_remove)
    normalized_score_list = [None]*6
    for x in range(6):
        normalized_score_list[x] = (score_list[x]-value_remove)/(max(score_list)-value_remove)
    print("normalized_score_list,sumTotal",normalized_score_list,sum(normalized_score_list))

#    before_pa=param_list[idx_attack]
    param_list_attack = param_list.copy()
    param_list_attack = byzantine1.trim_attack(param_list_attack, net, lr, f)
    # 1st malicious performs attack
    print("Before_param_list[idx_attack1][0:10]",param_list[0][0:10])
    before_param_list = param_list[1][0:3]
    param_list[0] = param_list_attack[0]
    print("After_param_list[idx_attack1][0:10]",param_list[0][0:10])
    print("check if =",param_list[1][0:3]==before_param_list)

    n = len(param_list)
    print("n",n)
    print("param_list[0]deLEN",len(param_list[0]))
    print("param_list.shape",len(param_list))
    # use the last gradient (server update) as the trusted source
#    baseline = nd.array(param_list[-1]).squeeze()
#    print("pbaseline",baseline.shape)

#    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
#    for each_param_list in param_list:
#        each_param_array = nd.array(each_param_list).squeeze()
#        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        
#    cos_sim = nd.stack(*cos_sim)[:-1]
#    cos_sim = nd.maximum(cos_sim, 0) # relu
#    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score

    # weighted averaging by the score list
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_score_list[i])
    print("new_param_list[0][0:10]:",new_param_list[0][0:10])
    print("new_param_list.size:",len(new_param_list),len(new_param_list[0]))
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    print("global_update.shape:",global_update.shape)
    print("global_update[0:10]:",global_update[0:10])
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size       


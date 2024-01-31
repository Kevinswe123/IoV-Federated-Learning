from __future__ import print_function
import nd_aggregation1
import mxnet as mx
from mxnet import nd, autograd, gluon
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import byzantine1
import wandb
import logging
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.006)
    parser.add_argument("--nworkers", help="# workers", type=int, default=100)
    parser.add_argument("--niter", help="# iterations", type=int, default=2500)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=-1)
    parser.add_argument("--nrepeats", help="seed", type=int, default=-1)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=20)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fltrust")
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--frac", help="fraction of minigroup clients", type=float, default=0.6)
    return parser.parse_args()
#done
def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
#done
def get_cnn(num_outputs=5):
    # define the architecture of the CNN
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn
def plot_sample(X,y,index):
    #classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    classes=['automobile','deer','dog','horse','truck']
    z=y.asnumpy()
    z=z.astype('int32')
    transposed=nd.transpose(X[index],(1,2,0))
    plt.imshow(transposed.asnumpy())
    plt.xlabel(classes[z[index]])
    plt.show()

def get_net(net_type, num_outputs=5):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    else:
        raise NotImplementedError
    return net
#done
def get_shapes(dataset):
    # determine the input/output shapes
    if dataset == 'FashionMNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'MNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'CIFAR10':
        num_inputs = 32 * 32 * 3
        num_outputs = 5
        num_labels = 5
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine1.no_byz
    elif byz_type == 'trim_attack':
        return byzantine1.trim_attack
    else:
        raise NotImplementedError

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'MNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'CIFAR10':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        #dataset
        train_dataset = mx.gluon.data.vision.CIFAR10(train=True, transform=transform)
        test_dataset = mx.gluon.data.vision.CIFAR10(train=False, transform=transform)
        #TRAIN_DATASET:choose 5 classes from CIFAR10:labels=1,4,5,7,9
        print("train_dataset has shape {}, and train_dataset[0] has shape {}, train_datasettype {}, train_dataset[0]type {}, train_dataset[0][0]type {}, train_dataset[0][1]type {}".format(len(train_dataset),len(train_dataset[0]),type(train_dataset),type(train_dataset[0]),type(train_dataset[0][0]),type(train_dataset[0][1])))
        print("train_dataset[1][0] {}, and train_dataset[1][1] {}, and train_dataset[0][0].shape {}".format(train_dataset[1][0],train_dataset[1][1],train_dataset[0][0].shape))
        index_1 = list(zip(*train_dataset))
        index_2 = list(index_1[1])
        print("index_2 {}, and index_2 has shape {}, and index_2 type {}".format(index_2[0:5],len(index_2),type(index_2)))
        y_index = np.array(index_2)
        class_1_index = np.where(y_index == 1)[0]
        class_4_index = np.where(y_index == 4)[0]
        class_5_index = np.where(y_index == 5)[0]
        class_7_index = np.where(y_index == 7)[0]
        class_9_index = np.where(y_index == 9)[0]
        allclasses_index = np.concatenate((class_1_index,class_4_index,class_5_index,class_7_index,class_9_index))
        print("allclasses_index_beforesort {}, and allclasses_indexLEN {}".format(allclasses_index,len(allclasses_index)))
        allclasses_index = np.sort(allclasses_index)
        print("allclasses_index_aftersort {}, and allclasses_indexLEN {}, and indextype {}".format(allclasses_index,len(allclasses_index),type(allclasses_index)))
        train_dataset_5classes_raw = np.array(train_dataset)[allclasses_index]
        X_5classes_data = train_dataset_5classes_raw[:,0]
        y_5classes_label = train_dataset_5classes_raw[:,1]
        #change labels from 14579 to 01234
        y_5classes_label[np.where(y_5classes_label==1)] = 0
        y_5classes_label[np.where(y_5classes_label==4)] = 1
        y_5classes_label[np.where(y_5classes_label==5)] = 2
        y_5classes_label[np.where(y_5classes_label==7)] = 3
        y_5classes_label[np.where(y_5classes_label==9)] = 4
        print('5classes_label[0:5]',y_5classes_label[0:5])
        train_dataset_5classes = mx.gluon.data.ArrayDataset(X_5classes_data,y_5classes_label)
        print("5classes has len {}, and 5classes[0]  {}, 5classestype {}, 5classes[0]type {}, 5classes[0][1]type {}".format(len(train_dataset_5classes),train_dataset_5classes[0],type(train_dataset_5classes),type(train_dataset_5classes[0]),type(train_dataset_5classes[0][1])))
        print("train_dataset[0][0] {}, and train_dataset[0][1] {}, and train_dataset[0][0].shape {}".format(train_dataset_5classes[0][0],train_dataset_5classes[0][1],train_dataset_5classes[0][0].shape))
        #TEST_DATASET:choose 5 classes from CIFAR10:labels=1,4,5,7,9
        print("test_dataset has shape {}, and test_dataset[0] has shape {}, test_datasettype {}, test_dataset[0]type {}, test_dataset[0][0]type {}, tseyt_dataset[0][1]type {}".format(len(test_dataset),len(test_dataset[0]),type(test_dataset),type(test_dataset[0]),type(test_dataset[0][0]),type(test_dataset[0][1])))
        print("test_dataset[1][0] {}, and test_dataset[1][1] {}, and test_dataset[0][0].shape {}".format(test_dataset[1][0],test_dataset[1][1],test_dataset[0][0].shape))
        index_1_test = list(zip(*test_dataset))
        index_2_test = list(index_1_test[1])
        print("index_2_test {}, and index_2_test has shape {}, and index_2_test type {}".format(index_2_test[0:5],len(index_2_test),type(index_2_test)))
        y_index_test = np.array(index_2_test)
        class_1_index_test = np.where(y_index_test == 1)[0]
        class_4_index_test = np.where(y_index_test == 4)[0]
        class_5_index_test = np.where(y_index_test == 5)[0]
        class_7_index_test = np.where(y_index_test == 7)[0]
        class_9_index_test = np.where(y_index_test == 9)[0]
        allclasses_index_test = np.concatenate((class_1_index_test,class_4_index_test,class_5_index_test,class_7_index_test,class_9_index_test))
        print("allclasses_index_test_beforesort {}, and allclasses_index_testLEN {}".format(allclasses_index_test,len(allclasses_index_test)))
        allclasses_index_test = np.sort(allclasses_index_test)
        print("allclasses_index_test_aftersort {}, and allclasses_index_testLEN {}, and indextype {}".format(allclasses_index_test,len(allclasses_index_test),type(allclasses_index_test)))
        test_dataset_5classes_raw = np.array(test_dataset)[allclasses_index_test]
        X_5classes_data_test = test_dataset_5classes_raw[:,0]
        y_5classes_label_test = test_dataset_5classes_raw[:,1]
        #change labels from 14579 to 01234
        y_5classes_label_test[np.where(y_5classes_label_test==1)] = 0
        y_5classes_label_test[np.where(y_5classes_label_test==4)] = 1
        y_5classes_label_test[np.where(y_5classes_label_test==5)] = 2
        y_5classes_label_test[np.where(y_5classes_label_test==7)] = 3
        y_5classes_label_test[np.where(y_5classes_label_test==9)] = 4
        print('5classes_label_test[0:5]',y_5classes_label_test[0:5])
        test_dataset_5classes = mx.gluon.data.ArrayDataset(X_5classes_data_test,y_5classes_label_test)

        print("5classes_test has len {}, and 5classes_test[0]  {}, 5classes_testtype {}, 5classes_test[0]type {}, 5classes_test[0][1]type {}".format(len(test_dataset_5classes),test_dataset_5classes[0],type(test_dataset_5classes),type(test_dataset_5classes[0]),type(test_dataset_5classes[0][1])))
        print("test_dataset[0][0] {}, and test_dataset[0][1] {}, and test_dataset[0][0].shape {}".format(test_dataset_5classes[0][0],test_dataset_5classes[0][1],test_dataset_5classes[0][0].shape))

        #dataloader
        train_data = mx.gluon.data.DataLoader(train_dataset_5classes, 25000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(test_dataset_5classes, 5000, shuffle=False, last_batch='rollover')
        
    else:
        raise NotImplementedError
    return train_data, test_data
#GAILE JIDE gaihui FashionMNIST!!!
def assign_data(train_data, bias, ctx, num_labels=5, num_workers=100, server_pc=10, p=0.1, dataset="CIFAR10", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    server_data = []
    server_label = []

    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1]) #samp_dis=[22,10,22,22,24]

    # randomly assign the data points based on the labels
    #server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == "MNIST":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == "CIFAR10":
                x = x.as_in_context(ctx).reshape(1,3,32,32)
            else:
                raise NotImplementedError
            y = y.as_in_context(ctx)

            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)

    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]


    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    for j in range(10):
        edata_NO = len(each_worker_data[j])
        elabel_NO = len(each_worker_label[j])
        print("edata_NO",edata_NO)
        print("elabel_NO",elabel_NO)

    return each_worker_data, each_worker_label


def main(args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    # device to use
    ctx = get_device(args.gpu)
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter
    frac = args.frac
#generate 80 random trust degrees
    trust_value_list60 = []
    for i in range(40):
        x = round(random.uniform(0.9,1.0),6)
        trust_value_list60.append(x)
    for i in range(60):
        x = round(random.uniform(0.8,0.9),6)
        trust_value_list60.append(x)
#
    print("trust_value_list60",trust_value_list60)

    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"

    with ctx:

        # model architecture
        net = get_net(args.net, num_outputs)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        #

        grad_list = []
        score_list = []
        test_acc_list = []

        wandb.init(
            project="FL_application_control_testTrustDegreeWithoutAttacks",
            name="FL_app_controlGroup(MMS)_5classes_testTrustDegreeWithoutAttacks_100XUAN60" + str(args.dataset) + "+" + "net " + str(args.net) + "+" +"niter " + str(args.niter) +"+nworkers " + str(
        args.nworkers)+ "bias" + str(args.bias) + "batch_size " + str(args.batch_size) + "lr" + str(args.lr),
            config=args
        )
        # load the data
        # fix the seeds for loading data
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        train_data, test_data = load_data(args.dataset)

#########
        for X_batch,y_batch in train_data:
            print("X_batch_train has shape {}, and y_batch_train has shape {}".format(X_batch.shape,y_batch.shape))
        print("train_data row",len(train_data))
        print("test_data row",len(test_data))


        # assign data to the server and clients
        each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers,
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        print("each_worker_data SHAPE",len(each_worker_data),len(each_worker_data[0]))
        print("each_worker_data[0][0:1]",each_worker_data[0][0:1])
        print("each_worker_label[0][0:5]",each_worker_label[0][0:5])
        # begin training
        for e in range(niter):
            logging.info("################Communication round : {}".format(e))
            m = max(int(frac*num_workers),1)
            tic = time()

            train_locals_loss = []
            print("m:",m)
            idxs_minigroup = np.random.choice(range(num_workers), size=m, replace=False) 
            for i in idxs_minigroup:
                print("the_ith_client:",i)
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                score_list.append(trust_value_list60[i])
                train_locals_loss.append(nd.mean(loss).asscalar())
            train_loss_avg = sum(train_locals_loss)/len(train_locals_loss)
            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
#                minibatch = np.random.choice(list(range(server_data.shape[0])), size=args.server_pc, replace=False)
#                with autograd.record():
#                    output = net(server_data)
#                    loss = softmax_cross_entropy(output, server_label)
#                loss.backward()
#                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                # perform the aggregation
                nd_aggregation1.fltrust(grad_list, net, lr, args.nbyz, byz, score_list)

            del grad_list
            grad_list = []
            del score_list
            score_list = []

            # evaluate the model accuracy
            if (e + 1) % 10 == 0:
                logging.info("################evaluation_test : {}".format(e))
                
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)

                stats = {'training_loss': train_loss_avg, 'round': e}
                wandb.log({"Train/Loss": train_loss_avg, "round": e})
                logging.info(stats)

                stats = {'test_accuracy': test_accuracy, 'round': e}
                wandb.log({"Test/Acc": test_accuracy, "round": e})
                logging.info(stats)

                stats = {'time': time()-tic, 'round': e}
                wandb.log({"Time": time()-tic, "round": e})
                logging.info(stats)


                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))

        del test_acc_list
        test_acc_list = []

if __name__ == "__main__":
    args = parse_args()
    main(args)

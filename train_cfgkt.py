# @Time    : 2024/10/12
# @File    : train_cfgkt.py
# @Software: PyCharm






import argparse
import yaml
from Dataprocess.dataloader_cfgkt import Data_set
# from Dataprocess.dataloader_lpkt import Data_set
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from CFGKT import CFGKT
from BCDN_cfgkt import BCDN_cfgkt
from sklearn import metrics





def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    all_pred = np.array(all_pred)
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred = np.array(all_pred)
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)



parser = argparse.ArgumentParser(description='myDemo')
parser.add_argument('--epochs',type=int,default=30,metavar='N',help='number of epochs to train (defauly 10 )')
parser.add_argument('--data_dir', type=str, default='dataset/Processed_data/',help="the data directory, default as './data")




def train(model, train_dataloader,query_data, optimizer, criterion,device,cfgktcof):
    model_name = cfgktcof["train"]["name"]
    lambda_l2 = cfgktcof["train"]["lambda_l2"]
    model.train()
    train_loss = []
    num_corrects = 0
    num_total = 0
    label_num = []
    outs = []
    index = 0

    for exercise_seq, concept_seq, response_seq, taken_time_seq, interval_time_seq in tqdm(train_dataloader,
                                                                                           desc='Training',
                                                                                           mininterval=2):
        index = index + 1

        exercise_seq = exercise_seq.to(device)
        concept_seq = concept_seq.to(device)
        response_seq = response_seq.to(device)
        taken_time_seq = taken_time_seq.to(device)
        interval_time_seq = interval_time_seq.to(device)
        targetid = exercise_seq.clone()

        target = response_seq.float().clone()

        optimizer.zero_grad()
        if model_name == "cfgkt":
            output = model(concept_seq, response_seq, exercise_seq, taken_time_seq, interval_time_seq)
            info_loss = 0
            l2_loss = 0
        elif model_name == "bcdn_cfgkt":

            # Load the learner's knowledge state obtained by pre-training
            student_query_matrix = query_data['all_hidden_states']
            query_correct_sequence = query_data['all_exercise_correct']

            output, info_loss = model(concept_seq, response_seq, exercise_seq, student_query_matrix,
                                      query_correct_sequence, taken_time_seq, interval_time_seq)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            l2_loss = lambda_l2 * l2_norm

        else:
            pred = 0

        targetid[:, 0] = 0
        mask = targetid > 0
        masked_pred = output[mask]
        masked_truth = target[mask]

        cross_loss = criterion(masked_pred, masked_truth)
        final_loss = cross_loss + info_loss / 100 + l2_loss
        final_loss.backward()
        optimizer.step()
        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        train_loss.append(final_loss.item())
        label_num.extend(masked_truth)
        outs.extend(masked_pred)






    loss = np.average(train_loss)
    auc = compute_auc(label_num, outs)
    accuracy = compute_accuracy(label_num, outs)
    return loss, accuracy, auc



def test_epoch(model,test_loader,query_data, criterion,cfgktcof,device="cpu"):
    model_name = cfgktcof["train"]["name"]
    lambda_l2 = cfgktcof["train"]["lambda_l2"]
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    label_num = []
    outs = []
    for exercise_seq, concept_seq, response_seq, taken_time_seq, interval_time_seq in tqdm(test_loader, desc='Testing',
                                                                                           mininterval=2):
        exercise_seq = exercise_seq.to(device)
        concept_seq = concept_seq.to(device)
        response_seq = response_seq.to(device)
        taken_time_seq = taken_time_seq.to(device)
        interval_time_seq = interval_time_seq.to(device)
        targetid = exercise_seq.clone()
        target = response_seq.float().clone()

        with torch.no_grad():
            if model_name == "cfgkt":
                output = model(concept_seq, response_seq, exercise_seq, taken_time_seq, interval_time_seq)
                info_loss = 0
                l2_loss = 0
            elif model_name == "bcdn_cfgkt":

                # Load the learner's knowledge state obtained by pre-training
                student_query_matrix = query_data['all_hidden_states']
                query_correct_sequence = query_data['all_exercise_correct']

                output, info_loss = model(concept_seq, response_seq, exercise_seq, student_query_matrix,
                                          query_correct_sequence, taken_time_seq, interval_time_seq)

                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                l2_loss = lambda_l2 * l2_norm
            else:
                output = 0

        targetid[:, 0] = 0

        mask = targetid > 0
        masked_pred = output[mask]
        masked_truth = target[mask]
        cross_loss = criterion(masked_pred, masked_truth)
        final_loss = cross_loss + info_loss / 100 + l2_loss
        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        train_loss.append(final_loss.item())
        label_num.extend(masked_truth)
        outs.extend(masked_pred)



    loss = np.average(train_loss)
    auc = compute_auc(label_num,outs)
    acc = compute_accuracy(label_num, outs)
    return loss, acc, auc


if __name__ == "__main__":

    datasetname = "JUNYI"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if datasetname == 'ASSIST09':
        parser.add_argument('--datasetname', type=str, default='ASSIST09', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2009/',
                            help="train data file, default as './ASSISTments2009/'.")
        parser.add_argument('--n_question', type=int, default=123, help='the number of unique questions in the dataset')

    elif datasetname == 'ASSIST12':
        parser.add_argument('--datasetname', type=str, default='ASSIST12', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train data file, default as 'train_data.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train data file, default as 'test_data.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSIST2012/',
                            help="train data file, default as './ASSIST2012/'.")
        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')


    elif datasetname == 'ASSIST17':
        parser.add_argument('--datasetname', type=str, default='ASSIST17', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSIST2017/',
                            help="train data file, default as './ASSIST2017/'.")
        parser.add_argument('--n_question', type=int, default=102, help='the number of unique questions in the dataset')


    elif datasetname == 'JUNYI':
        parser.add_argument('--datasetname', type=str, default='JUNYI', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train data file, default as 'train_data.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train data file, default as 'test_data.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./JUNYI/',
                            help="train data file, default as './JUNYI/'.")
        parser.add_argument('--n_question', type=int, default=25784,
                            help='the number of unique questions in the dataset')


    parsers = parser.parse_args()

    print("parser:", parsers)
    # train path and test path
    print(f'loading Dataset  {parsers.datasetname}...')

    # Training Settings
    f = open("conf/cfgkt.yml", 'r', encoding='utf-8')
    cfgktcof = yaml.safe_load(f.read())

    # Dataset Settings
    f = open("conf/dataset.yml", 'r', encoding='utf-8')
    dataset_cof = yaml.safe_load(f.read())



    modelname =  cfgktcof["train"]["name"]
    input_dim = cfgktcof["train"]["input_dim"]
    student_num = dataset_cof[datasetname.lower()]["student_number"]
    exercise_size = dataset_cof[datasetname.lower()]["exercise_size"]
    concept_size = dataset_cof[datasetname.lower()]["concept_size"]
    hidden_dim = cfgktcof["train"]["hidden_dim"]
    n_blocks = cfgktcof["train"]["n_blocks"]
    dropout = cfgktcof["train"]["dropout"]
    kq_same = cfgktcof["train"]["kq_same"]
    col_student_num = cfgktcof["train"]["col_student_num"]
    seq_max_length = cfgktcof["train"]["max_seq_length"]
    q_gamma = cfgktcof["train"]["q_gamma"]
    n_at = dataset_cof[datasetname.lower()]["HDtimeTaken"]
    n_it = dataset_cof[datasetname.lower()]["HDinterval_time"]
    d_ff = cfgktcof["train"]["d_ff"]
    n_heads = cfgktcof["train"]["n_heads"]
    final_fc_dim = cfgktcof["train"]["final_fc_dim"]
    memory_size = cfgktcof["train"]["memory_size"]

    user_short_nh = cfgktcof["train"]["user_short_nh"]
    user_short_nv = cfgktcof["train"]["user_short_nv"]

    transformer_encoder_layers = cfgktcof["train"]["transformer_encoder_layers"]
    transformer_encoder_heads = cfgktcof["train"]["transformer_encoder_heads"]
    transformer_encoder_dim_feedforward = cfgktcof["train"]["transformer_encoder_dim_feedforward"]
    transformer_encoder_layer_norm_eps = cfgktcof["train"]["transformer_encoder_layer_norm_eps"]
    sequence_last_m = cfgktcof["train"]["sequence_last_m"]



    if modelname.lower() == "cfgkt":
        train_path = parsers.data_dir + parsers.datasetname + parsers.train_file
        test_path = parsers.data_dir + parsers.datasetname + parsers.test_file
        train_set = Data_set(path=train_path, max_seq_length=cfgktcof["train"]["max_seq_length"])
        test_set = Data_set(path=test_path, max_seq_length=cfgktcof["train"]["max_seq_length"])
        train_loader = DataLoader(train_set, cfgktcof["train"]["batch_size"], shuffle=True)
        test_loader = DataLoader(test_set, cfgktcof["train"]["batch_size"])
        query_data = ""
        kt_model = CFGKT(n_concepts=concept_size,n_pid=exercise_size,d_model=input_dim,n_blocks=n_blocks,kq_same=kq_same,
                         dropout=dropout,model_type='CFGKT',memory_size=memory_size,final_fc_dim=final_fc_dim,n_heads= n_heads,
                         d_ff=d_ff,time=n_at,interval=n_it)
    elif modelname.lower() == "bcdn_cfgkt":
        train_path = parsers.data_dir + parsers.datasetname + parsers.train_file
        test_path = parsers.data_dir + parsers.datasetname + parsers.test_file
        train_set = Data_set(path=train_path, max_seq_length=cfgktcof["train"]["max_seq_length"])
        test_set = Data_set(path=test_path, max_seq_length=cfgktcof["train"]["max_seq_length"])
        train_loader = DataLoader(train_set, cfgktcof["train"]["batch_size"], shuffle=True)
        test_loader = DataLoader(test_set, cfgktcof["train"]["batch_size"])
        query_data = torch.load('dataset/Processed_data/' + datasetname.upper() +'/hidden_states_and_correctfinal1.pt')
        kt_model = BCDN_cfgkt(n_at, n_it,exercise_size,concept_size, input_dim,dropout,hidden_dim,col_student_num,seq_max_length,n_blocks,kq_same,
                 memory_size, final_fc_dim, n_heads,d_ff)
    else:
        print(f"Model name not found: {cfgktcof['train']['name']}")
        raise RuntimeError("Runtime error: Model name not found{dktcof['train']['name']}")


    if cfgktcof["optimizer"]["name"] == "adam":
        opt = torch.optim.Adam(kt_model.parameters(), lr=cfgktcof["optimizer"]["lr"], betas=(0.9, 0.999), eps=1e-8)

    criterion = nn.BCELoss(reduction="mean")
    kt_model.to(device)
    criterion.to(device)

    patience = 8
    best_val_loss = float('inf')
    patience_counter = 0
    final_result = {}

    for epoch in range(cfgktcof["train"]["epoch"]):
        train_loss, train_acc, train_auc = train(kt_model, train_loader,query_data, opt, criterion,device,cfgktcof)

        print(
            "epoch - {} train_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, train_loss, train_acc, train_auc,))

        val_loss, avl_acc, val_auc = test_epoch(kt_model, test_loader, query_data,criterion, cfgktcof,device=device)
        print("epoch - {} test_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, val_loss, avl_acc, val_auc))
        final_result["epoch" + str(epoch)] = [avl_acc, val_auc]


        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    sorted_dict = dict(sorted(final_result.items(), key=lambda item: item[1][1], reverse=True))
    top_five = list(sorted_dict.items())[:5]
    first_values = [value[0] for _, value in top_five]
    second_values = [value[1] for _, value in top_five]
    print("The number of col：{}".format(col_student_num))
    # print("learning rate：{}".format(cfgktcof["optimizer"]["lr"]))
    print("lambda_l2:{}".format(cfgktcof["train"]["lambda_l2"]))
    print(modelname + ":")
    print(datasetname + ":")
    print("ACC：", first_values)
    print("AUC：", second_values)
    # torch.save(kt_model, 'dataset/Processed_data/' + datasetname + '/' + modelname + '_' + datasetname + '.pth')





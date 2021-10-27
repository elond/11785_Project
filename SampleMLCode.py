import csv
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support as score
from random import randint
import matplotlib.pyplot as plt
import sys
################################################################################
#--------Directories, Data Titles, and Conditionals ---------------------------#
################################################################################
INPUT_DIR = "/home/esteban/Documents/Research/Data/All_Viruses/"
OUTPUT_DIR = "/home/esteban/Documents/Research/Data/Results/"
TRAIN_DATA_FILE = INPUT_DIR + "trainingReads/strain_run_train_data.npy"
TRAIN_LABEL_FILE = INPUT_DIR + "trainingReads/strain_run_train_labels.npy"
VAL_DATA_FILE = INPUT_DIR + "validationReads/strain_run_valid_data.npy"
VAL_LABEL_FILE = INPUT_DIR + "validationReads/strain_run_valid_labels.npy"
TEST_DATA_FILE = INPUT_DIR + "testReads/strain_run_test_data.npy"
TEST_LABEL_FILE = INPUT_DIR + "testReads/strain_run_test_labels.npy"

PPPE = 100                                                                       # How many plot points per epoch do you want?
NUM_DATA = 1                                                                    # Number of runs through the same architecture
FIRST_LAYER_RANGE = [200]
SECOND_LAYER_RANGE = [200]
WANT_CUDA = True                                                                # Do you want to run on a GPU?
if WANT_CUDA == True:
    DEV = "cuda:0"                          # Assigns GPU as device------------#
else:
    DEV = "cpu"                             # Assigns CPU as device------------#
DEVICE = torch.device(DEV)
################################################################################
#--------Hyperparameters-------------------------------------------------------#
################################################################################
LR_LIST = [0.001]                                                               # What learning rate do you want?
EPOCH_NUM = 15                                                                  # For how many epochs do you want to run the data?
BATCH_SIZE_LIST = [128]                                                         # How big is the mini-batch size?
window = 3
PADDING_LIST = 0
################################################################################
#--------Tracking and Saving Code----------------------------------------------#
################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.recent = [0]*50
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.recent = self.recent[1:(len(self.recent)+1)] + [self.val]
        self.mov_avg = sum(self.recent)/len(self.recent)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

################################################################################
#--------Network Archictecture-------------------------------------------------#
################################################################################
class Net(nn.Module):
    def __init__(self, conv1_hu, conv2_hu, window):
        super(Net, self).__init__()

        conv_input_rows = 250
        conv_input_col = 4
        stride = 1
        padding = 0
        self.conv1 = nn.Conv1d(conv_input_col,conv1_hu,window,stride,padding)
        self.out_dim = int((conv_input_rows + 2*padding - window)/stride + 1)
        self.conv2 = nn.Conv1d(conv1_hu, conv2_hu,window,stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)
        self.conv3 = nn.Conv1d(conv2_hu, 200, window, stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)

        self.fc1 = nn.Linear(self.out_dim*200, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16,2)

    def forward(self, x):
        x = x.float()
        m = nn.MaxPool1d(1)
        x = m(F.relu(self.conv1(x)))
        x = m(F.relu(self.conv2(x)))
        x = m(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]                                                     # Selects all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

################################################################################
#--------Load the appropriate data---------------------------------------------#
################################################################################
class LoadData(Dataset):
    def __init__(self, data, labels):
        data_x = np.load(data)                                                  # Load the examples-------------#
        data_y = np.load(labels)                                                # Load the labels---------------#
        self.data = torch.from_numpy(data_x).permute(0,2,1)                     # Convert examples to tensor----#
        self.labels = torch.from_numpy(data_y.astype(int))                      # Convert labels to tensor------#
        self.len = len(data_y)                                                  # Determine length of dataset---#

    def __getitem__(self,index):                                                # Get item and length functions #
        return self.data[index], self.labels[index]                             # used for data loader function-#                                                                                # Length function used for   #
    def __len__(self):
        return self.len

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def TrainModel(train_loader, val_loader, net, optimizer, device, epoch, loss_func, run_los_interval, train_losses, val_losses):
    val_iter = iter(val_loader)
    progress = ProgressMeter(len(train_loader), [train_losses, val_losses], prefix="Epoch: [{}]".format(epoch))
    train_mov_avg = []
    val_mov_avg = []
    val_los_interval = int(len(val_loader)/len(train_loader)*run_los_interval)

    for i, data in enumerate(train_loader, 1):
        net.train()
        inputs, labels = data                                                   # Get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad()                                                   # Zero gradient
        output = net(inputs.to(device))                                         # Forward feed
        loss = loss_func(output, labels.long().to(device))                      # Determine loss
        loss.backward()                                                         # Backward propagation
        optimizer.step()                                                        # Optimize model

        train_losses.update(loss.item(), inputs.size(0))

        if i % run_los_interval == 0:                                                       # Print every {pppe} mini-batches
            net.eval()
            with torch.no_grad():
                for x in range(val_los_interval):
                    try:
                        val_inputs, val_labels = next(val_iter)
                    except:
                        pass
                    val_output = net(val_inputs.to(device))
                    val_loss = loss_func(val_output, val_labels.long().to(device))
                    val_losses.update(val_loss.item(), val_inputs.size(0))

            train_mov_avg.append(train_losses.mov_avg)
            val_mov_avg.append(val_losses.mov_avg)
            progress.display(i)

    return train_mov_avg, val_mov_avg

def RunDataset(loader, ml_data, device, net, csv_writer,network_name,fig_dir,t_time, hyp_par):
    with torch.no_grad():
        #--------Determine results from datasets-------------------------------#
        start_time = time.time()
        pred_list = []
        labels_list = prob_list = np.array([])
        for i, data in enumerate(loader):
            inputs, labels = data
            output = net(inputs.to(device))
            softmax = F.softmax(output, dim=1).cpu()
            prob = list(softmax.numpy())
            predictions = np.argmax((prob), axis=1)

            prob_list = np.concatenate((prob_list, np.array(prob)[:,1]), axis = 0)
            pred_list += list(predictions)
            labels_list = np.concatenate((labels_list,labels.numpy()), axis = 0)

        #--------Classification metrics----------------------------------------#
        tn, fp, fn, tp = confusion_matrix(labels_list, pred_list).ravel()           # true and false positives and negatives-#
        pr,rc = score(labels_list, pred_list,average='macro')[0:2]                  # precision and recall-------------------#
        ac = (tp+tn) / (tn+tp+fp+fn)                                                # accuracy-------------------------------#
        sp = tn/(tn+fp)                                                             # specifcity, tnr------------------------#
        f1 = 2*pr*rc/(pr+rc)                                                        # f1 score-------------------------------#
        fpr, tpr, thresholds = roc_curve(labels_list, prob_list)                    # roc curve metrics----------------------#
        roc_auc = roc_auc_score(labels_list, prob_list)                             # auroc----------------------------------#
        run_time = time.time()-start_time                                           # run time-------------------------------#
        data_list = [tp, fp, tn, fn, ac, rc, pr, sp, f1, roc_auc, run_time, t_time] # classification metric summary----------#

        #--------ploting roc curve---------------------------------------------#
        if ml_data == "Test":
            plt.figure()
            plt.title("{} receiver operating characteristic".format(network_name))
            plt.plot(fpr, tpr, 'b', label = 'auc = %0.5f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('true positive rate')
            plt.xlabel('false positive rate')
            plt.savefig(fig_dir + network_name + ".png")
            plt.close()

        #--------output csv----------------------------------------------------#
        csv_writer.writerow(hyp_par + [ml_data] + data_list)

file_name = "{}FirstPaper_deep_3_model_results.csv".format(OUTPUT_DIR)
with open(file_name, mode="a+") as output_csv:
    CSV_WRITER = csv.writer(output_csv, delimiter=',')
    CSV_WRITER.writerow(["1st Hidden Units", "2nd Hidden Units", "Window Size",
                         "Padding", "Batch Size", "LR", "Epochs", "ML_Dataset",
                         "True Positives", "False Positives", "True Negatives",
                         "False Negatives", "Accuracy", "Recall", "Precision",
                         "Specificity", "F1 score", "AUROC", "Test Run Time (s)", "Train time (s)"])
for x in range(NUM_DATA):
    for BATCH_SIZE in BATCH_SIZE_LIST:
        TRAIN_DATASET = LoadData(TRAIN_DATA_FILE, TRAIN_LABEL_FILE)
        TRAIN_LOADER = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
        VALID_DATA = LoadData(VAL_DATA_FILE, VAL_LABEL_FILE)
        VAL_LOADER = DataLoader(dataset=VALID_DATA, batch_size=BATCH_SIZE, shuffle=False)
        TEST_DATA = LoadData(TEST_DATA_FILE, TEST_LABEL_FILE)
        TEST_LOADER = DataLoader(dataset=TEST_DATA, batch_size=BATCH_SIZE, shuffle=False)
        plt.ioff()

        if len(TRAIN_LOADER) < PPPE or len(VAL_LOADER) < PPPE or len(TEST_LOADER) < PPPE:
            print("Error: Plot points per epoch (PPPE) too big for current dataset length and batch size")
            sys.exit()

        for LEARNING_RATE in LR_LIST:
            NETWORK_NAME = "3CNN2FC_BS{}_LR{}_Padding{}_W{}".format(BATCH_SIZE,LEARNING_RATE,PADDING_LIST,window)
            FIGURE_DIR = OUTPUT_DIR + "Figures/" + NETWORK_NAME + "/"
            if os.path.isdir(FIGURE_DIR) == False:
                print("Making directory " + FIGURE_DIR)
                os.mkdir(FIGURE_DIR)

            for fir_cnn_hu in FIRST_LAYER_RANGE:
                for sec_cnn_hu in SECOND_LAYER_RANGE:
                    NETWORK = Net(fir_cnn_hu,sec_cnn_hu,window).to(DEVICE)
                    NETWORK.apply(weights_init)
                    LOSS_FUNCTION = nn.CrossEntropyLoss()
                    OPTIMIZER = optim.Adam(NETWORK.parameters(), lr=LEARNING_RATE)
                    ################################################################################
                    #--------Train the model-------------------------------------------------------#
                    ################################################################################
                    FIGURE_NAME = FIGURE_DIR + "{}_1_{}_2_{}_LogLoss.png".format(NETWORK_NAME,str(fir_cnn_hu),str(sec_cnn_hu))
                    TRAIN_MOV_AVG =[]
                    VAL_MOV_AVG = []
                    RUN_LOS_INTERVAL = int(len(TRAIN_LOADER)//PPPE)
                    START_TIME = time.time()
                    TRAIN_LOSS = AverageMeter('Train Loss', ':.4e')
                    VAL_LOSS = AverageMeter('Val Loss', ':.4e')
                    for EPOCH in range(1, EPOCH_NUM+1):
                        TEMP_MOV_T, TEMP_MOV_V = TrainModel(TRAIN_LOADER, VAL_LOADER, NETWORK, OPTIMIZER, DEVICE, EPOCH, LOSS_FUNCTION, RUN_LOS_INTERVAL, TRAIN_LOSS, VAL_LOSS)
                        TRAIN_MOV_AVG += TEMP_MOV_T
                        VAL_MOV_AVG += TEMP_MOV_V

                    TRAIN_TIME = str((time.time() - START_TIME))
                    print("Training Time: " + TRAIN_TIME)
                    print('Finished Training')
                    ################################################################################
                    #--------Plot the training loss over time--------------------------------------#
                    ################################################################################
                    MINI_BATCH_LIST = range(1, len(TRAIN_MOV_AVG)+1)
                    plt.figure()
                    plt.plot(MINI_BATCH_LIST, TRAIN_MOV_AVG, label="50-Data Training loss")
                    plt.plot(MINI_BATCH_LIST, VAL_MOV_AVG, label="50-Data Validation loss")
                    plt.legend()
                    plt.xlabel("Training examples")
                    plt.ylabel("Loss")
                    plt.title("Loss over training")
                    plt.savefig(FIGURE_NAME)
                    plt.close()
                    ################################################################################
                    #--------Test the model--------------------------------------------------------#
                    ################################################################################
                    with open(file_name, mode="a+") as output_csv:
                        CSV_WRITER = csv.writer(output_csv, delimiter=',')
                        DATA_LIST = {"Train": TRAIN_LOADER, "Validation": VAL_LOADER, "Test": TEST_LOADER}
                        H_P_LIST = [fir_cnn_hu, sec_cnn_hu, window, PADDING_LIST, BATCH_SIZE, LEARNING_RATE, EPOCH_NUM]
                        for DATA_SET in DATA_LIST:
                            print("Testing {} dataset...".format(DATA_SET))
                            RunDataset(DATA_LIST[DATA_SET],DATA_SET,DEVICE,NETWORK,CSV_WRITER,"{}_1_{}_2_{}_ROC".format(NETWORK_NAME,str(fir_cnn_hu),str(sec_cnn_hu)),FIGURE_DIR,TRAIN_TIME,H_P_LIST)
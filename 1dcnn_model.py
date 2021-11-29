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
train_d = "GenBank/" #or "RefSeq/"
test_d = "RefSeq/" #or "GenBank/"
encoding_type = "CAT" # or "One_hot"
input_rows = 21 # 4 if One_hot and 21 if CAT
input_len = 83 # 250 if One_hot and 83 if CAT
MODEL_NAME = "{}".format(encoding_type)
INPUT_TRAIN_DIR = "/home/esteban/Documents/School/Class_11785/Project/Data/NPY_Data/" + train_d + encoding_type +"/"
INPUT_TEST_DIR = "/home/esteban/Documents/School/Class_11785/Project/Data/NPY_Data/" + test_d + encoding_type +"/"
OUTPUT_DIR = "/home/esteban/Documents/School/Class_11785/Project/Data/NPY_Data/Results/"
TRAIN_DATA_FILE = INPUT_TRAIN_DIR + "train_data.npy"
TRAIN_LABEL_FILE = INPUT_TRAIN_DIR + "train_labels.npy"
VAL_DATA_FILE = INPUT_TRAIN_DIR + "val_data.npy"
VAL_LABEL_FILE = INPUT_TRAIN_DIR + "val_labels.npy"
TEST_DATA_FILE = INPUT_TEST_DIR + "val_data.npy"
TEST_LABEL_FILE = INPUT_TEST_DIR + "val_labels.npy"

PPPE = 10                                                                       # How many plot points per epoch do you want?
NUM_DATA = 5                                                                    # Number of runs through the same architecture
FIRST_LAYER_RANGE = [21]
SECOND_LAYER_RANGE = [16,32,64,128,256]
THIRD_LAYER_RANGE = [16,32,64,128,256]
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
EPOCH_NUM = 10                                                                  # For how many epochs do you want to run the data?
BATCH_SIZE_LIST = [128]                                                         # How big is the mini-batch size?
window = 1 # 3 if one_hot and 1 if CAT
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
    def __init__(self, input_rows, input_len, conv1_hu, conv2_hu, conv3_hu, window):
        super(Net, self).__init__()

        conv_input_rows = input_len # 250 for One_hot 83 for CAT
        conv_input_col = 3 # 4 for One_hot and 3 if CAT
        stride = 1
        padding = 0
        """ self.conv1 = nn.Conv1d(conv_input_col,conv1_hu,window,stride,padding)
        self.out_dim = int((conv_input_rows + 2*padding - window)/stride + 1)
        self.conv2 = nn.Conv1d(conv1_hu, conv2_hu,window,stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)
        self.conv3 = nn.Conv1d(conv2_hu, conv3_hu, window, stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)"""
        self.conv1 = nn.Conv2d(conv_input_col,conv1_hu,(21,window),stride,padding)
        self.out_dim = int((conv_input_rows + 2*padding - window)/stride + 1)
        self.conv2 = nn.Conv2d(conv1_hu, conv2_hu,window,stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)
        self.conv3 = nn.Conv2d(conv2_hu, conv3_hu, window, stride)
        self.out_dim = int((self.out_dim - window)/stride + 1)

        self.fc1 = nn.Linear(self.out_dim*conv3_hu, 8)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

################################################################################
#--------Load the appropriate data---------------------------------------------#
################################################################################
class LoadData(Dataset):
    def __init__(self, data, labels):
        label_dict = {"data":0,"Adenovirus":1,"Astrovirus":2,"Calicivirus":3,"Enterovirus":4,
                      "Hepatitis_A": 5, "Hepatitis_E":6, "Rotavirus":7,"Orthoreovirus":8}
        with open(data, mode='rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            data_x = np.load(f)
            i = 0
            while f.tell() < fsz:
                data_x = np.vstack((data_x, np.load(f)))                                                  # Load the examples-------------#
                i+=1
        data_y = np.load(labels)                                                # Load the labels---------------#
        #data_y = np.array([label_dict[species] for species in data_y.tolist()])
        self.data = torch.from_numpy(data_x).permute(0,1,3,2)                     # Convert examples to tensor----# (This should be permute(0,2,1) for One_hot)
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
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        roc_auc = roc_auc_score(labels_list, prob_list, multi_class="ovr")          # auroc----------------------------------#
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

file_name = "{}{}_results.csv".format(OUTPUT_DIR,MODEL_NAME)
with open(file_name, mode="a+") as output_csv:
    CSV_WRITER = csv.writer(output_csv, delimiter=',')
    CSV_WRITER.writerow(["1st Hidden Units", "2nd Hidden Units", "3rd Hidden Units",
                         "Window Size","Padding", "Batch Size", "LR", "Epochs",
                         "ML_Dataset","True Positives", "False Positives","True Negatives",
                         "False Negatives", "Accuracy", "Recall", "Precision",
                         "Specificity", "F1 score", "AUROC", "Test Run Time (s)", "Train time (s)"])
for x in range(NUM_DATA):
    for BATCH_SIZE in BATCH_SIZE_LIST:
        print("-"*20,"Loading Data", "-"*20)
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
            NETWORK_NAME = "{}_BS{}_LR{}_Padding{}_W{}".format(MODEL_NAME,BATCH_SIZE,LEARNING_RATE,PADDING_LIST,window)
            FIGURE_DIR = OUTPUT_DIR + "Figures/" + NETWORK_NAME + "/"
            if os.path.isdir(FIGURE_DIR) == False:
                print("Making directory " + FIGURE_DIR)
                os.mkdir(FIGURE_DIR)

            for fir_cnn_hu in FIRST_LAYER_RANGE:
                for sec_cnn_hu in SECOND_LAYER_RANGE:
                    for thir_cnn_hu in THIRD_LAYER_RANGE:
                        NETWORK = Net(input_rows, input_len,fir_cnn_hu,sec_cnn_hu,thir_cnn_hu,window).to(DEVICE)
                        NETWORK.apply(weights_init)
                        LOSS_FUNCTION = nn.CrossEntropyLoss()
                        OPTIMIZER = optim.Adam(NETWORK.parameters(), lr=LEARNING_RATE)
                        ################################################################################
                        #--------Train the model-------------------------------------------------------#
                        ################################################################################
                        FIGURE_NAME = FIGURE_DIR + "{}_2_{}_3_{}_LogLoss.png".format(NETWORK_NAME,str(sec_cnn_hu),str(thir_cnn_hu))
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
                            H_P_LIST = [fir_cnn_hu, sec_cnn_hu, thir_cnn_hu, window, PADDING_LIST, BATCH_SIZE, LEARNING_RATE, EPOCH_NUM]
                            for DATA_SET in DATA_LIST:
                                print("Testing {} dataset...".format(DATA_SET))
                                RunDataset(DATA_LIST[DATA_SET],DATA_SET,DEVICE,NETWORK,CSV_WRITER,"{}_2_{}_3_{}_ROC".format(NETWORK_NAME,str(sec_cnn_hu),str(thir_cnn_hu)),FIGURE_DIR,TRAIN_TIME,H_P_LIST)

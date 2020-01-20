import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from data_loader import loaddata, get_regoions, memd_all_process
from config import Config
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
import time
import random


batch_size = 8
device = torch.device("cuda")
device1 = torch.device('cuda')
N=4975
p_bernoulli = None

class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs=9950,
                 num_latent=4975, tied=True,
                 num_classes=2, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent

        self.fc_encoder = nn.Linear(num_inputs, num_latent)

        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)

        self.fc_encoder = nn.Linear(num_inputs, num_latent)

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.num_latent, 1),

            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.num_latent, 1),
            )


    def forward(self, x, eval_classifier=False):
        x = self.fc_encoder(x)
        x = torch.tanh(x)
        if eval_classifier:
            x_logit = self.classifier(x)
        else:
            x_logit = None

        if self.tied:
            x = F.linear(x, self.fc_encoder.weight.t())
        else:
            x = self.fc_decoder(x)

        return x, x_logit


def train(model, epoch, train_loader, p_bernoulli=None, mode='both', lam_factor=1.0):
    model.train()
    train_losses = []
    for i,(batch_x,batch_y) in enumerate(train_loader):
        if len(batch_x) != batch_size:
            continue
        if p_bernoulli is not None:
            if i == 0:
                p_tensor = torch.ones_like(batch_x).to(device1)*p_bernoulli
            rand_bernoulli = torch.bernoulli(p_tensor).to(device1)

        data, target = batch_x.to(device1), batch_y.to(device1)
        optimizer.zero_grad()

        if mode in ['both', 'ae']:
            if p_bernoulli is not None:
                rec_noisy, _ = model(data*rand_bernoulli, False)
                loss_ae = criterion_ae(rec_noisy, data) / len(batch_x)
            else:
                rec, _ = model(data, False)
                loss_ae = criterion_ae(rec, data) / len(batch_x)

        if mode in ['both', 'clf']:
            rec_clean, logits = model(data, True)
            loss_clf = criterion_clf(logits, target)

        if mode == 'both':
            loss_total = loss_ae + lam_factor*loss_clf
            train_losses.append([loss_ae.detach().cpu().numpy(),
                                 loss_clf.detach().cpu().numpy()])
        elif mode == 'ae':
            loss_total = loss_ae
            train_losses.append([loss_ae.detach().cpu().numpy(),
                                 0.0])
        elif mode == 'clf':
            loss_total = loss_clf
            train_losses.append([0.0,
                                 loss_clf.detach().cpu().numpy()])

        loss_total.backward()
        optimizer.step()

    return train_losses


def confusion(g_turth,predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth,predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)
    return accuracy,sensitivity,specificty


def test(model, criterion, test_loader, eval_classifier=False, num_batch=None):
    test_loss, n_test, correct = 0.0, 0, 0
    all_predss=[]
    if eval_classifier:
        y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for i,(batch_x,batch_y) in enumerate(test_loader, 1):
            if num_batch is not None:
                if i >= num_batch:
                    continue
            data = batch_x.to(device1)
            rec, logits = model(data, eval_classifier)

            test_loss += criterion(rec, data).detach().cpu().numpy()
            n_test += len(batch_x)
            print(i, n_test)
            if eval_classifier:
                proba = torch.sigmoid(logits).detach().cpu().numpy()
                preds = np.ones_like(proba, dtype=np.int32)
                preds[proba < 0.5] = 0
                all_predss.extend(preds)###????
                y_arr = np.array(batch_y, dtype=np.int32)

                correct += np.sum(preds == y_arr)
                y_true.extend(y_arr.tolist())
                y_pred.extend(proba.tolist())
                print('y_true and all_predss', y_true, all_predss)
        mlp_acc,mlp_sens,mlp_spef = confusion(y_true,all_predss)

    return  mlp_acc,mlp_sens,mlp_spef#,correct/n_test


def get_data(data, regoins):
    new_data = []
    for item in data:
        new_data.append(item[regoins])
    data = np.array(new_data)
    return data


data, label = loaddata(Config.All_Info, Config.CC200_nyu)
data = np.load('ohsu_19900.npy')
print(data.shape)
print(label)


for i in range(len(label)):
    if label[i] == 2:
        label[i]=0
#print(label)
'''data = torch.from_numpy(data)
data = data.float()'''
'''label = torch.from_numpy(label)
label = label.view(label.size()[0], 1)
label = label.float()'''
#print(data.shape, label.shape)

start = time.time()
kk=0
kf = StratifiedKFold(n_splits=10)
crossval_res_kol = []
accrancy = []
for kk, (train_index, test_index) in enumerate(kf.split(data, label)):
    print('============fold%d==========='%(kk+1))
    train_data, train_label = data[train_index], label[train_index]
    test_data, test_label = data[test_index], label[test_index]

    regoins = get_regoions(train_data, N)
    train_data = get_data(train_data, regoins)
    test_data = get_data(test_data, regoins)

    #train_data, train_label = memd_all_process(train_data, train_label)
    '''np.save('train_data_5970_%d.npy'%kk,arr=train_data)
    np.save('train_label_5970_%d.npy'%kk,arr=train_label)
    np.save('test_data_5970_%d.npy'%kk,arr=test_data)
    np.save('test_label_5970_%d.npy'%kk,arr=test_label)'''

    train_data = torch.FloatTensor(train_data)
    test_data = torch.FloatTensor(test_data)
    train_label = torch.FloatTensor(train_label)
    train_label = train_label.view(train_label.size()[0], 1)
    test_label = torch.FloatTensor(test_label)
    test_label = test_label.view(test_label.size()[0], 1)
    print('train_data and test_data shape:  ', train_data.shape, test_data.shape)

    verbose = (True if (kk == 0) else False)

    torch_train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(
        dataset=torch_train_dataset,  # torch TensorDataset format
        batch_size=8,  # mini batch size
        shuffle=True,  # random shuffle for training 随机洗牌训练
        num_workers=0,  # subprocesses for loading data
    )

    torch_test_dataset = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(
        dataset=torch_test_dataset,  # torch TensorDataset format
        batch_size=8,  # mini batch size
        shuffle=False,  # random shuffle for training 随机洗牌训练
        num_workers=0,  # subprocesses for loading data
    )

    model = MTAutoEncoder(tied=True, num_inputs=2*N, num_latent=N, use_dropout=False)
    model.to(device)

    criterion_ae = nn.MSELoss(reduction='sum')
    criterion_clf = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD([{'params': model.fc_encoder.parameters(), 'lr': 0.0001},
                           {'params': model.classifier.parameters(), 'lr': 0.0001}],
                          momentum=0.9)

    for epoch in range(1, 25+1):
        if epoch <= 20:
            train_losses = train(model, epoch, train_loader, p_bernoulli, mode='both')
        else:
            train_losses = train(model, epoch, train_loader, p_bernoulli, mode='clf')
        #print(epoch, train_losses, len(train_losses))

    res_mlp = test(model, criterion_ae, test_loader, eval_classifier=True)
    print(i+1,'fold',res_mlp)
    crossval_res_kol.append(res_mlp)
    accrancy.append(res_mlp[0])

print("averages:")
print(np.mean(np.array(crossval_res_kol), axis=0))
print('accrancy:')
print(accrancy)
finish = time.time()

print(finish-start)
print(list(model.parameters()))







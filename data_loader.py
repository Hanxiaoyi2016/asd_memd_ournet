import numpy as np
#from config import Config
import pandas as pd
from scipy.stats import pearsonr
import os
from sklearn import preprocessing

from scipy.io import loadmat
import numpy as np
from config import Config
import pandas as pd
from scipy.stats import pearsonr
import os
from sklearn import preprocessing
from memd import memd
import random
from keras.utils import np_utils
import heapq



N = 4975

def loaddata(subjects_info=Config.Train_Info,datasets_path=Config.CC200_nyu):
    csv_data=pd.read_csv(subjects_info,dtype={'FILE_ID':np.str,'DX_GROUP':np.int8})
    data_id=csv_data['FILE_ID']
    data_label=np.array(csv_data['DX_GROUP'])
    data=[]
    for i in data_id:
        temp=np.loadtxt(os.path.join(datasets_path,i+'_rois_cc200.1D'),dtype=np.float64)
        data.append(temp)
    data=np.array(data)
    return data,data_label

def separatedata(data, data_label):
    asd_data = []
    control_data = []
    for i in range(len(data_label)):
        if data_label[i] == 0:
            asd_data.append(data[i])
        else:
            control_data.append(data[i])
    asd_data = np.array(asd_data)
    control_data =np.array(control_data)
    return asd_data, control_data


def preprocess(data,normal_type):
    scale_data=[]
    if normal_type=='zsore':
        for temp in data:
            temp = preprocessing.StandardScaler().fit_transform(temp)
            scale_data.append(temp)
        scale_data=np.array(scale_data)
    if normal_type=='max-min':
        for temp in data:
            temp = preprocessing.MinMaxScaler().fit_transform(temp)
            scale_data.append(temp)
        scale_data=np.array(scale_data)
    return scale_data

def function_connectity(data):
    corr_matrix=[]
    for temp in data:
        i=0
        j=0
        corr=np.zeros((200,200))
        for series1 in temp.T:#u
            for series2 in temp.T:#v
                #print(pearsonr(series1,series2))
                #print(i)
                corr[i][j]=pearsonr(series1,series2)[0]
                j+=1
            i+=1    
            j=0  
        corr_matrix.append(corr)
    corr_matrix=np.array(corr_matrix)
    return corr_matrix

def get_mean(matrix):
    mean_matrix=np.zeros((200, 200))
    for temp in matrix:
        for i in range(200):
            for j in range(200):
                mean_matrix[i][j]+=temp[i][j]
    mean_matrix/=matrix.shape[0]
    return mean_matrix

def get_matrix(mean_matrix):
    matrix = []
    for i in range(1, 200):
        for j in range(0, i):
            matrix.append(mean_matrix[i][j])
    matrix = np.array(matrix)
    return matrix

def getmax(arr):
    max_member = np.argpartition(arr.ravel(), N-1)[-N:]#求1/4最大值/求3/8最大值(7463)
    #x_max, y_max = np.unravel_index(max_member, arr.shape)
    #print(max_member,x_max,y_max)

    min_member = np.argpartition(arr.ravel(), N-1)[:N]
    #x_min, y_min = np.unravel_index(min_member,arr.shape)#求1/4最小值/求3/8最小值
    #print(min_member,x_min,y_min)

    return max_member, min_member

def create_arr(matrix, max_member, min_member):
    new_matrix = []
    for temp in matrix:
        new_arr = []
        arr = get_matrix(temp)
        for i in range(N):
            new_arr.append(arr[max_member[i]])
        for j in range(N):
            new_arr.append(arr[min_member[j]])
        new_arr = np.array(new_arr)
        new_matrix.append(new_arr)
    new_matrix = np.array(new_matrix)
    return new_matrix

def memd_release(matrix):
    inp = matrix.T  # matrix.shape = (124,9950)
    imf = memd(inp, 160)  # 对训练数据进行memd
    print('判断有误nan数据',np.isnan(imf))
    return imf

'''def pre_memdprocess(data):
    #scale_data = preprocess(data, 'zsore')
    matrix = function_connectity(data)
    print('功能矩阵',matrix.shape)
    mean_matrix = get_mean(matrix)
    mean_matrix = get_matrix(mean_matrix)
    # print(matrix.shape)
    max_member, min_member = getmax(mean_matrix)
    # print(max_member.shape, min_member.shape)
    new_matrix = create_arr(matrix, max_member, min_member) #返回一个(N,9950)的矩阵
    new_matrix = memd_release(new_matrix) # 返回一个(num,N,9950)的矩阵
    return new_matrix'''


def memdprocess(matrix):# matrix.shape = (
    new_matrix = []
    for num in range(1000):#从53个样本中依次取出17个分量，17个分量相加得到一个新的data，总共需要1000个新数据
        new_datas = []
        for i in range(matrix.shape[0]):
            j = random.randint(0, matrix.shape[1]-1)
            new_datas.append(matrix[i][j])
        new_datas = np.array(new_datas)
        new_array = new_datas.sum(axis=0)
        new_matrix.append(new_array)
    new_matrix = np.array(new_matrix)
    return new_matrix

def process():
    asd_data = np.load('asd.npy')
    control_data = np.load('control.npy')
    # print(asd_data.shape, control_data.shape)
    asd_data = memdprocess(asd_data)
    control_data = memdprocess(control_data)
    print('最终向量', asd_data.shape, control_data.shape)
    return asd_data, control_data

def test_preprocess(data):
    # scale_data = preprocess(data, 'zsore')
    matrix = function_connectity(data)
    print('function connect matrix shape is:', matrix.shape)
    new_data = []
    for i in range(matrix.shape[0]):
        new_data.append(get_matrix(matrix[i]))
    data = np.array(new_data)
    return data
    #mean_matrix = get_mean(matrix)
    #mean_matrix = get_matrix(mean_matrix)
    # print(matrix.shape)
    #max_member, min_member = getmax(mean_matrix)
    #new_matrix = create_arr(matrix, max_member, min_member)  # 返回一个(N,9950)的矩阵
    #return mean_matrix

def memd_all_process(data, labels):
    asd_data, control_data = separatedata(data, labels)
    '''asd_data = test_preprocess(asd_data)
    control_data = test_preprocess(control_data)#9950矩阵'''
    asd_data = memd_release(asd_data)  # asd数据memd之后的imf数据
    control_data = memd_release(control_data)  # control数据memd之后的imf数据
    print('imf向量', asd_data.shape, control_data.shape)
    asd_data = memdprocess(asd_data)
    control_data = memdprocess(control_data)
    print('memd asd and control data', asd_data.shape, control_data.shape)
    asd_label = np.zeros(1000)
    control_label = np.ones(1000)
    data = np.vstack((asd_data, control_data))
    labels = np.hstack((asd_label, control_label))
    print('all_process', data.shape, labels.shape)
    return data, labels


def get_regoions(train_matrix, regnum):
    print('train_matrix', train_matrix.shape)
    avg = []
    for ie in range(train_matrix.shape[1]):
        avg.append(np.mean(train_matrix[:, ie]))
    avg = np.array(avg)
    highs = avg.argsort()[-regnum:][::-1]
    lows = avg.argsort()[:regnum][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions





'''test_process   memd_relase   process'''
if __name__ == '__main__':
    '''data, data_label = loaddata(Config.Train_Info, Config.CC200_nyu)
    asd_data, control_data = separatedata(data, data_label)
    #print(asd_data.shape, control_data.shape)
    asd_data = test_preprocess(asd_data)
    control_data = test_preprocess(control_data)
    control_data = np.delete(control_data, 18, 0)
    control_data = np.delete(control_data, 39, 0)
    print('9950矩阵',asd_data.shape, control_data.shape)
    np.save('asdnomend.npy', arr=asd_data)
    np.save('controlnomend.npy', arr=control_data)'''
    '''asd_data = np.load('asdnomend.npy')
    control_data = np.load('controlnomend.npy')
    print('9950矩阵', asd_data.shape, control_data.shape)


    asd_data = memd_release(asd_data)# asd数据memd之后的imf数据
    control_data = memd_release(control_data)# control数据memd之后的imf数据
    print('imf向量', asd_data.shape, control_data.shape)
    np.save('asd.npy', arr=asd_data)
    np.save('control.npy', arr=control_data)
    asd_data = np.load('asd.npy')
    control_data = np.load('control.npy')


    print('asd and control data',asd_data.shape, control_data.shape)
    asd_data, control_data = process()
    np.save('asd_memd.npy', arr=asd_data)
    np.save('control_memd.npy', arr=control_data)
    print('memd asd and control data',asd_data.shape,control_data.shape)'''
    '''以上是生成训练数据的代码'''

    '''validate_data, validata_label = loaddata(Config.Validate_Info, Config.CC200_nyu)
    validate_data = test_preprocess(validate_data)
    np.save('validate_data.npy',arr=validate_data)
    validate_data = np.load('validate_data.npy')
    print('validate', validate_data.shape)

    test_data, test_label = loaddata(Config.Test_Info, Config.CC200_nyu)
    test_data = test_preprocess(test_data)
    np.save('test_data.npy', arr=test_data)
    test_data = np.load('test_data.npy')
    print('testdata', test_data.shape)'''

    all_data, all_label = loaddata(Config.All_Info, Config.CC200_nyu)
    all_data = test_preprocess(all_data)
    np.save('ucla_19900.npy', arr=all_data)
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if np.isnan(all_data[i][j]):
                print(i)
    print(all_data.shape)









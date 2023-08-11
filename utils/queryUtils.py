# show images
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.transforms import CenterCrop, Resize, Compose, ToTensor
import pinecone
from .time_test import TestTime

cnt = 0
data_score_correct = []
display_freq = 1
timer = None
Inference = False  # for test
predict_result = 11


def get_result():
    return predict_result

def set_timer(timer_):
    global timer
    timer = timer_
    pass

def show(img, label):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
    ])

def add_into_pinecone(x, id:str, index, y=None):
    '''
    x: [3, 224, 224]
    y: [1]
    id: vecx
    加入pinecone数据库
    '''
    global cnt
    cnt = cnt + 1
    # 把多维tensor x 变成一维tensor
    x = x.flatten()
    # 把tensor x 变成list
    x = x.tolist()
    # 把多维tensor y 变成一维tensor
    y = y.flatten()
    # 把tensor y 变成list
    y = int(y)
    # 使用upsert方法插入pinecone数据库
    index.upsert([
        (str(cnt), x, {"label": y})
    ])
    print("cnt = ", cnt)
    print("add_into_pinecone success")
    pass

def debatch(x, y , id : str, batch_index : int = 0):
    # 这里需要保证index已经创建，否则会报错, 由于没钱，如果要创建index，需要先删除已有的index
    index = pinecone.Index("index-vec" + id)
    x = x.transpose(0, batch_index)
    print("after transpose, x.shape = ", x.shape)
    '''
    x: [batch_size, 3, 224, 224]
    y: [batch_size, 1]
    id: vecx
    batch_index: which dimension is batch_size
    '''
    for i in range(x.shape[0]):
        if(Inference):
            add_into_pinecone(x[i], id, index)
        else:
            add_into_pinecone(x[i], id, index, y[i])
    pass

def query_accuracy(x, y, id : str, batch_index : int = 0):
    '''
    x: [batch_size, 2048]
    y: [batch_size, 1]
    id: vecx
    batch_index: which dimension is batch_size

    this function is used to query the accuracy of the model
    fisrt, we need to query the similar vector of x in pinecone
    and then we need to compare the label of the similar vector and the label of x
    '''
    # 这里需要保证index已经创建且插入了数据，否则会报错或者无意义
    index = pinecone.Index("index-vec" + id)
    x = x.transpose(0, batch_index)
    for i in range(x.shape[0]):
        if(Inference):
            query_in_pinecone(x[i], id, index)
        else:
            query_in_pinecone(x[i], id, index, y[i])
    pass

def query_in_pinecone(x, id:str, index, y = None):
    '''
    we need the score, label, and metadata of the similar vector, 
    so we need to set include_values=True, include_metadata=True
    and then we need to compare the label of the similar vector and the label of x,
    if they are the same, then the query is correct, otherwise, the query is wrong, 
    we should count the number of correct query
    and then we can get the accuracy of the model

    so first, label, second, score, third, vector
    '''
    result = index.query(
        vector=x.flatten().tolist(),
        top_k=1,
        include_values=True,
        include_metadata=True
    )   
    data_score_correct.append([result['matches'][0]['score'], result['matches'][0]['metadata']['label'] == y])
    if(Inference):
        global predict_result
        predict_result = result['matches'][0]['metadata']['label']
        return
    global cnt
    cnt = cnt + 1
    if(cnt % display_freq == 0):
        print("the similarity score : ", result['matches'][0]['score'])
        print("the predict label : ", result['matches'][0]['metadata']['label'])
        print("the true label : ", y)
        if result['matches'][0]['metadata']['label'] == y:
            print("query is correct")
        else:
            print("query is wrong")
        print("*****************************************")
    # # 等待用户按下任意键继续
    # input("Press Enter to continue...")

    pass


def cal_accuracy():
    correct = 0
    for i in range(len(data_score_correct)):
        data_score_correct[i][1] = int(data_score_correct[i][1])
        if(data_score_correct[i][1] == 1):
            correct = correct + 1
    print(f"*********the total accuracy is {correct / len(data_score_correct)}**************")

def set_display_freq(freq):
    global display_freq
    display_freq = freq
    pass
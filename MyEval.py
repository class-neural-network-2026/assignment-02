import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from MyModel import MyModel


class MyEval():
    def __init__(self):
        pass

    # value range of prediction: 0 ~ 1
    # value range of label: 0 or 1
    # accuracy range: 0 ~ 1 
    def compute_accuracy(self, prediction, label):
        pred_label = torch.argmax(prediction, dim=1)
        pred_label = pred_label.to(torch.uint8)
        label      = label.to(torch.uint8)
        bCorrect    = (pred_label == label)
        accuracy    = bCorrect.sum() / len(label)
        # print(f'prediction: {pred_label}')
        # print(f'label: {label}')
        # print(f'correct: {bCorrect}, accuracy: {accuracy}')
        return accuracy
        
if __name__ == '__main__':
    dataset     = MyDataset(path='data', split='train')
    dataloader  = DataLoader(dataset, batch_size=8, shuffle=True)
    iter_data   = iter(dataloader)
    data, label = next(iter_data)
    vec         = nn.Flatten()(data)

    model = MyModel()
    model.eval()
    
    y       = model(vec)
    eval    = MyEval()
    acc     = eval.compute_accuracy(y, label)
    
    print(f'dataset: {len(dataset)}')
    print(f'data: {data.shape}, label: {label.shape}, vec: {vec.shape}')
    print(f'input: {vec.shape}, prediction: {y.shape}, label: {label.shape}, accuracy: {acc}')
    print(f'prediction: {y}, label: {label}')
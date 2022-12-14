"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch
import torch.nn as nn

from data_load import data_load
from config import *
from models.build_model import build_model

def train(model, optimizer, criterion, train_iter, device):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, criterion, val_iter, device):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = criterion(logit, y)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

def main():
    TEXT, train_iter, val_iter, test_iter = data_load()
    model = build_model(TEXT, MODEL_DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, DEVICE, DROPOUT_RATIO)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #best_val_loss = None
    for e in range(1, EPOCHS+1):
        train(model, optimizer, criterion, train_iter, DEVICE)
        val_loss, val_accuracy = evaluate(model, criterion, val_iter, DEVICE)

        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

        """
        # 검증 오차가 가장 적은 최적의 모델을 저장
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), 'D:/IIPL_internship/Week1/snapshot/txtclassification.pt')
            best_val_loss = val_loss
        """

    test_loss, test_accuracy = evaluate(model, criterion, test_iter, DEVICE)
    print("test loss : %5.2f | test accuracy : %5.2f" % (test_loss, test_accuracy))

if __name__ == "__main__":
    main()
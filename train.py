import math
import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from Model import Transformer
from Dataloader import Dataloader

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=2):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    start = time.time()
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        model.zero_grad()
        # leave out the last <EOS>
        out = model(src_batch, tgt_batch[:, :-1])   
        tgt_words = tgt_batch[:,1:].contiguous().view(-1)
        nllloss = criterion(out, tgt_words) / tgt_batch.size(0)
        nllloss.backward()
        clip_grad_norm(model.parameters(), max_norm=5)
        optim.step()
        preds = torch.max(out,1)[1]
        corrects = preds.data.eq(tgt_words.data).masked_select(tgt_words.data.ne(0))  
        batch_loss += nllloss.data[0]
        batch_words += len(corrects)
        batch_corrects += corrects.sum()
        if (i+1)%print_batch==0 or (i+1)==len(dataloader):
            print("Epoch %2d, Batch %6d/%6d, Acc: %6.2f, Plp: %8.2f, %4.0f seconds" % 
                 (epoch+1, i+1, len(dataloader), batch_corrects/batch_words, 
                  math.exp(batch_loss/batch_words), time.time()-start))
            epoch_loss += batch_loss
            epoch_words += batch_words
            epoch_corrects += batch_corrects
            batch_loss, batch_words, batch_corrects = 0, 0, 0
            start = time.time()
    epoch_accuracy = epoch_corrects/epoch_words
    epoch_perplexity = math.exp(epoch_loss/epoch_words)
    return epoch_accuracy, epoch_perplexity

if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    
    print("Building Dataloader ...")
    train_path = "/home/ubuntu/translation-data/dev."
    dataloader = Dataloader(train_path+"de.id", train_path+"en.id", 64, cuda=True)
    
    print("Building Model ...")
    model = Transformer(32000, 8, 512, 0.1, 1024).cuda()
    nllloss_weights = torch.ones(32000)
    nllloss_weights[0] = 0      
    criterion = nn.NLLLoss(nllloss_weights, size_average=False).cuda()
    optim = torch.optim.SGD(model.parameters(), lr=1)
    
    print("Start Training ...")
    for epoch in range(10):
        if epoch > 1:
            dataloader.shuffle()
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, dataloader, optim)
        print("[Train] Accuracy: %6.2f, Perplexity: %6.2f" % (train_acc, train_ppl))
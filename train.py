import math
import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from Model import Transformer
from Dataloader import Dataloader

def trainEpoch(epoch, model, criterion, dataloader, optim, print_every_iter=2):
    model.train()
    epoch_loss, epoch_words, epoch_correctpreds = 0, 0, 0
    iter_loss, iter_words, iter_correctpreds = 0, 0, 0
    start = time.time()
    for i, (src_batch, tgt_batch) in enumerate(dataloader):
        model.zero_grad()
        # leave out the last <EOS>
        out = model(src_batch, tgt_batch[:, :-1])   
        out = out.view(-1, BPE_SIZE)
        tgt_words = tgt_batch[:,1:].contiguous().view(-1)
        nllloss = criterion(out, tgt_words) / tgt_batch.size(0)
        nllloss.backward()
        clip_grad_norm(model.parameters(), max_norm=5)
        optim.step()
        preds_id = torch.max(out,1)[1]
        correctpreds = preds_id.data.eq(tgt_words.data)
        # ignore preds at paddings 
        correctpreds = correctpreds.masked_select(tgt_words.data.ne(0))  
        epoch_loss += nllloss.data[0]
        epoch_words += len(correctpreds)
        epoch_correctpreds += correctpreds.sum()
        iter_loss += nllloss.data[0]
        iter_words += len(correctpreds)
        iter_correctpreds += correctpreds.sum()
        if (i+1) % print_every_iter == 0:
            print("Epoch %2d, Iter %6d/%6d, Acc: %6.2f, Perplexity: %8.2f, %4.0f seconds" % 
                 (epoch+1, i+1, len(dataloader), iter_correctpreds/iter_words, 
                  math.exp(iter_loss/iter_words), time.time()-start))
            iter_loss, iter_words, iter_correctpreds = 0, 0, 0
            start = time.time()
    epoch_accuracy = epoch_correctpreds/epoch_words
    epoch_perplexity = math.exp(epoch_loss/epoch_words)
    return epoch_accurary, epoch_perplexity


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    
    print("Building Dataloader ...")
    dataloader = Dataloader("/home/ubuntu/translation-data/dev.de.id",
                  "/home/ubuntu/translation-data/dev.en.id", 32)
    
    print("Building Model ...")
    model = Transformer(32000, 8, 512, 0.1, 512)
    nllloss_weights = torch.ones(32000)
    nllloss_weights[0] = 0      
    criterion = nn.NLLLoss(nllloss_weights, size_average=False)
    optim = torch.optim.SGD(model.parameters(), lr=1)
    
    print("Start Training ......")
    optim = torch.optim.SGD(model.parameters(), lr=1.0)
    for epoch in range(10):
        if epoch > 1:
            dataloader.shuffle()
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, dataloader, optim)
        print("[Train] Accuracy: %6.2f, Perplexity: %6.2f" % (train_acc, train_ppl))
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch, 'optimizer': optim}
        torch.save(checkpoint, 'epoch%d_acc_%.2f_ppl_%.2f.pt' % (epoch, 100*val_acc, val_ppl))
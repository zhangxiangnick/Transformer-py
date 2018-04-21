import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from Model import Transformer
from Dataloader import Dataloader
from Optimizer import TransformerOptimizer

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=20):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    batch_size = dataloader.batch_size
    start = time.time()
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        model.zero_grad()
        # leave out the last <EOS> in target
        out, _ = model(src_batch, tgt_batch[:,:-1])  
        # label smoothing 
        # randomly set 10% target labels to 0, which doesn't contribute to loss
        labelsmoothing_mask = torch.le(torch.zeros(tgt_batch[:,1:].size()).uniform_(), 0.1).cuda()
        tgt_words = tgt_batch[:,1:].data.clone().masked_fill_(labelsmoothing_mask, 0)
        tgt_words = Variable(tgt_words.contiguous().view(-1))    
        loss = criterion(out, tgt_words) / batch_size   
        loss.backward()
        optim.step()
        preds = torch.max(out,1)[1]        
        corrects = preds.data.eq(tgt_words.data).masked_select(tgt_words.data.ne(0))          
        batch_loss += loss.data[0]     
        batch_words += len(corrects)      
        batch_corrects += corrects.sum()
        if (i+1)%print_batch==0 or (i+1)==len(dataloader):
            print("Epoch %2d, Batch %6d/%6d, Acc: %6.2f, Plp: %8.2f, %4.0f seconds" % 
                 (epoch+1, i+1, len(dataloader), batch_corrects/batch_words, 
                  math.exp(batch_loss*batch_size/batch_words), time.time()-start))
            epoch_loss += batch_loss
            epoch_words += batch_words
            epoch_corrects += batch_corrects
            batch_loss, batch_words, batch_corrects = 0, 0, 0
            start = time.time()
    epoch_accuracy = epoch_corrects/epoch_words
    epoch_perplexity = math.exp(epoch_loss*batch_size/epoch_words)
    return epoch_accuracy, epoch_perplexity

def evaluate(epoch, model, criterion, dataloader):
    model.eval()
    eval_loss, eval_words, eval_corrects = 0, 0, 0
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        out, _ = model(src_batch, tgt_batch[:, :-1])
        tgt_words = tgt_batch[:,1:].contiguous().view(-1)      
        loss = criterion(out, tgt_words)    
        preds = torch.max(out,1)[1]        
        corrects = preds.data.eq(tgt_words.data).masked_select(tgt_words.data.ne(0))          
        eval_loss += loss.data[0]     
        eval_words += len(corrects)      
        eval_corrects += corrects.sum()
    eval_accuracy = eval_corrects/eval_words
    eval_perplexity = math.exp(eval_loss/eval_words)
    return eval_accuracy, eval_perplexity

if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    
    print("Building Dataloader ...")
    train_path = "/home/ubuntu/translation-data/train."
    traindataloader = Dataloader(train_path+"en.id", train_path+"de.id", 96, cuda=True)
    dev_path = "/home/ubuntu/translation-data/dev."
    devdataloader = Dataloader(dev_path+"en.id", dev_path+"de.id", 96, cuda=True, volatile=True)    
    
    print("Building Model ...")
    model = Transformer(bpe_size=32000, h=8, d_model=512, p=0.1, d_ff=1024).cuda()
    nllloss_weights = torch.ones(32000)   
    criterion = nn.NLLLoss(nllloss_weights, size_average=False, ignore_index=0).cuda()
    base_optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optim = TransformerOptimizer(base_optim, warmup_steps=32000, d_model=512)

    print("Start Training ...")
    for epoch in range(60):
        if epoch > 0:
            traindataloader.shuffle(1024)
        if epoch == 20:
            optim.init_lr = 0.5 * optim.init_lr 
        if epoch == 40:
            optim.init_lr = 0.1 * optim.init_lr 
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, traindataloader, optim)
        print("[Train][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, train_acc, train_ppl))
        eval_acc, eval_ppl = evaluate(epoch, model, criterion, devdataloader)
        print("[Eval][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, eval_acc, eval_ppl))
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch, 'optimizer': optim}
        torch.save(checkpoint, 'epoch%d_acc_%.2f_ppl_%.2f.pt' % (epoch+1, 100*eval_acc, eval_ppl))

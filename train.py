import math
import time
import torch
import torch.nn as nn
from Model import Transformer
from Dataloader import Dataloader
from Optimizer import TransformerOptimizer

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=10):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    batch_size = dataloader.batch_size
    start = time.time()
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        model.zero_grad()
        # leave out the last <EOS>
        out = model(src_batch, tgt_batch[:, :-1])   
        tgt_words = tgt_batch[:,1:].contiguous().view(-1)      
        loss = criterion(out, tgt_words)    
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
        out = model(src_batch, tgt_batch[:, :-1])
        # length of tgt as training input
        len_tgt_train = tgt_batch.size(1) - 1   
        # calculate at most 32 positions per slice to reduce memory usuage
        slice_size = 32
        loss = 0
        for start_pos in range(0, len_tgt_train, slice_size):
            slice_out = out[:, start_pos:(start_pos+slice_size), :]
            slice_out = model.generator(slice_out)        
            slice_out = model.logsoftmax(slice_out.view(-1, model.bpe_size))
            # shift tgt by one for loss calculation
            tgt_words = tgt_batch[:, (start_pos+1):(start_pos+1+slice_size)].contiguous().view(-1) 
            loss = criterion(slice_out, tgt_words) 
            preds = torch.max(slice_out,1)[1]
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
    nllloss_weights[0] = 0      
    criterion = nn.NLLLoss(nllloss_weights, size_average=False).cuda()
    base_optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optim = TransformerOptimizer(base_optim, warmup_steps=16000, d_model=512)

    print("Start Training ...")
    for epoch in range(10):
        if epoch > 0:
            traindataloader.shuffle(1024)
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, traindataloader, optim)
        print("[Train][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, train_acc, train_ppl))
        eval_acc, eval_ppl = evaluate(epoch, model, criterion, devdataloader)
        print("[Eval][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, eval_acc, eval_ppl))
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch, 'optimizer': optim}
        torch.save(checkpoint, 'epoch%d_acc_%.2f_ppl_%.2f.pt' % (epoch+1, 100*eval_acc, eval_ppl))
import math
import torch
from torch.autograd import Variable

MAX_WORDPIECE_LEN = 512

class Dataloader(object):
    """Class to Load Language Pairs and Make Batch

    Args:
        path: path to the output of Google SentencePiece

    """   
    def __init__(self, path, batch_size, cuda=False, volatile=False):
        # Need to reload every time because memory error in pickle
        srcFile = open(path + "train.de.id")
        tgtFile = open(path + "train.en.id")
        src = []
        tgt = []
        nb_pairs = 0
        while True:
            src_line = srcFile.readline()
            tgt_line = tgtFile.readline()        
            if src_line=='' and tgt_line=='':
                break            
            src_ids = list(map(int, src_line.strip().split()))
            tgt_ids = list(map(int, tgt_line.strip().split()))
            if len(src_ids)<=256 and len(tgt_ids)<=256:
                src.append(src_ids)
                tgt.append(tgt_ids)  
                nb_pairs += 1
        print('%d pairs are converted in the data' %nb_pairs)
        srcFile.close()
        tgtFile.close()
        self.src = src
        self.tgt = tgt
        self.batch_size = batch_size
        self.nb_pairs = nb_pairs
        self.nb_batches = math.ceil(nb_pairs/batch_size)
        self.cuda = cuda
        self.volatile = volatile
        
    def __len__(self):
        return self.nb_batches  
    
    def shuffle(self):
        ids = torch.permute(nb_pairs)
        self.src = [self.src[i] for i in ids]
        self.tgt = [self.tgt[i] for i in ids] 
        
    def __getitem__(self, index): 
    """Generate the index-th batch

    Returns:
        (torch.LongTensor, torch.LongTensor)

    """
        def wrap(sentences):
            max_size = max([len(s) for s in sentences])
            out = [s + [0]*(max_size-len(s)) for s in sentences]
            out = torch.LongTensor(out)
            if self.cuda:
                out = out.cuda()
            return Variable(out, volatile=self.volatile)

        src_batch = self.src[index*self.batch_size:(index+1)*self.batch_size]
        tgt_batch = self.tgt[index*self.batch_size:(index+1)*self.batch_size]        
        return wrap(src_batch), wrap(tgt_batch)

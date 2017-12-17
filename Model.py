import numpy as np
import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer, PositionalEncoding

class Transformer(nn.Module):
    """Main model in 'Attention is all you need'
    
    Args:
        bpe_size:   vocabulary size from byte pair encoding
        h:          number of heads
        d_model:    dimension of model
        p:          dropout probabolity 
        d_ff:       dimension of feed forward
        
    """
    def __init__(self, bpe_size, h, d_model, p, d_ff):
        super(Transformer, self).__init__()
        self.bpe_size = bpe_size
        self.word_emb = nn.Embedding(bpe_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, p)
        self.encoder = nn.ModuleList([EncoderLayer(h, d_model, p, d_ff) for _ in range(6)]) 
        self.decoder = nn.ModuleList([DecoderLayer(h, d_model, p, d_ff) for _ in range(6)])
        self.generator = nn.Linear(d_model, bpe_size, bias=False)
        # tie weight between word embedding and generator 
        self.generator.weight = self.word_emb.weight
        self.logsoftmax = nn.LogSoftmax()
        # pre-save a mask to avoid future information in self-attentions in decoder
        # save as a buffer, otherwise will need to recreate it and move to GPU during every call
        mask = torch.ByteTensor(np.triu(np.ones((512,512)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
        
    def encode(self, src):
        context = self.word_emb(src)                                  # batch_size x len_src x d_model
        context = self.pos_emb(context)
        mask_src = src.data.eq(0).unsqueeze(1)                                       
        for _, layer in enumerate(self.encoder):                          
            context = layer(context, context, context, mask_src)      # batch_size x len_src x d_model
        return context, mask_src
    
    def decode(self, tgt, context, mask_src):
        out = self.word_emb(tgt)                                      # batch_size x len_tgt x d_model
        out = self.pos_emb(out)
        len_tgt = tgt.size(1)
        mask_tgt = tgt.data.eq(0).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        for _, layer in enumerate(self.decoder):
            out, coverage = layer(out, out, out, context, mask_tgt, mask_src)  # batch_size x len_tgt x d_model   
        out = self.generator(out)                                              # batch_size x len_tgt x bpe_size
        out = self.logsoftmax(out.view(-1, self.bpe_size))          
        return out, coverage
        
    def forward(self, src, tgt):
        """
        Inputs Shapes: 
            src: batch_size x len_src  
            tgt: batch_size x len_tgt
        
        Outputs Shapes:
            out:      batch_size*len_tgt x bpe_size
            coverage: batch_size x len_tgt x len_src
            
        """
        context, mask_src = self.encode(src)
        out, coverage = self.decode(tgt, context, mask_src)            
        return out, coverage
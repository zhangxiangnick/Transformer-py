import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer, PositionalEncoding

class Transformer(nn.Module):
    """Main model in 'Attention is all you need'
    
    Args:
        bpe_size:   size of byte pair encoding
        h:          number of heads
        d_model:    dimension of model
        p:          dropout probabolity 
        d_ff:       dimension of feed forward
        
    Inputs Shapes: 
        src: batch_size x len_src  
        tgt: batch_size x len_tgt
        
    Outputs Shapes:
        out:   batch_size x len_tgt x bpe_size
    """
    def __init__(self, bpe_size, h, d_model, p, d_ff):
        super(Transformer, self).__init__()
        self.word_emb = nn.Embedding(bpe_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, p)
        self.encoder = nn.ModuleList([EncoderLayer(h, d_model, p, d_ff) for i in range(6)])     
        
    def forward(self, src, tgt):
        batch_size, len_src = src.size()
        out = self.word_emb(src)          # batch_size x len_src x d_model
        out = self.pos_emb(out)
        for _, layer in enumerate(self.encoder):
            out = layer(out, out, out, src.eq(0).unsqueeze(1))
        return out
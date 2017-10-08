import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """
    Applies layer normalization to last dimension
    """
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta


class MultiHeadAttention(nn.Module):
    """
    Applies multi-head attentions to inputs (query, key, value)

    Args:
        h:       number of heads
        d_model: dimention of model
        p:       dropout probabolity  
        
    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)
        
    Inputs Shapes: 
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
        
    Outputs Shapes:
        out:   batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p):
        super(MultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        self.d_head = d_model//h
        self.fc_query = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_key = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_value = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_concat = nn.Linear(h*self.d_head, d_model, bias=False)
        self.sm = nn.Softmax()
        self.dropout = nn.Dropout(p)
        self.layernorm = LayerNorm(d_model)
      
    def _prepare_proj(self, x):
        """
        Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head).transpose(1,2).contiguous().view(b*self.h, l, self.d_head)
        
    def forward(self, query, key, value, mask):
        b, len_query = query.size(0), query.size(1)
        len_key = key.size(1)
        
        # project inputs to multi-heads
        proj_query = self.fc_query(query)       # batch_size x len_query x h*d_head
        proj_key = self.fc_key(key)             # batch_size x len_key x h*d_head
        proj_value = self.fc_value(value)       # batch_size x len_key x h*d_head
        
        # prepare the shape for applying softmax
        proj_query = self._prepare_proj(proj_query)  # batch_size*h x len_query x d_head
        proj_key = self._prepare_proj(key)           # batch_size*h x len_key x d_head
        proj_value = self._prepare_proj(value)       # batch_size*h x len_key x d_head
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(proj_query, proj_key.transpose(1,2))  # batch_size*h x len_query x len_key
        attns = attns / math.sqrt(self.d_head) 
        attns = attns.view(b, self.h, len_query, len_key) 
        attns = attns.masked_fill_(mask.unsqueeze(1), -float('inf'))
        attns = self.sm(attns.view(-1, len_key)).view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, proj_value)      # batch_size*h x len_query x d_head
        out = out.view(b, self.h, len_query, self.d_head).transpose(1,2).contiguous() 
        out = self.fc_concat(out.view(b, len_query, self.h*self.d_head))
        out = self.layernorm(query + self.dropout(out))   
        return out

    
class FeedForward(nn.Module):
    """
    Applies position-wise feed forward to inputs
    
    Args:
        d_model: dimension of model 
        d_ff:    dimension of feed forward
        p:       dropout probabolity 
        
    Params:
        fc_1: FC layer from d_model to d_ff
        fc_2: FC layer from d_ff to d_model
        
    Input Shapes:
        input: batch_size x len x d_model
        
    Output Shapes:
        out: batch_size x len x d_model
    """
    
    def __init__(self, d_model, d_ff, p):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p)
        self.layernorm = LayerNorm(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        out = self.dropout(self.fc_2(self.relu(self.fc_1(input))))
        out = self.layernorm(out + input)
        return out

    
class EncoderLayer(nn.Module):
    """
    Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimention of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer
    
    Input Shapes:
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p, d_ff):
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(h, d_model, p)
        self.feedforward = FeedForward(d_model, d_ff, p)
    
    def forward(self, query, key, value, mask):
        out = self.multihead(query, key, value, mask)
        out = self.feedforward(out)
        return out
    
    
class DecoderLayer(nn.Module):
    """
    Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    
    Args:
        h:       number of heads
        d_model: dimention of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer        
        feedforward:    feed forward layer
    
    Input Shapes:
        query:    batch_size x len_query x d_model 
        key:      batch_size x len_key x d_model   
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable 
        mask_src: batch_size x len_query x len_src or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """    
    
    def __init__(self, h, d_model, p, d_ff):
        super(DecoderLayer, self).__init__()
        self.multihead_tgt = MultiHeadAttention(h, d_model, p)
        self.multihead_src = MultiHeadAttention(h, d_model, p)
        self.feedforward = FeedForward(d_model, d_ff, p)
    
    def forward(self, query, key, value, context, mask_tgt, mask_src):
        out = self.multihead_tgt(query, key, value, mask_tgt)
        out = self.multihead_src(out, context, context, mask_src)
        out = self.feedforward(out)
        return out
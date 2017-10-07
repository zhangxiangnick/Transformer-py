import torch
import torch.nn as nn
import math

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
        mask:  batch_size x len_key
        
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
      
    def _prepare_proj(self, x):
        """
        Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head).transpose(1,2).contiguous().view(b*self.h, l, self.d_head)
        
    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        len_query = query.size(1)
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
        if mask is not None:
            attns = attns.view(b, self.h, len_query, len_key) 
            attns = attns.masked_fill_(mask.unsqueeze(1).unsqueeze(1), -float('inf'))
        attns = self.sm(attns.view(-1, len_key)).view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, proj_value)      # batch_size*h x len_query x d_head
        out = out.view(b, self.h, len_query, self.d_head).transpose(1,2).contiguous() 
        out = self.fc_concat(out.view(b, len_query, self.h*self.d_head))
        out = query + self.dropout(out) 
        
        # TODO: layer normalization
        
        return out
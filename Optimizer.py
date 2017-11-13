class TransformerOptimizer(object):
    """Optimizer in 'Attentions is all you need' 
    
    Args:
        optimizer: Adam optimizer specified in the paper
        warmup_steps: warmup steps for updating learning rate as in the paper
        d_model: dimension of Transformer model
    """
    def __init__(self, optimizer, warmup_steps, d_model=512):
        self.base_optim = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = d_model**(-0.5) 
        self.lr = 0
        self._step = 0
        
    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            self.lr = self.init_lr*self._step*self.warmup_steps**(-1.5)
        else:
            self.lr = self.init_lr*self._step**(-0.5)
        for param_group in self.base_optim.param_groups:
            param_group['lr'] = self.lr                            
        self.base_optim.step()
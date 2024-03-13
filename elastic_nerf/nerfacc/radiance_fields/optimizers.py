import math
from torch.optim import Adam


class ElasticAdam(Adam):
    def __init__(
        self,
        **kwargs
    ):
        super(ElasticAdam, self).__init__(**kwargs)

    def step(self, active_width=None, closure=None):
        if active_width is None:
            # If active_width is not provided, fallback to the standard Adam step
            return super(ElasticAdam, self).step(closure)
        
        # Compute the elastic learning rate based on mu-parameterization.
        # Make a copy of the learning rate to avoid modifying the original.
        
        base_lrs = {}
        for group in self.param_groups:
            if 'elastic' in group:
                elastic_group_name = group['elastic']
                base_lrs[elastic_group_name] = float(group['lr'])
                
                # Compute the elastic learning rate based on the mu-parameterization.
                mu = elastic['mu']
                lr = group['lr']
                elastic_lr = lr * (1 + mu * (active_width - 1) / self.max_width)
                group['lr'] = elastic_lr

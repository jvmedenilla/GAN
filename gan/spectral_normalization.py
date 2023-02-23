import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        """
        Reference:
        SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1802.05957.pdf
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        """
        TODO: Implement Spectral Normalization
        Hint: 1: Use getattr to first extract u, v, w.
              2: Apply power iteration.
              3: Calculate w with the spectral norm.
              4: Use setattr to update w in the module.
        """
        
        u = getattr(self.module, self.name + '_u')
        u_flat = u
        u_height = u_flat.size(0)
        u_flat = u_flat.reshape(u_height, -1)
        v = getattr(self.module, self.name + '_v')
        v_flat = v
        v_height = v_flat.size(0)
        v_flat = v_flat.reshape(v_height, -1)
        #print(u, v)
        w = getattr(self.module, self.name + '_bar')
        w_flat = w
        height = w_flat.size(0)
        w_flat = w_flat.reshape(height, -1)
        
        u = getattr(self.module, self.name + "_u")  # shape = [128]
        #u = u.reshape(u.data.shape[0],1)
        v = getattr(self.module, self.name + "_v")  # shape = [48]
        #v = v.reshape(v.data.shape[0],1)
        w = getattr(self.module, self.name + "_bar")
        w_flat = w
        height = w_flat.data.shape[0]
        #w_flat = w_flat.reshape(height, -1) # shape = [128,48]
        #w_flat_t = w_flat.T # shape = [48,128]
        #print(u.shape, v.shape, w_flat_t.shape)

        #size = w_flat.data.shape[0]
    
        for i in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w_flat.view(height, -1).data), u.data), dim=0)
            u.data = F.normalize(torch.mv(w_flat.view(height, -1).data, v.data), dim=0)

        sigma = u.dot(w_flat.view(height, -1).mv(v))
        setattr(self.module, self.name, w_flat / sigma.expand_as(w_flat))
        
        #for i in range(self.power_iterations):
        #    #print(w_flat.shape, u.shape)
        #    v = Tensor.dot(w_flat_t, u)/(torch.linalg.norm(Tensor.dot(w_flat_t, u), dim=0, ord=2))
        #    u = Tensor.dot(w_flat,v)/(torch.linalg.norm(Tensor.dot(w_flat,v), dim=0, ord=2))
#
        #w_spec = w/(torch.dot(u.T ,torch.dot(w, v)))
        #setattr(self.module, self.nam, w_spec)
        #w_spec = w_mat.view(size, -1)/(torch.dot(u_mat.T ,torch.dot(w_mat.view(size, -1), v_mat)))
        #setattr(self.module, self.name, w_spec)
        
        #        v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
        #        u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        #    if self.n_power_iterations > 0:
        #        # See above on why we need to clone
        #        u = u.clone(memory_format=torch.contiguous_format)
        #        v = v.clone(memory_format=torch.contiguous_format)
#
        #sigma = torch.dot(u, torch.mv(weight_mat, v))
        #weight = weight / sigma
        #return weight

    
    def _make_params(self):
        """
        No need to change. Initialize parameters.
        v: Initialize v with a random vector (sampled from isotropic distrition).
        u: Initialize u with a random vector (sampled from isotropic distrition).
        w: Weight of the current layer.
        """
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        """
        No need to change. Update weights using spectral normalization.
        """
        self._update_u_v()
        return self.module.forward(*args)


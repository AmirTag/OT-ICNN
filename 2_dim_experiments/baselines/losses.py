import torch
import utils
from torch.autograd import Variable

def calc_cosine(x, approx_fx, fx):
    a = approx_fx - x
    b = fx - x
    num = (a * b).sum(dim=1)
    denom = torch.norm(a, 2, 1)*torch.norm(b, 2, 1)
    return torch.mean(num/denom)

def calc_l2(x, approx_fx, fx):
    a = approx_fx - x
    b = fx - x
    nm = torch.norm(a - b, 2, 1)
    return torch.mean(nm)

def cost(x, y, l=2, p=2):
    diff = x - y
    diff = diff.view(diff.size(0), -1)
    cost = (diff.norm(l, dim=1)**p)/p
    return cost

def ineq_loss(x, y, ux, vy, c, lm, reg_type='l2'):
    if reg_type == 'l2':
        return lm*torch.mean((torch.clamp((ux + vy - c(x, y)), min=0))**2)
    elif reg_type == 'entropy':
        return lm*torch.mean(torch.exp((ux + vy - c(x, y))/lm))

def eq_loss(x, y, ux, vy, c, lm):
    return lm*torch.mean((torch.abs(ux + vy - c(x, y)))**2)

def map_loss(x, y, fy, ux, vy, c, lm, reg_type='l2'):
  if reg_type == 'l2':
      return 2*torch.mean(c(x, fy)*lm*torch.clamp(ux + vy - c(x, y), min=0))
  elif reg_type == 'entropy':
      return torch.mean(c(x, fy)*torch.exp((ux + vy - c(x, y))/lm))

def calc_eq(x, y, phi, psi, cost, lambda_eq):
    ux, vy = phi(x), psi(y)
    return eq_loss(x, y, ux, vy, cost, lambda_eq)

def calc_interp_ineq(x, y, phi, psi, cost, lambda_ineq, loss=None):
    batch_size = x.size(0)
    x_dim = x.dim()
    alpha_x = utils.to_var(utils.unsqueeze(torch.rand(batch_size), ndim=x_dim))
    alpha_y = utils.to_var(utils.unsqueeze(torch.rand(batch_size), ndim=x_dim))
    interp_x = x * alpha_x + y * (1 - alpha_x)
    interp_y = y * alpha_y + x * (1 - alpha_y)
    interp_x = interp_x.detach()
    interp_y = interp_y.detach()
    interp_ux, interp_vy = phi(interp_x), psi(interp_y)
    return loss(interp_x, interp_y, interp_ux, interp_vy, cost, lambda_ineq)

def gp_loss(x, y, disc, lm, clamp=True):
    batch_size = x.size()[0]
    gp_alpha = utils.unsqueeze(torch.rand(batch_size), ndim=x.dim())
    gp_alpha = gp_alpha.cuda()
    interp = Variable(gp_alpha * x.data + (1 - gp_alpha) * y.data, requires_grad=True)
    d_interp = disc(interp)
    grad_interp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                          grad_outputs=torch.ones(d_interp.size()).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_interp = grad_interp.view(grad_interp.size(0), -1)
    diff = grad_interp.norm(2, dim=1) - 1
    if clamp:
        diff = torch.clamp(diff, 0)
    return lm*torch.mean(diff**2)

import autograd.numpy as np
from autograd import grad
from scipy.optimize import fsolve
from sklearn.decomposition import PCA

# Personal libraries
from utils import tensor_alg_prod
from utils import tensor_alg_inv
from utils import depth_inds
from utils import group_exp
from utils import group_log
from group_mean import mean


def d2h(t, v, y, depth, channels):
    inds = depth_inds(channels, depth)
    a = group_exp(-0.5*t*v, depth, inds)
    b = tensor_alg_prod(a, y, depth, inds)
    b = tensor_alg_prod(b, a, depth, inds)
    logb = group_log(b, depth, inds)
    res = -2*np.dot(logb, v) ######################### ici choix produit scalaire
    return res


dt_d2h    = grad(d2h, argnum=0)
gradv_d2h = grad(d2h, argnum=1)


def d2h_2(t, v, y, depth, channels):
    """
    Only for depth = 2, terms of higher depth in BCH formula are set to zero.
    """
    depth = 2
    inds = depth_inds(channels, depth)
    logy = group_log(y, depth, inds)
    return 2*np.dot(t*v-logy, v)


def grad_fi(v, yi, depth, channels):
    d2h_t = lambda t: d2h(t, v, yi, depth,channels)
    initial_guess = 0.
    tistar = fsolve(d2h_t, initial_guess)
    
    lambda_  = dt_d2h(tistar, v, yi, depth, channels)
    gradv = gradv_d2h(tistar, v, yi, depth, channels)
    return -1./lambda_*gradv
    

def grad_F(v, Y, depth, channels):
    s = 0.
    for i in range(len(Y)):
        s += grad_fi(v, Y[i], depth, channels)
    return s


def my_adam(grad, x, argmax, datasig, depth, channels, callback=None, num_iters=100,
            step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, datasig, depth, channels)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        if argmax:
            x = x + step_size*mhat/(np.sqrt(vhat) + eps)
        else:
            x = x - step_size*mhat/(np.sqrt(vhat) + eps)
        x[0] = 0.  # keep x in the Lie algebra
        xnorm = np.sqrt(np.sum(np.power(x, 2)))
        x = x / xnorm
    return x


def recenter_group_elems(SX, channels, depth, inds):
    """Center g_1, ..., g_N around identity element."""
    batch = len(SX)
    weights = 1./batch*np.ones(batch)
    Sm = mean(SX, channels, depth, weights)
    vec_ones = np.ones((batch, 1))
    SX = np.concatenate((vec_ones, SX), axis=1)
    Sm = np.concatenate(([1.], Sm))
    Sm_inv = tensor_alg_inv(Sm, depth, inds)
    SX_centered = np.zeros_like(SX)
    for i in range(batch):
        SX_centered[i] = tensor_alg_prod(Sm_inv, SX[i], depth, inds)
    return SX_centered


def pga(SX, channels, depth, n_components):
    batch = len(SX)
    inds = depth_inds(channels, depth)
    np.random.seed(1206)

    principal_directions = []  
    SXk = recenter_group_elems(SX, channels, depth, inds)

    for icomp in range(n_components):
        print(f"Starting computation Principal Component #{icomp+1}")
        # principal geodesic analysis
        random_obs_idx = np.random.randint(0, len(SXk))
        initial_guess_v = group_log(SXk[random_obs_idx], depth, inds)
        optimized_v = my_adam(grad_F, initial_guess_v, argmax=True, datasig=SXk, 
                                depth=depth, channels=channels)#, step_size=step_size, num_iters=num_epochs * num_batches, callback=print_perf)
        
        principal_directions.append(optimized_v)

        for i in range(batch):
            d2h_t = lambda t: d2h(t, optimized_v, SXk[i], depth, channels)
            initial_guess_t = 0.
            optimized_ti = fsolve(d2h_t, initial_guess_t)[0]
            sig_proj = group_exp(optimized_ti*optimized_v, depth, inds)
            sig_proj_inv = tensor_alg_inv(sig_proj, depth, inds)
            SXk[i] = tensor_alg_prod(sig_proj_inv, SXk[i], depth, inds)

    return principal_directions


def tangent_pga(SX, channels, depth, n_components):
    """
    Euclidean PCA in the tangent space at the group mean.
    """
    batch = len(SX)
    inds = depth_inds(channels, depth)
    SXk = recenter_group_elems(SX, channels, depth, inds)
    for i in range(batch):
        SXk[i] = group_log(SXk[i], depth, inds)
    pca = PCA(n_components=n_components)
    pca.fit(SXk[:, 1:])
    vec_zeros = np.zeros((n_components, 1))
    principal_directions = np.concatenate((vec_zeros, pca.components_), axis=1)    
    return principal_directions
    

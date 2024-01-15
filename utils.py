# import numpy as np
import autograd.numpy as np
# import jax.numpy as np

"""


Basic utility functions

Support Autograd for Automatic Differentiation.


"""

def depth_inds(channels, depth):
    """
    Most libraries computing the signature transform output the signature as a
    vector. This function outputs the indices corresponding to first value of
    each signature depth in this vector. Example: with depth=4 and channels=2,
    returns [0, 1, 3, 7, 15, 31].
    """
    return np.concatenate((np.array([0]), np.cumsum(np.power(channels, np.arange(0, depth+1)))))


def tensor_alg_prod(x, y, depth, inds):
    """
    Product between two elements in the tensor algebra. Supported by Autograd.

    Parameters
    ----------
    x, y : nd.array
        The two signature we want the product of. Caution: scalar value must be
        included.

    inds : nd.array
        The output of :func:`depth_inds`.

    Returns
    -------
    prod : nd.array
        The product of x and y. NB: scalar value is included.

    """
    prod = np.zeros(inds[-1])#, dtype="float")
    for idx_depth in range(1, depth+2):
        start = inds[idx_depth-1]
        end = inds[idx_depth]
        for i in range(0, idx_depth):
            kron = np.kron(
                x[inds[i]:inds[i+1]],
                y[inds[idx_depth-i-1]:inds[idx_depth-i]]
            )
            kron = np.pad(kron, (start, inds[-1]-end), 'constant', constant_values=(0., 0.))
            prod = prod + kron
    return prod


def tensor_alg_inv(x, depth, inds):
    inv = -x
    identity = np.concatenate((np.array([1.]), np.zeros(len(x)-1)))
    left = identity - x
    it = 0
    while it < depth+1:
        inv = tensor_alg_prod(left, inv, depth, inds)
        inv = inv + identity
        it = it + 1
    return inv


def group_exp(logsig, depth, inds):
    identity = np.concatenate((np.array([1.]), np.zeros(len(logsig)-1)))
    right = 1./depth*logsig
    right = right + identity
    # right[0] = 1.  # not supported by autograd
    it = 0
    while it < depth-1:  # Horner's method 
        left = 1./(depth-1-it)*logsig
        right = tensor_alg_prod(left, right, depth, inds)
        right = right + identity
        it = it + 1
    return right


def group_log(sig, depth, inds):
    identity = np.concatenate((np.array([1.]), np.zeros(len(sig)-1)))
    a = np.repeat((1.-depth)/depth, len(sig)-1)
    a = np.concatenate((np.array([1./sig[0]]), a))
    right = a*sig
    it = 0
    while it < depth-1:
        if it == depth-2:
            b = np.ones(len(sig)-1)
        else:
            b = np.repeat((it+2-depth)/(depth-1-it), len(sig)-1)
        b = np.concatenate((np.array([0.]), b))
        left = b*sig
        right = tensor_alg_prod(left, right, depth, inds)
        right = right + identity
        it = it + 1
    right = right - identity
    return right
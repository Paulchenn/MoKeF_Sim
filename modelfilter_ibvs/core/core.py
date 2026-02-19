import numpy as np

def skew(v):
    """Erstellt eine schiefsymmetrische Matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def flatten_H(H): return H.reshape(9)
def unflatten_H(h_vec): return h_vec.reshape(3, 3)
import scipy.sparse as sp

def dict_to_sparse_matrix(d):
    coo_matrix = sp.coo_matrix(
        (
            d['data'], 
            (d['rows'], d['cols'])
        ),
        shape=d['shape']
    ).tocsc()
    return coo_matrix
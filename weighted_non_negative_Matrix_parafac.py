# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""put the part to the end of candecom-parafec code

    This is a two matrix  A  and B  to reconstructe X tensor.
    A~A*((W*X)B)/((W*(AB.T))B)
    B~B*(A.T(W*X))/(A.T((AB.T)*W))
    
"""

def weighted_non_negative_Matrix_parafac(tensor, rank, w_tensor, n_iter_max=100, init='svd', tol=10e-7,
                         random_state=None, verbose=0):
    """Non-negative CP decomposition

        Uses multiplicative updates, see [2]_

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity

    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    """
    epsilon = 10e-8

    # Initialisation
    if init == 'svd':
        factors = parafac(tensor, rank)
        nn_factors = [T.abs(f) for f in factors]
    else:
        rng = check_random_state(random_state)
        nn_factors = [T.tensor(np.abs(rng.random_sample((s, rank))), **T.context(tensor)) for s in tensor.shape]
    MatrixA = T.dot(nn_factors[0],T.transpose(nn_factors[1]))
    MatrixB = T.dot(nn_factors[2],T.transpose(nn_factors[3]))
    #print('\n\n********check the Initial nn_factors*****','\n')    
    #print(nn_factors)  #check the Initial nn_factors
    #print('****************','\n')
    n_factors = len(nn_factors)
    norm_tensor = T.norm(tensor, 2)
    rec_errors = []

    for iteration in tqdm(range(n_iter_max),desc= 'Non-negative Matrix_parafac:'):
        for mode in range(T.ndim(tensor)/T.ndim(tensor)):
            # khatri_rao(factors).T.dot(khatri_rao(factors))
            # simplifies to multiplications
            '''sub_indices = [i for i in range(n_factors) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    #lefthalf = T.dot(unfold(w_tensor, mode), T.dot(nn_factors[mode],accum)) #add line
                    accum = accum*T.dot(T.transpose(nn_factors[e]), nn_factors[e])
                    #denominator = T.dot(lefthalf, T.dot(T.transpose(nn_factors[e]), nn_factors[e]))   #add line
                else:
                    accum = T.dot(T.transpose(nn_factors[e]), nn_factors[e])
                    #denominator = T.dot(T.dot(unfold(w_tensor, mode),T.dot(nn_factors[mode], T.transpose(nn_factors[e]))), nn_factors[e])  #dot AB
            '''
            ##Now A
            MatrixA_numerator = T.dot(tensor * w_tensor, MatrixB)   #   A
            MatrixA_numerator = T.clip(MatrixA_numerator, a_min=epsilon, a_max=1)
            ##
            MatrixA_denominator = T.dot((w_tensor * T.dot(MatrixA, T.transpose(MatrixB))),MatrixB)
            MatrixA_denominator = T.clip(MatrixA_denominator, a_min=epsilon, a_max=None)
            MatrixA = MatrixA* MatrixA_numerator / MatrixA_denominator
            ##Now B 
            MatrixB_numerator = T.dot(T.transpose(MatrixA), tensor * w_tensor)   #   B
            MatrixB_numerator = T.clip(MatrixB_numerator, a_min=epsilon, a_max=1)
            ##
            MatrixB_denominator = T.dot(T.transpose(MatrixA), T.dot(MatrixA, T.transpose(MatrixB)) *w_tensor )
            MatrixB_denominator = T.clip(MatrixB_denominator, a_min=epsilon, a_max=None)
            MatrixB = MatrixB* MatrixB_numerator / MatrixB_denominator
    

        rec_error = T.norm(tensor - T.dot(MatrixA, T.transpose(MatrixB)), 2) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print('reconstruction error={}, variation={}.'.format(
                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

        if iteration > 1 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break

    return MatrixA, MatrixB


import numpy as np
from scipy.sparse import csr_array, csc_array, coo_array, csr_matrix, csc_matrix, coo_matrix, dia_array, diags
from scipy.sparse._sputils import get_index_dtype, isscalarlike, isintlike


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set (repeated offsets are not allowed):
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square array large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse array format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the array.

    Returns
    -------
    new_array : dia_array
        `dia_array` holding the values in `diagonals` offset from the main diagonal
        as indicated in `offsets`.

    Notes
    -----
    Repeated diagonal offsets are disallowed.

    The result from ``diags_array`` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    ``diags_array`` differs from `dia_array` in the way it handles off-diagonals.
    Specifically, `dia_array` assumes the data input includes padding
    (ignored values) at the start/end of the rows for positive/negative
    offset, while ``diags_array`` assumes the input data has no padding.
    Each value in the input `diagonals` is used.

    .. versionadded:: 1.11

    See Also
    --------
    dia_array : constructor for the sparse DIAgonal format.

    Examples
    --------
    >>> from scipy.sparse import diags_array
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1., 0., 1., 0.],
           [1., 2., 0., 2.],
           [0., 2., 3., 0.],
           [0., 0., 3., 4.]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])

    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(np.atleast_1d, diagonals))

    offsets = np.atleast_1d(offsets)

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        dtype = np.common_type(*diagonals)

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError(f"Offset {offset} (index {j}) out of bounds")
        try:
            data_arr[j, k:k+length] = diagonal[...,:length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    f"Diagonal length (index {j}: {len(diagonal)} at"
                    f" offset {offset}) does not agree with array size ({m}, {n})."
                ) from e
            raise

    return dia_array((data_arr, offsets), shape=(m, n)).asformat(format)


def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Sparse array of chosen shape with ones on the kth diagonal and zeros elsewhere.

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Returns
    -------
    new_array : sparse array
        Sparse array of chosen shape with ones on the kth diagonal and zeros elsewhere.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <DIAgonal sparse array of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    # TODO: delete next 15 lines [combine with _eye()] once spmatrix removed
    return _eye(m, n, k, dtype, format)


def _eye(m, n, k, dtype, format, as_sparray=True):
    if as_sparray:
        csr_sparse = csr_array
        csc_sparse = csc_array
        coo_sparse = coo_array
        diags_sparse = diags_array
    else:
        csr_sparse = csr_matrix
        csc_sparse = csc_matrix
        coo_sparse = coo_matrix
        diags_sparse = diags

    if n is None:
        n = m
    m, n = int(m), int(n)

    if m == n and k == 0:
        # fast branch for special formats
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n+1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_sparse, 'csc': csc_sparse}[format]
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_sparse((data, (row, col)), (n, n))

    data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)

def matrix_power(A, power):
    """
    Raise a square matrix to the integer power, `power`.

    For non-negative integers, ``A**power`` is computed using repeated
    matrix multiplications. Negative integers are not supported.

    Parameters
    ----------
    A : (M, M) square sparse array or matrix
        sparse array that will be raised to power `power`
    power : int
        Exponent used to raise sparse array `A`

    Returns
    -------
    A**power : (M, M) sparse array or matrix
        The output matrix will be the same shape as A, and will preserve
        the class of A, but the format of the output may be changed.

    Notes
    -----
    This uses a recursive implementation of the matrix power. For computing
    the matrix power using a reasonably large `power`, this may be less efficient
    than computing the product directly, using A @ A @ ... @ A.
    This is contingent upon the number of nonzero entries in the matrix.

    .. versionadded:: 1.12.0

    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csc_array([[0,1,0],[1,0,1],[0,1,0]])
    >>> A.todense()
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    >>> (A @ A).todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A2 = sparse.linalg.matrix_power(A, 2)
    >>> A2.todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A4 = sparse.linalg.matrix_power(A, 4)
    >>> A4.todense()
    array([[2, 0, 2],
           [0, 4, 0],
           [2, 0, 2]])

    """
    M, N = A.shape
    if M != N:
        raise TypeError('sparse matrix is not square')

    if isintlike(power):
        power = int(power)
        if power < 0:
            raise ValueError('exponent must be >= 0')

        if power == 0:
            return eye_array(M, dtype=A.dtype)

        if power == 1:
            return A.copy()

        tmp = matrix_power(A, power // 2)
        if power % 2:
            return A @ tmp @ tmp
        else:
            return tmp @ tmp
    else:
        raise ValueError("exponent must be an integer")

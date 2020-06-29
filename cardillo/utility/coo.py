from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.sputils import isshape, check_shape
from numpy import repeat, tile

class Coo(object):
    """Small container storing the sparse matrix shape and three lists for accumulating the entries for row, column and data [1]_.

        Parameters
        ----------
        shape : tuple, 2D
            tuple defining the shape of the matrix

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
    """

    def __init__(self, shape):
        # check shape input
        if isinstance(shape, tuple):
            pass
        else:
            try:
                shape = tuple(shape)
            except Exception:
                raise ValueError('input argument shape is not tuple or cannot be interpreted as tuple')

        # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L210
        if isshape(shape, nonneg=True):
            M, N = shape
            # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L267
            self.__shape = check_shape((M, N))
        else:
            raise TypeError('input argument shape cannot be interpreted as correct shape')
        
        # initialice empty lists for row, columns and data
        self.__row = []
        self.__col = []
        self.__data = []

    @property
    def shape(self):
        return self.__shape

    @property
    def row(self):
        return self.__row

    @property
    def col(self):
        return self.__col
        
    @property
    def data(self):
        return self.__data

    def extend(self, matrix, DOF):
        """Extend container with data given in matrix and indices stored in the tuple shift.

        Parameters
        ----------
        matrix: numpy.ndarray, 2D
            dense matrix which has to be stored
        DOF : tuple, 2D
            tuple defining the global row and column indices of the dense matrix
        transposd: bool
            should the transposed matrix should be stored 
        """
        self.data.extend(matrix.reshape(-1, order='C').tolist())
        self.row.extend( repeat(DOF[0], DOF[1].size).tolist() )
        self.col.extend( tile(DOF[1], DOF[0].size).tolist() )

    def tosparse(self, scipy_matrix):
        """Convert container to scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix((self.data, (self.row, self.col)), shape=self.shape)

    def tocsc(self):
        """Convert container to scipy csc_matrix.
        """
        return self.tosparse(csc_matrix)

    def tocsr(self):
        """Convert container to scipy csr_matrix.
        """
        return self.tosparse(csr_matrix)

    def tocoo(self):
        """Convert container to scipy coo_matrix.
        """
        return self.tosparse(coo_matrix)

    def toarray(self):
        """Convert container to 2D numpy array.
        """
        return self.tocoo().toarray()
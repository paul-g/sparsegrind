"""
    This module is used for the precision analysis in lossy/mixed precision
    matrix representations.

    Here we analyse how much we can reduce precision of matrix values (and,
    apparently, drop some very tiny matrix entries), keeping matrix norms
    (max and min singular values) close to original, up to a given tolerance.
    By doing this we ensure that iterative methods won't feel much difference
    between original and compressed matrix --- up to a given tolerance :-)

"""

import scipy.sparse.linalg as linalg
import numpy as np
import collections
import math
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

# bitwidth of IEEE double mantissa including implicit leading bit
double_man_bitwidth = 53

bytes_per_data = 8
bytes_per_metadata = 4

def trim_ieee_mantissa(data, target_bitwidth, split_rep=None):
    """
    Here we simply reduce precision of each input entry by representing it in a
    IEEE 745 floating point with same (as before) exponent but mantissa of target_bitwidth.

    Returns:
        numpy.ndarray: same shape as before, each entry of reduced precision
    """

    # all binary 1 for the leading bitwidth positions in IEEE double float mantissa
    mask = np.sum([2**k for k in range(target_bitwidth)])*2**(double_man_bitwidth - target_bitwidth)
    # get representation in the form m*2^exp, where 0.5 <= abs(m) < 1.
    m, exp = np.frexp(data)
    # re-construct how mantissa should look in given bit width
    integer_m_rep = (m * 2**double_man_bitwidth).astype(np.int64)
    # working around bitwise & on negative integers
    pos_integer_m_rep = np.where(integer_m_rep > 0, integer_m_rep, 0)
    neg_integer_m_rep = np.where(integer_m_rep < 0, -integer_m_rep, 0)
    pos_m_in_target_bitwidth = (pos_integer_m_rep & mask).astype(np.float64)/2**(double_man_bitwidth)
    neg_m_in_target_bitwidth = (neg_integer_m_rep & mask).astype(np.float64)/2**(double_man_bitwidth)
    m_in_target_bitwidth = pos_m_in_target_bitwidth - neg_m_in_target_bitwidth
    # construct representation of matrix values in IEEE float with given mantissa bitwidth
    value_in_target_bitwidth = np.ldexp(m_in_target_bitwidth, exp)

    split_rep.append((m_in_target_bitwidth, exp))

    print "------trim_ieee_mantissa----------------------"
    print (m,exp), integer_m_rep,
    print pos_integer_m_rep, neg_integer_m_rep, pos_m_in_target_bitwidth, neg_m_in_target_bitwidth
    print "mantissas=", m_in_target_bitwidth, value_in_target_bitwidth

    return value_in_target_bitwidth

def reduce_elementwise(n, matrix, target_bitwidth):
    """
    Here we simply reduce precision of each matrix entry by representing it in a
    IEEE 745 floating point with same exponent but mantissa of target_bitwidth.

    Returns:
        scipy.sparse.csr_matrix: CSR matrix of same shape and sparsity but reduced precision values
        float64: total running error (vector L1 norm) in data representation
    """

    total_error = 0

    # all binary 1 for the leading bitwidth positions in IEEE double float mantissa
    mask = np.sum([2**k for k in range(target_bitwidth)])*2**(double_man_bitwidth - target_bitwidth)

    ### ============================================

    # get representation in the form m*2^exp, where 0.5 <= abs(m) < 1.
    m, exp = np.frexp(matrix.data)
    # re-construct how mantissa should look in given bit width
    integer_m_rep = (m * 2**double_man_bitwidth).astype(np.int64)
    # working around bitwise & on negative integers
    pos_integer_m_rep = np.where(integer_m_rep > 0, integer_m_rep, 0)
    neg_integer_m_rep = np.where(integer_m_rep < 0, -integer_m_rep, 0)
    pos_m_in_target_bitwidth = (pos_integer_m_rep & mask).astype(np.float64)/2**(double_man_bitwidth)
    neg_m_in_target_bitwidth = (neg_integer_m_rep & mask).astype(np.float64)/2**(double_man_bitwidth)
    m_in_target_bitwidth = pos_m_in_target_bitwidth - neg_m_in_target_bitwidth
    # construct representation of matrix values in IEEE float with given mantissa bitwidth
    value_in_target_bitwidth = np.ldexp(m_in_target_bitwidth, exp)
    # absolute error
    total_error = np.sum(np.abs(matrix.data - value_in_target_bitwidth));

    target_matrix = csr_matrix( (value_in_target_bitwidth, matrix.indices, matrix.indptr), matrix.shape )

#    print "vector of errors: ", np.abs(matrix.data-value_in_target_bitwidth)

    return target_matrix, total_error


def split_to_buckets(n, matrix, target_bitwidth = 16, tol = 1e-8, bucket_size = None, integer_bits = 2):
    """
    Here we delta-encode matrix entries by representing them in a form

        bucket_representative_64bit + fixed_point_mantissa_correction*2^exp,

    where bucket_representative_64bit is IEEE 745 floating point number and
    mantissa correction term is a signed fixed point number of target_bitwidth
    of the form

       sign bit | integer bits . fractional bits

    which shares exponent with bucket representative. If bucket_size is None, we
    start a new bucket when
     - bucket representative - new member candidate > tol
       (representational error of a new member is unacceptable)
     - total running error within the bucket is above threshold.

    We assume spacially close matrix entries are close to each other
    numerically. The bucket representative is chosen to minimise the common
    representational error (e.g. average of entries). For the variable bucket
    size we use given tolerance for both criteria above.

    We explicitly form matrix values in this form and then convert them back to
    IEEE doubles to check for convergence properties. In the case bucket
    representative falls below the threshold, we drop whole bucket and return
    the number of entries to purge.

    Returns:
        scipy.sparse.csr_matrix: CSR matrix of same shape and sparsity but reduced precision values
        float64: total running error (vector L1 norm) in data representation
        int: number of entries in given matrix to be dropped
    """

    # do not support variable bucket size yet
    assert bucket_size is not None

    # we do allow for sign and integer bits in our fixed point deltas.
    mantissa_bits = target_bitwidth - 1 - integer_bits

    print matrix.nnz, matrix.data, target_bitwidth, integer_bits, mantissa_bits

    target_values = []
    buckets = []
    position = 0
    total_error = 0
    print "---------------------------"
    while (position < matrix.nnz):

        # take matrix entries to form the bucket
        original_values = matrix.data[position:(position+bucket_size)].copy()

        # delta encoding
        # TODO: average might not be the best, e.g. avg(250.5,-280)=-14.75
        bucket_representative = np.average(original_values)
        deltas = bucket_representative - original_values
        # reduce precision of deltas
        deltas_trimmed_split = []
        deltas_in_target_bitwidth = trim_ieee_mantissa(deltas, mantissa_bits, deltas_trimmed_split)

        # get deltas in the form m*2^exp, where 0.5 <= abs(m) < 1.
        m,exp = deltas_trimmed_split[0]
        # bring them to the common exponent with bucket representative
        rm, rexp = np.frexp(bucket_representative)
        exponent_diff = exp - rexp
        m_for_common_exp = m * 2.0**exponent_diff
        # check that integer components do not exceed integer_bits,
        # being brought to fixed point representation.
        if (np.any(np.greater(m_for_common_exp, 2**integer_bits-1))):
            print "cannot fit deltas to +/-{:1d}.{:2d} bit form with bucket_size={:d}\n".format(
                                                       integer_bits, mantissa_bits,bucket_size)
            return

        # put data in a bucket
        buckets.append([bucket_representative, deltas_in_target_bitwidth])
        position += bucket_size

        # error estimate
        new_values = bucket_representative + deltas_in_target_bitwidth
        target_values.append(new_values)
        total_error += np.sum(np.abs(original_values - new_values))

        print "---------------------------"
        print deltas_trimmed_split

        print position, original_values, bucket_representative, deltas
        print (m,exp), (rm, rexp), exponent_diff
        print deltas_in_target_bitwidth

    target_matrix = csr_matrix( (target_values, matrix.indices, matrix.indptr), matrix.shape )

    #print "vector of errors: ", np.abs(matrix.data-value_in_target_bitwidth)

    return target_matrix, total_error


def matrix_norms(matrix):
    """Calculating matrix norms for a given matrix:
        l_{fro}: frobenius norm
        l_{1}:  max(sum(abs(x), axis=0));
        l_{2}:  euclidian operator norm == max singular value;
        l_{-2}: Python specific peculiar notatation for min singular value.

        This helps estimating matrix singular values important for convergence
        analysis of iterative methods. """

    # FIXME Correct is to call matrix.todense()! Since this way
    # we calculate vector norms: matrix.data is a ndarray, and there's no way
    # to correctly reshape it. Converting to dense... you don't want it.

    #### Frobenius norm is not available for the vector objects
    ####frobenius_norm      = np.linalg.norm(matrix.data, 'fro');
    operator_l1_norm    = np.linalg.norm(matrix.data, 1);
    max_singular_value  = np.linalg.norm(matrix.data, 2);
    min_singular_value  = np.linalg.norm(matrix.data, -2);
    condition_number = max_singular_value/min_singular_value

    return (operator_l1_norm, max_singular_value, min_singular_value, condition_number)

def l2_error(vector1, vector2):
    """Calculating L2 norm of abs difference between two vectors"""

    return np.linalg.norm(vector1-vector2, 2)


def solve_cg(matrix, solve_tol = 1e-4, max_iterations=2000):
    """Solves matrix problem iteratively to count number of iterations"""

    rank = matrix.shape[0]
    # constructing diagonal preconditioner: inverse diagonal entries of
    # a given matrix
    P = spdiags(1. / matrix.diagonal(), 0, rank, rank)
    # vector of all 1/1000.0
    b = np.ones(rank)/10000.0

    iteration_count = np.zeros(1, dtype='int')
    def iteration_counter(v):
        iteration_count[0] += 1

    sol = linalg.cg(matrix, b, tol=solve_tol, maxiter=max_iterations, M=P, callback=iteration_counter)

    # if sol[1] == 0, solution successful.
    # if positive, did not converge, equals the num of iterations
    # if negative, break down
    return sol[1], iteration_count[0], sol[0]

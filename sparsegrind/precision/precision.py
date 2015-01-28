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

verbose = False

def trim_ieee_mantissa(data, target_bitwidth, split_rep=None):
    """
    Here we simply reduce precision of each input entry by representing it in a
    IEEE 745 floating point with same (as before) exponent but mantissa of target_bitwidth.

    Returns:
        numpy.ndarray: same shape as before, each entry of reduced precision
        split_rep : optional, frexp representation of trimmed data
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

    # optionally return m*2^exp representation of output
    if split_rep is not None:
        split_rep.append((m_in_target_bitwidth, exp))

#    print "------trim_ieee_mantissa----------------------"
#    print (m,exp), integer_m_rep,
#    print pos_integer_m_rep, neg_integer_m_rep, pos_m_in_target_bitwidth, neg_m_in_target_bitwidth
#    print "mantissas=", m_in_target_bitwidth, value_in_target_bitwidth

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


def bucketize(data, bitwidth, tolerance, split_buckets):
    """
        A recursive helper function which aims to build a (list of) buckets out
        of input data. For the structure of bucket elements see split_to_buckets().

        It either picks a single bucket representative so that all other bucket
        members are all smaller and fit number representation, or splits data on
        violating member and recursively applies itself on both lists.
    """

    bucket_representative = (np.max(data))
    # delta encoding
    deltas = bucket_representative - data
    # reduce precision of deltas
    deltas_trimmed_split = []
    deltas_in_target_bitwidth = trim_ieee_mantissa(deltas, bitwidth, deltas_trimmed_split)
    # get deltas in the form m*2^exp, where 0.5 <= abs(m) < 1.
    m,exp = deltas_trimmed_split[0]
    # bring them to the common exponent with bucket representative
    rm, rexp = np.frexp(bucket_representative)
    exponent_diff = exp - rexp
    # in the actual encoding, only this part is to be stored
    m_for_common_exp = m * 2.0**exponent_diff

    if (verbose):
        print "---------------------------"
        print data, bucket_representative, deltas
        print (m,exp), (rm, rexp), exponent_diff
        print "to be stored:", m_for_common_exp
        print deltas_in_target_bitwidth

    num_buckets = 1
    # does any correction term require non-zero integer part?
    if (np.any(np.greater_equal(m_for_common_exp, 1))):
        if (verbose):
            print "bucket entry requires an integer part: ", m_for_common_exp
            print "cannot fit deltas to +/-{:1d}.{:2d} bit form\n".format(0, bitwidth)

        # split data into 2 parts starting from violating member
        branching_point = np.argmax(m_for_common_exp >= 1)
        left, right = np.split(data,[branching_point])
        # re-run bucketize on second part (first part is already fine)
        right_num_buckets, right_reduced, droplist = bucketize(right, bitwidth, tolerance, split_buckets)
        # concatenate already processed deltas on the left to bucket(s) on the right
        new_values = np.where(m_for_common_exp < 1, bucket_representative - deltas_in_target_bitwidth, right_reduced)
#        new_values_left = bucket_representative - 
#        new_values = bucket_representative - final_deltas
        num_buckets += right_num_buckets

        if (verbose):print "end of if", branching_point, left, right, right_num_buckets, right_reduced#, final_deltas
    else:
        new_values = bucket_representative - deltas_in_target_bitwidth
        if (verbose):print "else branch"


    if (verbose):print num_buckets, new_values, m_for_common_exp
    return num_buckets, new_values, []


def split_to_buckets(n, matrix, target_bitwidth = 16, tol = 1e-8, bucket_size = 1024, split_buckets = False):
    """
    Here we delta-encode matrix entries by representing them in a form

        bucket_representative_64bit - 0.fixed_point_correction*2^exp,

    where bucket_representative_64bit is the average number within the bucket
    stored as IEEE 745 double. The mantissa correction term is a fixed point
    number, which shares exponent with bucket representative and its fractional
    part of mantissa has target_bitwidth.

    0.fixed_point_fractional_correction*2^exp,

    We assume spacially close matrix entries are close numerically. The bucket
    representative is chosen as maximum in the bucket, so that all fixed point
    correction terms are always below zero and always subtracted. This avoids
    necessity of explicitly representing sign, integer part and leading zero,
    thus maximizing the precision of a correction term. Drawback of this choice
    for bucket representative: one correction term is always zero. If
    reorder=True we assume the ordering which eliminates explicitly stored zero.

    In the case bucket representative falls below the threshold, we drop whole
    bucket and return the number of entries to purge.

    If bucket_size is None, we start a new bucket when
     - bucket representative - new member candidate > tol
       (representational error of a new member is unacceptable)
     - total running error within the bucket is > tol.

    Returns:
        scipy.sparse.csr_matrix: CSR matrix of same shape and sparsity but reduced precision values
        float64: total running error (vector L1 norm) in data representation
        int: number of entries in given matrix to be dropped
    """

    #print matrix.nnz, matrix.data, target_bitwidth

    # one sign bit, implicit leading zero, point, then fraction
    # no integer component: always implicitly 0.
    fraction_bits = target_bitwidth - 1

    # split matrix.nnz into (relatively) large chunks
    split_idx = np.arange(bucket_size, matrix.nnz, bucket_size)
    original_values = np.split(matrix.data, split_idx)
    split_idx = np.concatenate(([0],split_idx,[matrix.nnz]))

    # buckets: list of lists of matrix entries in reduced precision,
    # each sublist represent a bucket
    if (verbose):print "orig", original_values, "----"

    new_values = np.zeros(matrix.nnz)
    position = 0
    for i,v in zip(range(len(original_values)),original_values):
        # further recursively split them into buckets, if necessary.
        num_buckets, bucket, droplist = bucketize(v, fraction_bits, tol, split_buckets)

        if (verbose):print type(bucket), len(bucket), split_idx[i], split_idx[i+1]

        new_values[split_idx[i]:split_idx[i+1]] = bucket
        if (verbose):print "i = ", i, num_buckets, bucket, droplist, new_values

    # error estimate
    total_error = np.sum(np.abs(matrix.data - new_values))
    # updated matrix
    target_matrix = csr_matrix( (new_values, matrix.indices, matrix.indptr), matrix.shape )

    if (verbose):
        print matrix.data, new_values, matrix.data-new_values

    return target_matrix, total_error, num_buckets, droplist



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

from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy.linalg as npl
from numpy import log
from random import randint
from random import uniform
from re import split
import time


class Matrix:
    """
        Class of a real-valued matrix
    """

    def __init__(self, matrix):
        """
            Constructs a matrix from a 2D array or other matrix

            :param matrix: Matrix or 2D array representing a matrix
        """

        # if matrix is Matrix type
        if type(matrix) == type(self):
            (self.__height, self.__width) = matrix.size()
            self.__matrix = deepcopy(matrix.__matrix)
            return

        # private fields
        self.__height = len(matrix)
        self.__width = len(matrix[0] if len(matrix) > 0 else 0)
        self.__matrix = deepcopy(matrix)

    @classmethod
    def fromSizes(cls, width, height):
        """
            Constructs a zero-valued matrix of specified width and height

            :param width: width of matrix
            :param height: height of matrix
        """

        matrix = []

        for _ in range(height):
            matrix.append([0] * width)

        return cls(matrix)

    @classmethod
    def asVector(cls, vector):
        """
            Constructs a matrix as a vector

            :param vector: vector to convert
        """

        return cls([[item] for item in vector])

    @classmethod
    def getIdentity(cls, size):
        """
            Constructs an identity matrix of specified size

            :param size: size of the matrix
        """

        return cls([[int(i == j) for j in range(size)] for i in range(size)])

    @classmethod
    def getRandomSPD(cls, size, min_elem=0, max_elem=10):
        """
            Constructs a random symmetrical matrix of specified size

            :param size: size of the matrix
        """

        matrix = []

        for _ in range(size):
            matrix.append([0] * size)

        for i in range(size):
            for j in range(i + 1, size):
                matrix[i][j] = randint(min_elem, max_elem)
                matrix[j][i] = matrix[i][j]

        for diag_idx in range(size):
            matrix[diag_idx][diag_idx] += size * max_elem

        return cls(matrix)

    def size(self):
        """
            Get matrix dimension sizes (height, width)
        """

        return (self.__height, self.__width)

    def conditionality(self):
        """
            Get matrix conditionality
        """

        return self.norm() * self.getInverse().norm()

    def eigenvalsBounds(self):
        """
            Get lower and upper bound on matrix's eigenvalues
            Uses Gershgorin's intervals
        """

        if self.__height != self.__width:
            return None

        row_intervals = []
        col_intervals = []

        for row_idx in range(self.__height):
            row_sum, col_sum = 0, 0
            for col_idx in range(self.__width):
                if row_idx == col_idx:
                    continue

                row_sum += abs(self.__matrix[row_idx][col_idx])
                col_sum += abs(self.__matrix[col_idx][row_idx])

            row_intervals.append(
                (self.__matrix[row_idx][row_idx] - row_sum, self.__matrix[row_idx][row_idx] + row_sum))
            col_intervals.append(
                (self.__matrix[col_idx][col_idx] - col_sum, self.__matrix[col_idx][col_idx] + col_sum))

        # intersect two sets of intervals
        temp = []

        # add intervals to common set with 1 is start element
        # and -1 is end element
        for row_interval in row_intervals:
            temp.append((row_interval[0], 1))
            temp.append((row_interval[1], -1))

        for col_interval in col_intervals:
            temp.append((col_interval[0], 1))
            temp.append((col_interval[1], -1))

        # sort intervals
        temp.sort()

        result = []

        # when we reach intervals_amt == 0 it means that
        # we have found an interval as a union of several intervals
        intervals_amt, first_pt = 0, temp[0][0]
        for elem in temp:
            intervals_amt += elem[1]

            if intervals_amt == 0:
                if first_pt is None:
                    first_pt = elem[0]
                else:
                    result.append((first_pt, elem[0]))
                    first_pt = None

        # lower bound is the minimal element of the result
        # upper bound is the maximal element of the result
        return result[0][0], result[-1][1]

    def getEigenvalues(self):
        """
            Get matrix eigenvalues
        """

        return npl.eigvals(self.__matrix)

    def getMatrix(self):
        """
            Get matrix as a 2D array
        """

        return self.__matrix

    def norm(self):
        """
            Calculate matrix norm
        """

        # max_row_sum = 0

        # for row in self.__matrix:
        #     row_sum = 0

        #     for elem in row:
        #         row_sum += abs(elem)

        #     if row_sum > max_row_sum:
        #         max_row_sum = row_sum

        # return max_row_sum

        return math.sqrt(max((self.getTransposed() * self).getEigenvalues()))

    def __getitem__(self, row):
        """
            Get or set matrix's item at [row][col]

            :param row: row index
            :param col: column index
        """

        matrix = self.__matrix

        class Row:
            def __getitem__(self, col):
                return matrix[row][col]

            def __setitem__(self, col, new_val):
                matrix[row][col] = new_val

            def __repr__(self):
                return f"{matrix[row]}"

            def toList(self):
                return matrix[row]

        return Row()

    def __setitem__(self, row, new_val):
        """
            Set element on row = row to new_val

            :param row: needed row
            :param new_val: new value to assing
        """

        self.__matrix[row] = new_val

    def __add__(self, other_mat):
        """
            Add other matrix to self

            :param other_mat: matrix to add
        """

        # get other matrix's sizes
        (other_height, other_width) = other_mat.size()

        # check sizes
        if self.__height != other_height or self.__width != other_width:
            # error: sizes do not match
            return None

        result_mat = Matrix.fromSizes(self.__width, self.__height)

        for row_idx in range(self.__height):
            for col_idx in range(self.__width):
                result_mat[row_idx][col_idx] = self.__matrix[row_idx][col_idx] + \
                    other_mat[row_idx][col_idx]

        return result_mat

    def __sub__(self, other_mat):
        """
            Subtract other matrix from self

            :param other_mat: matrix to subtract
        """

        # get other matrix's sizes
        (other_height, other_width) = other_mat.size()

        # check sizes
        if self.__height != other_height or self.__width != other_width:
            # error: sizes do not match
            return None

        result_mat = Matrix.fromSizes(self.__width, self.__height)

        for row_idx in range(self.__height):
            for col_idx in range(self.__width):
                result_mat[row_idx][col_idx] = self.__matrix[row_idx][col_idx] - \
                    other_mat[row_idx][col_idx]

        return result_mat

    def __mul__(self, other_mat):
        """
            Multiply self with other matrix

            :param other_mat: matrix to multiply with
        """

        # get other matrix's sizes
        (other_height, other_width) = other_mat.size()

        # check sizes
        if self.__width != other_height:
            # error: cannot multiply matrices of inappropriate sizes
            return None

        result_mat = Matrix.fromSizes(other_width, self.__height)

        # multiplying two matrices (self and other_mat)
        for row_idx in range(self.__height):
            for col_idx in range(other_width):
                # row by column product sum
                prod_sum = 0

                for mult_idx in range(self.__height):
                    # print(f"{row_idx = }, {col_idx = }, {mult_idx = }")

                    prod_sum += self.__matrix[row_idx][mult_idx] * \
                        other_mat[mult_idx][col_idx]

                result_mat[row_idx][col_idx] = prod_sum

        return result_mat

    def scalarMultiply(self, scalar):
        """
            Multiply self with a scalar

            :param scalar: scalar to multiply with
        """

        result = Matrix.fromSizes(self.__width, self.__height)

        for row_idx in range(self.__height):
            for col_idx in range(self.__width):
                result[row_idx][col_idx] = self.__matrix[row_idx][col_idx] * scalar

        return result

    def getTransposed(self):
        """
            Makes a transposed matrix from self
        """

        transposed = Matrix.fromSizes(self.__height, self.__width)

        for row in range(self.__height):
            for col in range(self.__width):
                transposed[col][row] = self.__matrix[row][col]

        return transposed

    def getInverse(self):
        """
            Creates an inverse matrix out of self
        """

        # check if size is appropriate
        if self.__height != self.__width:
            return None

        # NOTE: no check that matrix is invertible

        copy = deepcopy(self.__matrix)
        inverted = Matrix.fromSizes(self.__width, self.__height)

        # make inverted matrix as E = I_n
        for diag_idx in range(self.__height):
            inverted[diag_idx][diag_idx] = 1

        # I'll use the method of (A|E) -> (E|A^(-1))

        # make elements lower the main diagonal zeros
        for diag_idx in range(self.__height):
            # if there is zero on diagonal need to swap with non-zero
            if copy[diag_idx][diag_idx] == 0:
                nonzero_found = False

                # look through all lower rows to find non-zero (nz) element
                # if there are none of them, there is nothing to do
                for nz_check_idx in range(diag_idx + 1, self.__height):
                    if copy[nz_check_idx][diag_idx] != 0:
                        nonzero_found = True

                        copy[diag_idx], copy[nz_check_idx] = \
                            copy[nz_check_idx], copy[diag_idx]

                        inverted[diag_idx], inverted[nz_check_idx] = \
                            inverted[nz_check_idx].toList(), \
                            inverted[diag_idx].toList()

                if not nonzero_found:
                    continue

            # subtract current row from all rows below
            for lower_row_idx in range(diag_idx + 1, self.__height):
                coeff = copy[lower_row_idx][diag_idx] / \
                    copy[diag_idx][diag_idx]

                for col_idx in range(self.__width):
                    copy[lower_row_idx][col_idx] -= coeff * \
                        copy[diag_idx][col_idx]

                    inverted[lower_row_idx][col_idx] -= coeff * \
                        inverted[diag_idx][col_idx]

        # make elements higher the main diagonal zeros
        for diag_idx in range(self.__height - 1, -1, -1):
            coeff = copy[diag_idx][diag_idx]

            # make current diagonal element a 1
            copy[diag_idx][diag_idx] = 1

            # update right matrix
            for col_idx in range(self.__width):
                inverted[diag_idx][col_idx] /= coeff

            for upper_row_idx in range(diag_idx - 1, -1, -1):
                coeff = copy[upper_row_idx][diag_idx]

                # no need in this, but ok
                copy[upper_row_idx][diag_idx] = 0

                # update right matrix
                for col_idx in range(self.__width):
                    inverted[upper_row_idx][col_idx] -= coeff * \
                        inverted[diag_idx][col_idx]

        return inverted

    def checkSymmetry(self):
        """
            Checks if a square matrix is symmetrical.
        """

        for row_idx in range(self.__height):
            for col_idx in range(self.__width):
                if self.__matrix[row_idx][col_idx] != self.__matrix[col_idx][row_idx]:
                    # matrix is nonsymmetrical
                    return False

        return True

    def __checkPositiveDefinite(self):
        """
            Checks if matrix is positive definite
        """

        # TODO: implement
        return True

    def __repr__(self):
        """
            String representation of matrix
        """

        result = ""

        for row in self.__matrix:
            result += "\t"
            for elem in row:
                result += f"{elem:7.3f} "
            result += "\n"

        return result

    def toList(self):
        """
            Represent the matrix as a list (aka flatMap)
        """

        result = [0] * (self.__height * self.__width)
        back_idx = 0

        for row in self.__matrix:
            for elem in row:
                result[back_idx] = elem
                back_idx += 1

        return result


class SLAE:
    """
        Class of system of linear algebraic equation
    """

    def __init__(self, matrix, rhs_vector):
        """
            Constructs a SLAE Ax = f

            :param matrix: left hand side of the SLAE
            :param rhs_vector: right hand side of the SLAE 
        """

        # construct a matrix from 2D array
        self.__matrix = Matrix(matrix)

        # construct a vector-like matrix from an array
        self.__vector = Matrix(rhs_vector) if type(rhs_vector) is Matrix \
            else Matrix.asVector(rhs_vector)

    def richardsonSolve(self, tau=None, iterations_amt=100, threshold=1e-6):
        """
            Solves SLAE Ax = f using Richardson's method

            :param tau: step size
            :param iterations_amt: number of iterations
            :param threshold: threshold for convergence
        """

        (height, width) = self.__matrix.size()

        # check if matrix is square
        if height != width:
            return None

        # symmetrize the SLAE
        self.__symmetrise()

        if tau is None:
            eigenvals = sorted(self.__matrix.getEigenvalues())

            # get optimal tau
            tau = 2 / (eigenvals[0] + eigenvals[-1])

        # get additional matrices and vectors
        identity = Matrix.getIdentity(width)
        tau_A = self.__matrix.scalarMultiply(tau)
        tau_f = self.__vector.scalarMultiply(tau)

        # check if tau parameter is appropriate
        if (identity - tau_A).norm() >= 1:
            return None

        # get initial guess
        guessed_vector = Matrix.asVector([1] * width)
        residual = self.__vector - self.__matrix * guessed_vector

        residual_norms = [residual.norm()]

        # iterate while residual norm is not less than threshold
        while residual_norms[-1] >= threshold and iterations_amt > 0:
            iterations_amt -= 1

            guessed_vector = (identity - tau_A) * guessed_vector + tau_f
            residual = self.__vector - self.__matrix * guessed_vector
            residual_norms.append(residual.norm())

        return (guessed_vector, residual_norms)

    def __symmetrise(self):
        """
            Left-multiplies the SLAE on A^T, transformig it into 
            A^T * A * x = A^T * f
        """

        if self.__matrix.checkSymmetry():
            return

        transposed = self.__matrix.getTransposed()
        self.__matrix = transposed * self.__matrix
        self.__vector = transposed * self.__vector

    def getMatrixCond(self):
        """
            Get SLAE's matrix conditionality 
        """

        return self.__matrix.conditionality()

    def getMatrix(self):
        """
            Get SLAE's matrix
        """

        return deepcopy(self.__matrix)

    def getVector(self):
        """
            Get SLAE's vector
        """

        return deepcopy(self.__vector)

    def __repr__(self):
        """
            String representation of a SLAE
        """

        return f"matrix = \n{self.__matrix}\nrhs vector = \n{self.__vector}"


def testRandomTau(size):
    """
        Test the program with random appropriate tau
    """

    # generate random matrix and vector
    matrix = Matrix.getRandomSPD(size)
    rhs_vector = [randint(0, 10) for _ in range(size)]

    # get eigenvalues and max appropriate tau
    eigenvals = matrix.getEigenvalues()
    max_appr_val = 2 / max(eigenvals)

    # get random tau
    tau = uniform(0, max_appr_val)

    # get result for Ax = f
    slae = SLAE(matrix, rhs_vector)

    program_start = time.time()
    result, res_norms = slae.richardsonSolve(tau)
    program_end = time.time()

    np_start = time.time()
    np_result = Matrix.asVector(
        npl.solve(matrix.getMatrix(), rhs_vector).tolist())
    np_end = time.time()

    print(f"\nDifference norm for RT is {(np_result - result).norm()}")
    print(f"Numpy solved the SLAE in {np_end - np_start} sec")
    print(f"Program solved the SLAE in {program_end - program_start} sec\n")

    return result, res_norms


def testOptimalFromBounds(size):
    """
        Test the program with optimal tau obtained 
        from eigenvalues bounds
    """

    # generate random matrix and vector
    matrix = Matrix.getRandomSPD(size)
    rhs_vector = [randint(0, 10) for _ in range(size)]

    # get lower and upped bounds for eigenvalues
    lower_b, upper_b = matrix.eigenvalsBounds()

    # get optimal tau
    tau = 2 / (upper_b + lower_b)

    # get result for Ax = f
    slae = SLAE(matrix, rhs_vector)

    program_start = time.time()
    result, res_norms = slae.richardsonSolve(tau)
    program_end = time.time()

    np_start = time.time()
    np_result = Matrix.asVector(
        npl.solve(matrix.getMatrix(), rhs_vector).tolist())
    np_end = time.time()

    print(f"\nDifference norm for OfB is {(np_result - result).norm()}")
    print(f"Numpy solved the SLAE in {np_end - np_start} sec")
    print(f"Program solved the SLAE in {program_end - program_start} sec\n")

    return result, res_norms


def testOptimalFromExact(size):
    """
        Test the program with optimal tau obtained 
        from exact eigenvalues
    """

    # generate random matrix and vector
    matrix = Matrix.getRandomSPD(size)
    rhs_vector = [randint(0, 10) for _ in range(size)]

    # get lower and upped bounds for eigenvalues
    eigenvals = sorted(matrix.getEigenvalues())

    # get optimal tau
    tau = 2 / (eigenvals[0] + eigenvals[-1])

    # get result for Ax = f
    slae = SLAE(matrix, rhs_vector)

    program_start = time.time()
    result, res_norms = slae.richardsonSolve(tau)
    program_end = time.time()

    np_start = time.time()
    np_result = Matrix.asVector(
        npl.solve(matrix.getMatrix(), rhs_vector).tolist())
    np_end = time.time()

    print(f"\nDifference norm for OfE is {(np_result - result).norm()}")
    print(f"Numpy solved the SLAE in {np_end - np_start} sec")
    print(f"Program solved the SLAE in {program_end - program_start} sec\n")

    return result, res_norms


def testSimpleSLAE():
    """
        Test the program on simple SLAEs
    """

    simple_slaes = [
        SLAE(
            [
                [5, -1, -1],
                [1, 2, 3],
                [4, 3, 2]
            ],
            [0, 14, 16]
        ),
        SLAE(
            [
                [3, 0.7, 0.2, 0.2],
                [0.6, 5, 0.5, 0.5],
                [1.3, 0.3, 3.5, 0.4],
                [0.3, 0.3, 0.4, 4]
            ],
            [4, 5, -5, 5]
        ),
        SLAE(
            [
                [3.6, 1.8, -4.7],
                [2.7, -3.6, 1.9],
                [1.5, 4.5, 3.3]
            ],
            [-1.5, -1.2, -0.8]
        )
    ]

    test_numbering = iter(range(1, len(simple_slaes) + 1))
    for slae in simple_slaes:
        print(f"<------- Test case #{next(test_numbering)} ------->")
        print(slae)

        result, _ = slae.richardsonSolve(threshold=1e-10)

        print(f"result = \n{result}")
        print("<---------------------------->\n\n")


def testHardSLAE(file, test_log="", delta_vector=None, rhs_vector=None, conditionality=False):
    """
        Test hard SLAEs obtained from a file

        :file: file where to find a SLAE
        :test_log: logs to print
        :delta_vector: right hand side vector delta
        :conditionality: whether to print conditionality
    """

    with open(file, mode="r") as matrix_file:
        matrix = []

        # read file and extract data
        while matrix_file:
            # read line
            line = matrix_file.readline()

            # if we reached end of file
            if line == "":
                break

            # extract data from line, ignore '\n' and other NaNs
            matrix.append([float(item)
                          for item in split("\s+", line)[:-1] if item])

            # read additional "spacer" line
            line = matrix_file.readline()

        # if rhs_vector not provided make it randomly
        if not rhs_vector:
            rhs_vector = [randint(0, 10) for _ in range(len(matrix))]

        # if delta vector is provided, add it to rhs vector
        if delta_vector:
            for i in range(len(matrix)):
                rhs_vector[i] += delta_vector[i]

        # construct a SLAE
        hard_slae = SLAE(matrix, rhs_vector)

        if test_log:
            print(test_log)
            print(hard_slae)

        cond = 0
        if conditionality:
            cond = hard_slae.getMatrixCond()
            print(f"Matrix conditionality = {cond}\n")

        # get solution to the SLAE
        result, _ = hard_slae.richardsonSolve()

        if test_log:
            print(f"result = \n{result}")
            print("<==================================>")

    # uncomment to see that the result is right
    # print("Check result validity: (must be equal to rhs vector) \n")
    # print(Matrix(matrix) * result)

    return (result, rhs_vector, cond)


def task_2_lab(size):
    _, rand_res_norms = testRandomTau(size)
    _, ofb_res_norms = testOptimalFromBounds(size)
    _, ofe_res_norms = testOptimalFromExact(size)

    _, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('log(residual norm)')
    ax.set_title(
        f"Correlation between iteration number and log of residual norm for {size = }")

    ax.grid(which='major', color='k', linestyle=':')
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle=':')

    plt.plot(range(len(rand_res_norms)),
             log(rand_res_norms), label="Random tau")
    plt.plot(range(len(ofb_res_norms)), log(ofb_res_norms),
             label="Optimal tau from eigenvals bounds")
    plt.plot(range(len(ofe_res_norms)), log(ofe_res_norms),
             label="Optimal tau from exact eigenvals")

    plt.legend()
    plt.savefig("task_2_lab.png")


if __name__ == "__main__":
    # testSimpleSLAE()
    # testHardSLAE("good_matrix18.txt",
    #              test_log="<========= Hard SLAE ==========>")
    task_2_lab(100)

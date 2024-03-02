from copy import deepcopy
import math
from random import randint
from random import uniform
from re import split


class Matrix:
    """
        Class of a real-valued matrix
    """

    def __init__(self, matrix):
        """
            Constructs a matrix from a 2D array or other matrix

            :matrix: Matrix or 2D array representing a matrix
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

            :width: width of matrix
            :height: height of matrix
        """

        matrix = []

        for _ in range(height):
            matrix.append([0] * width)

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

    def norm(self):
        """
            Calculate Frobenius norm
        """

        sum_of_squares = 0

        for row in self.__matrix:
            for elem in row:
                sum_of_squares += elem * elem

        return math.sqrt(sum_of_squares)

    def __getitem__(self, row):
        """
            Get or set matrix's item at [row][col]
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

            :row: needed row
            :new_val: new value to assing
        """

        self.__matrix[row] = new_val

    def __add__(self, other_mat):
        """
            Add other matrix to self

            :other_mat: matrix to add
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

            :other_mat: matrix to subtract
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

            :other_mat: matrix to multiply with
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

    def getTransposed(self):
        """
            Makes a transposed matrix from self
        """

        transposed = Matrix.fromSizes(self.__width, self.__height)

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

    def choleskyDecompose(self):
        """
            Decomposes a symmetric matrix A as L * L^T,
            where L is a lower triangular matrix.

            :matrix: symmetric matrix to decompose
            :return: lower triangular matrix C
        """

        if self.__height == 0:
            # return an empty matrix
            return []

        # validate that matrix is square
        if self.__height != self.__width:
            # error: matrix is not square
            return None

        # validate that matrix is symmetric
        if not self.__checkSymmetry():
            # error: matrix is nonsymmetrical
            return None

        # validate that matrix is positive definite
        if not self.__checkPositiveDefinite():
            return None

        # now width and height are equal
        # initialise resulting matrix L
        result_mat = Matrix.fromSizes(self.__height, self.__width)

        # initialise l_1,1
        result_mat[0][0] = math.sqrt(self.__matrix[0][0])

        for row in range(self.__height):
            for col in range(row):
                # initialise first column of matrix L
                if col == 0:
                    result_mat[row][col] = self.__matrix[row][0] / \
                        result_mat[0][0]
                    continue

                # initialise rest of element in the row, excluding main diagonal
                # helper sum
                prod_sum = 0

                for idx in range(col):
                    prod_sum += result_mat[col][idx] * result_mat[row][idx]

                result_mat[row][col] = (
                    self.__matrix[row][col] - prod_sum) / result_mat[col][col]

            # initialise element on the main diagonal
            # get sum of squares of precedent elements in the same row of L
            squares_sum = 0

            for col in range(row):
                squares_sum += result_mat[row][col] ** 2

            # initialise diagonal element
            result_mat[row][row] = math.sqrt(
                self.__matrix[row][row] - squares_sum)

        return result_mat

    def __checkSymmetry(self):
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

            :matrix: left hand side of the SLAE
            :rhs_vector: right hand side of the SLAE 
        """

        # construct a matrix from 2D array
        self.__matrix = Matrix(matrix)

        # construct a vector-like matrix from an array
        self.__vector = Matrix(rhs_vector) if type(rhs_vector) is Matrix \
            else Matrix([[elem] for elem in rhs_vector])

    def __solveBasic(self, reverse=False):
        """
            Solves basic SLAE Ax = f, where A is a triangle matrix

            :reverse: True if A is lower triangle, False if upper
        """

        # NOTE: no check that A is triangle

        equations_amt = self.__vector.size()[0]

        # make result vector as an array
        result_vec = [0] * equations_amt

        # forward or backward depending on reverse flag
        row_order = range(equations_amt - 1,
                          -1, -1) if reverse else range(equations_amt)

        # start with equation with one variable,
        # continue with two, etc
        for row_idx in row_order:
            # range in which to sum known elements
            sum_range = range(row_idx + 1,
                              equations_amt) if reverse else range(row_idx)

            # sum of known elements
            known_sum = 0

            for elem_idx in sum_range:
                known_sum += result_vec[elem_idx] * \
                    self.__matrix[row_idx][elem_idx]

            # NOTE: zero on diagonal not checked

            result_vec[row_idx] = (self.__vector[row_idx][0] - known_sum) / \
                self.__matrix[row_idx][row_idx]

        return Matrix([[elem] for elem in result_vec])

    def choleskySolve(self):
        """
            Solves the equation Ax = f
        """

        # NOTE: no check on Nones during the solution

        # left-multiply the SLAE on A^T
        self.__symmetrise()

        # get lower triangle matrix L in cholesky decomposition
        lower_triang = self.__matrix.choleskyDecompose()

        # solve the equation Ly = f
        first_basic_slae = SLAE(lower_triang, self.__vector)

        # get the solution to the equation Ly = f
        midterm_result = first_basic_slae.__solveBasic()

        # get transposed lower triangle matrix L in cholesky decomposition
        lower_triang_transposed = lower_triang.getTransposed()

        # solve the equation L^T x = y,
        # where y was obtained on the previous step
        second_basic_slae = SLAE(lower_triang_transposed, midterm_result)

        # get the solution to the SLAE
        result = second_basic_slae.__solveBasic(reverse=True)

        return result

    def __symmetrise(self):
        """
            Left-multiplies the SLAE on A^T, transformig it into 
            A^T * A * x = A^T * f
        """

        transposed = self.__matrix.getTransposed()
        self.__matrix = transposed * self.__matrix
        self.__vector = transposed * self.__vector

    def getMatrixCond(self):
        """
            Get SLAE's matrix conditionality 
        """

        return self.__matrix.conditionality()

    def __repr__(self):
        """
            String representation of a SLAE
        """

        return f"matrix = \n{self.__matrix}\nrhs vector = \n{self.__vector}"


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
                [1, 2, 3],
                [3, 5, 7],
                [1, 3, 4]
            ],
            [3, 0, 1]
        ),
        SLAE(
            [
                [1, -1, 3, 1],
                [4, -1, 5, 4],
                [2, -2, 4, 1],
                [1, -4, 5, -1]
            ],
            [5, 4, 6, 3]
        )
    ]

    test_numbering = iter(range(1, len(simple_slaes) + 1))
    for slae in simple_slaes:
        print(f"<------- Test case #{next(test_numbering)} ------->")
        print(slae)

        result = slae.choleskySolve()

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
        result = hard_slae.choleskySolve()

        if test_log:
            print(f"result = \n{result}")
            print("<==================================>")

    # uncomment to see that the result is right
    # print("Check result validity: (must be equal to rhs vector) \n")
    # print(Matrix(matrix) * result)

    return (result, rhs_vector, cond)


def task_1_lab():
    # make delta vector
    delta_vector = [uniform(1e-3, 9e-3) for _ in range(20)]
    delta_vector_norm = Matrix([[item] for item in delta_vector]).norm()

    # get result for Ax = f
    (bad_result, bad_rhs, bad_cond) = testHardSLAE("bad_matrix18.txt",
                                                   test_log="<====== Плохо обусловленная матрица ======>",
                                                   conditionality=True)

    # get result for A(x + dx) = f + df
    (bad_result_with_delta, _, _) = testHardSLAE(
        "bad_matrix18.txt", delta_vector=delta_vector, rhs_vector=bad_rhs)

    # get difference norm
    bad_diff_norm = (bad_result_with_delta - bad_result).norm()

    # get result for By = g
    (good_result, good_rhs, good_cond) = testHardSLAE("good_matrix18.txt",
                                                      test_log="<====== Хорошо обусловленная матрица ======>",
                                                      conditionality=True)

    # get result for B(y + dy) = g + dg
    (good_result_with_delta, _, _) = testHardSLAE(
        "good_matrix18.txt", delta_vector=delta_vector, rhs_vector=good_rhs)

    # get difference norm
    good_diff_norm = (good_result_with_delta - good_result).norm()

    print("Delta vector: ")

    for elem in delta_vector:
        print(f"\t{elem:.4f}")

    print(f"\nDelta vector norm = {delta_vector_norm}\n")

    print(f"\nRelative error for bad SLAE is no greater than "
          f"{bad_cond * delta_vector_norm / Matrix([[item] for item in bad_rhs]).norm():.6f}\n")

    print(f"Relative error for good SLAE is no greater than "
          f"{good_cond * delta_vector_norm / Matrix([[item] for item in good_rhs]).norm():.6f}\n")

    print(f"Difference norm for bad matrix is {bad_diff_norm:.3f}\n")
    print(f"Relative error for bad matrix is "
          f"{bad_diff_norm / bad_result.norm():.6f}\n")

    print(f"Difference norm for good matrix is {good_diff_norm:.3f}\n")
    print(f"Relative error for good matrix is "
          f"{good_diff_norm / good_result.norm():.6f}\n")


if __name__ == "__main__":
    testSimpleSLAE()
    testHardSLAE("bad_matrix18.txt",
                 test_log="<======== Testing hard SLAE ========>")

    # task_1_lab()

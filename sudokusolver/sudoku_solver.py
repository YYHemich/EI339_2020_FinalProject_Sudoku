from sudokusolver.DancingList import DanceSodukuList
import numpy as np
import time


class SudokuSolver:
    def __init__(self, puzzle):
        self.danceList = DanceSodukuList()
        self.puzzle = puzzle
        self.copy_puzzle = puzzle.copy()
        assert self.puzzle.shape == (9, 9)

        self.init = []
        self.valid_bit = 0
        self.solution_bit = 0
        self.solution = None

    def show_puzzle(self):
        print('Initial puzzle board is')
        print("+ - - - + - - - + - - - +")
        for i in range(9):
            print("| ", end='')
            for j in range(9):
                digit = self.puzzle[i, j] if self.puzzle[i, j] != 0 else ' '
                print("%s " % digit, end='')
                if (j + 1) % 3 == 0:
                    print("| ", end='')
            print()
            if (i + 1) % 3 == 0:
                print("+ - - - + - - - + - - - +")

    def show_result(self):
        print('[DIFFICULTY] ', end='')
        if self.valid_bit:
            print("Invalid puzzle.\nA possible solution is")
        elif self.solution_bit == 1:
            print("Unsolvable puzzle.\nChange %s number. A possible solution is" % self.danceList.pointer)
        elif self.solution_bit == 2:
            print("Unexpected exception occur! There might be bugs in this solver!")
            self.show_puzzle()
            return
        else:
            print("Solvable puzzle.")
        print("+ - - - + - - - + - - - +")
        for i in range(9):
            print("| ", end='')
            for j in range(9):
                digit = self.solution[i, j] if self.solution[i, j] != 0 else ' '
                print("%s " % digit, end='')
                if (j + 1) % 3 == 0:
                    print("|", end=' ')
            print()
            if (i + 1) % 3 == 0:
                print("+ - - - + - - - + - - - +")

    def solve(self):
        self.valid_bit = self.conflict_detect(self.copy_puzzle)

        initial = SudokuSolver.constrain_priority_sort_cvt(self.copy_puzzle)  # Standard heuristic
        # initial = SudokuSolver.constrain_priority_sort_cvt_stronger(self.copy_puzzle)  # Stronger heuristic
        self.solution_bit = self.danceList.solve(initial)

        if self.solution_bit < 2:
            solution = np.zeros((9, 9), dtype='uint8')
            for row_num in self.danceList.solution:
                i, j, k = SudokuSolver._cvtRownumToijk(row_num)
                solution[i-1, j-1] = k
            self.solution = solution
            return solution

    @staticmethod
    def cvtBoard(puzzle):
        initial = []
        for i in range(9):
            for j in range(9):
                num = puzzle[i, j]
                if num != 0:
                    row_num = SudokuSolver._cvtRowNum(i + 1, j + 1, num)
                    initial += [row_num]
        return initial

    @staticmethod
    def _cvtRowNum(i, j, k):
        return (i - 1) * 81 + (j - 1) * 9 + k

    @staticmethod
    def _cvtRownumToijk(row_num):
        k = (row_num - 1) % 9 + 1
        row_num = (row_num - k) // 9
        j = row_num % 9 + 1
        row_num -= (j - 1)
        i = row_num // 9 + 1
        return i, j, k

    @staticmethod
    def conflict_detect(puzzle):
        initial = SudokuSolver.cvtBoard(puzzle)
        valid = 0
        while True:
            # conflict = [0] * len(initial)
            conflict = np.zeros((len(initial),), dtype='uint8')
            # constraint = [0] * len(initial)
            for index, row_num in enumerate(initial):
                i, j, k = SudokuSolver._cvtRownumToijk(row_num)

                block_i, block_j = (i - 1) // 3 * 3, (j - 1) // 3 * 3
                block = np.delete(puzzle[block_i:block_i+3, block_j:block_j+3], i - block_i - 1, axis=0)
                block = np.delete(block, j-block_j-1, axis=1)

                tmp = (np.sum(puzzle[i - 1, :] == k) +
                       np.sum(puzzle[:, j - 1] == k) - 2 +
                       np.sum(block == k))
                conflict[index] = tmp

            if np.sum(conflict) == 0:
                return valid
            valid = 1
            pop_index = np.argmax(conflict)
            remove_row = initial.pop(int(pop_index))
            r_i, r_j, _ = SudokuSolver._cvtRownumToijk(remove_row)
            puzzle[r_i-1, r_j-1] = 0

    @staticmethod
    def constrain_priority_sort_cvt(puzzle):
        initial = SudokuSolver.cvtBoard(puzzle)
        constraint = [0] * len(initial)
        for index, row_num in enumerate(initial):
            i, j, k = SudokuSolver._cvtRownumToijk(row_num)

            block_i, block_j = (i - 1) // 3 * 3, (j - 1) // 3 * 3
            block = np.delete(puzzle[block_i:block_i + 3, block_j:block_j + 3], i - block_i - 1, axis=0)
            block = np.delete(block, j - block_j - 1, axis=1)

            tmp = (np.sum(puzzle[i - 1, :] == 0) +
                   np.sum(puzzle[:, j - 1] == 0) +
                   np.sum(block == 0) - 2)
            constraint[index] = tmp
        sort_li = sorted(list(zip(constraint, initial)), key=lambda x: x[0])
        return [pair[1] for pair in sort_li]

    @staticmethod
    def constrain_priority_sort_cvt_stronger(puzzle_in):
        puzzle = puzzle_in.copy()
        initial = SudokuSolver.cvtBoard(puzzle)
        init_l = len(initial)
        sort_init = []
        for i in range(init_l):
            constraint = 0
            target_index = -1
            for index, row_num in enumerate(initial):
                i, j, k = SudokuSolver._cvtRownumToijk(row_num)

                block_i, block_j = (i - 1) // 3 * 3, (j - 1) // 3 * 3
                block = np.delete(puzzle[block_i:block_i + 3, block_j:block_j + 3], i - block_i - 1, axis=0)
                block = np.delete(block, j - block_j - 1, axis=1)

                tmp = (np.sum(puzzle[i - 1, :] == 0) +
                       np.sum(puzzle[:, j - 1] == 0) +
                       np.sum(block == 0) - 2)
                if tmp > constraint:
                    constraint = tmp
                    target_index = index
            pop_num = initial.pop(target_index)
            i_r, j_r, _ = SudokuSolver._cvtRownumToijk(pop_num)
            puzzle[i_r-1, j_r-1] = 0
            sort_init += [pop_num]
        return sort_init[::-1]


if __name__ == '__main__':
    # Solvable example 1
    # init = [[0, 6, 1, 0, 3, 0, 0, 2, 0],
    #         [0, 5, 0, 0, 0, 8, 1, 0, 7],
    #         [0, 0, 0, 0, 0, 7, 0, 3, 4],
    #         [0, 0, 9, 0, 0, 6, 0, 7, 8],
    #         [0, 0, 3, 2, 0, 9, 5, 0, 0],
    #         [5, 7, 0, 3, 0, 0, 9, 0, 0],
    #         [1, 9, 0, 7, 0, 0, 0, 0, 0],
    #         [8, 0, 2, 4, 0, 0, 0, 6, 0],
    #         [0, 4, 0, 0, 1, 0, 2, 5, 0]]

    # Solvable example 2
    # init = [[0, 0, 2, 0, 0, 6, 0, 0, 0],
    #         [0, 0, 8, 2, 0, 0, 0, 0, 9],
    #         [4, 0, 6, 0, 9, 0, 0, 0, 0],
    #         [2, 0, 0, 0, 0, 5, 0, 6, 0],
    #         [0, 0, 7, 6, 8, 0, 0, 0, 5],
    #         [0, 3, 5, 0, 0, 0, 8, 2, 0],
    #         [0, 5, 0, 0, 4, 1, 0, 9, 0],
    #         [0, 6, 0, 8, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 3, 0]]

    # Unsolvable example
    init = [[0, 0, 7, 1, 4, 6, 2, 3, 5],
            [2, 3, 5, 7, 0, 0, 1, 4, 6],
            [6, 4, 1, 2, 3, 5, 0, 0, 7],
            [1, 6, 0, 5, 0, 7, 3, 2, 4],
            [7, 2, 3, 0, 6, 4, 5, 1, 0],
            [5, 0, 4, 3, 2, 1, 7, 6, 0],
            [4, 1, 6, 0, 5, 3, 0, 7, 2],
            [0, 7, 2, 4, 1, 0, 6, 5, 3],
            [3, 5, 0, 6, 7, 2, 4, 0, 1]]

    # Invalid example
    # init = [[0, 0, 2, 0, 0, 6, 0, 0, 0],
    #         [0, 0, 8, 2, 0, 0, 0, 0, 9],
    #         [4, 0, 6, 0, 9, 0, 0, 0, 0],
    #         [2, 0, 0, 0, 0, 5, 0, 6, 0],
    #         [0, 0, 6, 6, 8, 0, 0, 0, 5],
    #         [0, 3, 5, 0, 0, 0, 8, 2, 0],
    #         [0, 5, 0, 0, 4, 1, 0, 9, 0],
    #         [0, 6, 0, 8, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 3, 0]]

    init_z = np.zeros((9, 9), dtype='int')
    init_z[1, 1] = 7
    init_z[1, 6] = 7

    sudoku = np.array(init)
    solver = SudokuSolver(sudoku)
    # SudokuSolver.conflict_detect(init_z)
    # print(init_z)
    solver.show_puzzle()
    print()
    start = time.time()
    solution = solver.solve()
    end = time.time()
    solver.show_result()
    print('Time: %ss' % (end - start))

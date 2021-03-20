import numpy as np
import time


class DanceNode:
    def __init__(self, row, col, up=None, down=None, left=None, right=None, row_head=False, col_head=False):
        self.row = row
        self.col = col
        self.up = self if up is None else up
        self.down = self if down is None else down
        self.left = self if left is None else left
        self.right = self if right is None else right

        self.isRowHead = row_head
        self.isColHead = col_head
        self.hasFolded = False

    def foldCol(self):
        if self.isRowHead or self.hasFolded:
            return
        self.left.right = self.right
        self.right.left = self.left
        self.hasFolded = True

    def foldRow(self):
        if self.isColHead or self.hasFolded:
            return
        self.up.down = self.down
        self.down.up = self.up
        self.hasFolded = True

    def unfoldCol(self):
        if self.isRowHead or not self.hasFolded:
            return
        self.left.right = self
        self.right.left = self
        self.hasFolded = False

    def unfoldRow(self):
        if self.isColHead or not self.hasFolded:
            return
        self.up.down = self
        self.down.up = self
        self.hasFolded = False


class DanceSodukuList:
    def __init__(self, h=9*9*9, w=9*9*4):
        self.row = h
        self.col = w

        self.column_head = []
        self.row_head = []

        self.column_tail = []
        self.row_tail = []
        self.choices = []

        self.head = DanceNode(0, 0, row_head=True, col_head=True)
        self.column_head.append(self.head)
        self.row_head.append(self.head)
        self.column_tail.append(self.head)
        self.row_tail.append(self.head)

        self._init_col()
        self._init_row()
        self._init_digits()
        self._init_choices()

        self.solution = []
        self.pointer = 0

    def _init_col(self):
        for i in range(1, self.col + 1):
            node = DanceNode(0, i, left=self.row_tail[0], right=self.row_head[0], col_head=True)
            self.row_head[0].left = node
            self.row_tail[0].right = node
            self.row_tail[0] = node
            self.column_head.append(node)
            self.column_tail.append(node)

    def _init_row(self):
        for i in range(1, self.row + 1):
            node = DanceNode(i, 0, up=self.column_tail[0], down=self.column_head[0], row_head=True)
            self.column_head[0].up = node
            self.column_tail[0].down = node
            self.column_tail[0] = node
            self.row_head.append(node)
            self.row_tail.append(node)

    def _init_digits(self):
        for i in range(1, 10):
            for j in range(1, 10):
                for k in range(1, 10):
                    in_c = self._cvtInConstraint(i, j, k)
                    r_c = self._cvtRowConstraint(i, j, k)
                    c_c = self._cvtColConstraint(i, j, k)
                    ce_c = self._cvtCellConstraint(i, j, k)
                    col_li = [in_c, r_c, c_c, ce_c]
                    row_num = self._cvtRowNum(i, j, k)
                    self.appendRow(row_num, 4, col_li)

    def _init_choices(self):
        for i in range(0, len(self.column_head)):
            self.choices += [self.traverseColForward(i, verbose=False)]

    def appendRow(self, row_num, length, col_li):
        assert length == len(col_li)
        for i in range(length):
            col_num = col_li[i]
            node = DanceNode(row_num, col_num,
                             up=self.column_tail[col_num], down=self.column_head[col_num],
                             left=self.row_tail[row_num], right=self.row_head[row_num])

            self.column_head[col_num].up = node
            self.column_tail[col_num].down = node
            self.column_tail[col_num] = node

            self.row_head[row_num].left = node
            self.row_tail[row_num].right = node
            self.row_tail[row_num] = node

    def _cvtInConstraint(self, i, j, k=None):
        return (i - 1) * 9 + j

    def _cvtRowConstraint(self, i, j, k):
        return 81 + (i - 1) * 9 + k

    def _cvtColConstraint(self, i, j, k):
        return 81 * 2 + (j - 1) * 9 + k

    def _cvtCellConstraint(self, i, j, k):
        cell_num = self._cvtIJtoCell(i, j)
        return 81 * 3 + (cell_num - 1) * 9 + k

    def _cvtRowNum(self, i, j, k):
        return (i - 1) * 81 + (j - 1) * 9 + k

    def _cvtIJtoCell(self, i, j):
        cell_row = (i - 1) // 3 + 1
        cell_col = (j - 1) // 3 + 1
        return (cell_row - 1) * 3 + cell_col

    def select_row(self, row_num):
        node = self.row_head[row_num].right
        while node != self.row_head[row_num]:
            self.delete_col(node.col)
            node = node.right

    def unselect_row(self, row_num):
        node = self.row_head[row_num].left
        while node != self.row_head[row_num]:
            self.restore_col(node.col)
            node = node.left

    def delete_col(self,col_num):
        node = self.column_head[col_num].down
        self.column_head[col_num].foldCol()
        while node != self.column_head[col_num]:
            sub_node = node.right
            while sub_node != node:
                sub_node.foldRow()
                self.choices[sub_node.col] -= 1  # 优先级
                sub_node = sub_node.right
            node = node.down

    def restore_col(self, col_num):
        node = self.column_head[col_num].up
        self.column_head[col_num].unfoldCol()
        while node != self.column_head[col_num]:
            sub_node = node.left
            while sub_node != node:
                sub_node.unfoldRow()
                self.choices[sub_node.col] += 1  # 优先级
                sub_node = sub_node.left
            node = node.up

    def select_best_col_num(self):
        mini = 9*9*10
        best_i = 0
        for i, num in enumerate(self.choices[1:]):
            if num == 0 and not self.column_head[i+1].hasFolded:
                return -1
            if num < mini and num != 0 and not self.column_head[i+1].hasFolded:
                mini = num
                best_i = i + 1
        return best_i

    def backtrack(self):
        node = self.head.right
        if node == self.head:
            return True

        col_num = self.select_best_col_num()
        if col_num == -1:
            return False
        node = self.column_head[col_num]

        sub_node = node.down
        if sub_node == node:
            return False

        self.delete_col(node.col)
        while sub_node != node:
            self.solution.append(sub_node.row)
            self.select_row(sub_node.row)
            if self.backtrack():
                return True
            self.unselect_row(sub_node.row)
            self.solution.pop()
            sub_node = sub_node.down
        self.restore_col(node.col)
        return False

    def solve(self, initial):
        self.solution = initial[:]

        # 初始化
        for row_num in initial:
            self.select_row(row_num)

        solution_state = 0

        while self.solution:
            hasSolution = self.backtrack()
            if hasSolution:
                return solution_state
            else:
                solution_state = 1
                row_num = self.solution.pop()
                self.unselect_row(row_num)
                self.pointer += 1

        solution_state = 2
        return solution_state

    def traverseRowForward(self, row_num, verbose=True):
        cnt = 0
        node = self.row_head[row_num]
        while True:
            if verbose: print((node.row, node.col))
            cnt += 1
            node = node.right
            if node == self.row_head[row_num]:
                break
        return cnt

    def traverseRowBackward(self, row_num, verbose=True):
        cnt = 0
        node = self.row_head[row_num]
        while True:
            cnt += 1
            if verbose: print((node.row, node.col))
            node = node.left
            if node == self.row_head[row_num]:
                break
        return cnt

    def traverseColForward(self, col_num, verbose=True):
        cnt = 0
        node = self.column_head[col_num]
        while True:
            if verbose: print((node.row, node.col))
            cnt += 1
            node = node.down
            if node == self.column_head[col_num]:
                break
        return cnt

    def traverseColBackward(self, col_num, verbose=True):
        cnt = 0
        node = self.column_head[col_num]
        while True:
            if verbose: print((node.row, node.col))
            cnt += 1
            node = node.up
            if node == self.column_head[col_num]:
                break
        return cnt

import SudokuReader
import SudokuSolver

def readAndSolveSudoku():
    board = SudokuReader.readSudoku("Test Images/sudoku3.jpg")    

    SudokuSolver.printBoard(board, "Interpreted State of Board")
    print("\n\n")

    SudokuSolver.solveBoard(board, 0, -1)

if __name__ == "__main__":
    readAndSolveSudoku()

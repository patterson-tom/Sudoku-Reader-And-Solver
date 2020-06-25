import math

#prints the 2d board array in a readable format
def printBoard(board, name):
    rowCount=-1
    colCount=-1
    print(name.upper() + ":")
    for b in board:
        colCount=-1
        rowCount+=1
        if rowCount == 3:
            print("-"*21)
            rowCount=0
        for a in b:
            colCount+=1
            if colCount==3:
                print("|", end=" ")
                colCount=0
            print(a, end=" ")
        print("")



#checks a move is valid according to sudoku rules
def isMoveValid(board, x, y):

    #checks there are no duplicates in this row
    row = board[x]
    numZeroes = row.count(0)
    if numZeroes > 0:
        numZeroes -= 1
    if len(row) != len(set(row)) + numZeroes:
        return False

    #checks there are no duplicates in this column
    col = []
    for i in range(len(board)):
        col.append(board[i][y])
    numZeroes = col.count(0)
    if numZeroes > 0:
        numZeroes -= 1
    if len(col) != len(set(col)) + numZeroes:
        return False

    #checks there are no duplicates in this mini-grid (box)
    boxWidth = int(math.sqrt(len(board)))
    boxX = x //boxWidth * boxWidth
    boxY = y //boxWidth * boxWidth
    box = []
    for i in range(boxWidth):
        for j in range(boxWidth):
            box.append(board[i+boxX][j+boxY])
    numZeroes = box.count(0)
    if numZeroes > 0:
        numZeroes -= 1
    if len(box) != len(set(box)) + numZeroes:
        return False
    
    return True

#Backtracking algorithm to solve the sudoku grid
def solveBoard(board, x, y):
    if not isMoveValid(board, x, y): #if the last move wasn't valid
        return False
    elif x == len(board)-1 and y == len(board)-1: #if we've reached the end of the board, and found a solution
        printBoard(board, "Solved Board")
        return False #change to True in order to only find first solution

    #increment xy co-ords
    y = y + 1
    if y == len(board):
        y = 0
        x += 1
    
    if board[x][y] == 0: #if this square is blank

        for i in range(len(board)): #try each value 1-9 in this spot before returning False
            board[x][y] = i+1
            if solveBoard(board, x, y): #tries this one
                return True
        board[x][y] = 0 #if it doesn't work, undo changes
        return False

    else:
        return solveBoard(board, x, y) #if this square is filled in, move on with your life

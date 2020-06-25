import cv2
import numpy as np
from shapely.geometry import LineString, Point
import pytesseract
import SudokuSolver

#Given an image, attempts to find and read the sudoku grid at the centre of it
#Uses openCV's Canny edge detection + Hough lines to find all the lines in the image
#Filters these down through various metrics to find the ones composing the sudoku grid
#Finds the intersection points of all these lines
#Uses these intersection points to find the edges of and crop out each square
#Runs Tesseract OCR on the squares to find the number within it (or lack thereof)
#Returns a 2d array of the read sudoku grid (with 0s as blank squares)
def readSudoku(filename):
    image = loadImage(filename)
    edges, threshed = preprocessImage(image)

    #Sorted into horizontal and vertical lines to make calculating the grid of intersection points easier
    hlines, vlines = getSudokuGridLines(edges, image) 
    drawLines(hlines + vlines, image)

    #a 2D grid of tuples. Then a square is defined by its top left intersection point and the bottom right one
    intersections = findIntersectionPoints(hlines, vlines)
    drawIntersectionPoints(intersections, image)
    
    cv2.imshow("Result", image)
    cv2.waitKey(1) #wait 1 ms after showing so that it actually appears

    board = constructBoard(threshed, intersections)
    return board

#Finds all the sudoko grid lines, separated into horizontal and vertical lines
def getSudokuGridLines(edges, image):
    lines = findAllHoughLines(edges)
    lines = removeDuplicates(lines)
    hlines, vlines = sortLinesByOrientation(lines)
    hlines, vlines = filterOutAdditionalLines(hlines, vlines, image)
    return (hlines, vlines)

#loads the image and downscales it to a sensible size if necessary so they fit on the monitor
def loadImage(filename):
    image = cv2.imread(filename)
    if image.shape[1] > 560 or image.shape[0] > 1000:
        image = cv2.resize(image, (560, 1000))

    return image

#computes the Canny edge detected image and the binary image, used for hough line detection and OCR respectively
def preprocessImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200, apertureSize=3)
    threshed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 1)

    return (edges, threshed)

#Just calls the openCV function
def findAllHoughLines(edges):
    return cv2.HoughLines(edges, 1, 3.1415/180, 125)


#filter out duplicate/very similar lines.
#If any two lines have similar rho and theta values, one is removed
def removeDuplicates(lines):
    dropList = set()
    for i, l1 in enumerate(lines):
        r1, t1 = l1[0]
        if i in dropList:
            continue
        
        for j, l2 in enumerate(lines):
            r2, t2 = l2[0]
            if i == j:
                continue

            if j in dropList:
                continue

            if abs(r1-r2) < 15 and abs(t1-t2) < 0.05:
                dropList.add(j)

    lines = lines.tolist()
    for i in range(len(lines)-1, -1, -1):
        if i in dropList:
            del lines[i]

    return lines

#partition the lines into horizontal and vertical lines.
#also converts the lines to cartesian coordinates and stores them in LineString objects for later geometry calculations
#rho, theta and x0 values are stored along with each line as they are needed for later analysis
def sortLinesByOrientation(lines):
    
    hlines = []
    vlines = []
    
    #find coord end points of lines, and organise them into horizontal/vertical
    for i in range(len(lines)):
        
        rho, theta = lines[i][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        
        line = LineString([(x1, y1), (x2, y2)])
        if theta < 3.1415/4:
            hlines.append((line, rho, theta, x0))
        else:
            vlines.append((line, rho, theta, y0))

    return (hlines, vlines)
    
#filter out unrealistic lines that aren't part of the sudoku grid
def filterOutAdditionalLines(hlines, vlines, image):
    for linelist in [hlines, vlines]:

        #if we already only have 10 lines then don't bother
        if len(linelist) <= 10:
            continue

        #all the lines in each list should have very similar theta values (ideally they should be parallel)
        #So we can remove the largest outlier until they do
        while thetaRange(linelist) > 0.1 and len(linelist) > 10:
            thetas = [x[2] for x in linelist]
            avg = sum(thetas)/len(thetas)

            maxDist = -1
            maxI = None
            for i in range(len(linelist)):
                dist = abs(linelist[i][2] - avg)
                if dist > maxDist:
                    maxDist = dist
                    maxI = i
            del linelist[maxI]

        #if further processing is still required
        if len(linelist) <= 10:
            continue

        #All lines are now parallel, so unneeded lines are likely outside of the grid itself
        #The grid is likely to be in the centre of the image
        #So we just keep the 10 lines closest to the centre
        #This is kind of crude but it seems to work
        midPoint = Point(image.shape[1]/2, image.shape[0]/2)
        linelist.sort(key=lambda x : x[0].distance(midPoint))

    return (hlines[:10], vlines[:10])

#used in filterOutAdditionalLines to determine if any rotational outliers remain
def thetaRange(linelist):
        thetas = [x[2] for x in linelist]
        return max(thetas) - min(thetas)

#draw the lines onto the image
def drawLines(lines, image):
    for line in lines:
        l = line[0]
        x1, y1, x2, y2 = l.bounds
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 2)

#Since we stored the relevant coord from the midpoint of the line earlier, we can easily sort the lines into order
#Now, each list is from left to right or top to bottom, enabling us to easily find and store the intersection points
def findIntersectionPoints(hlines, vlines):
    hlines.sort(key=lambda x : x[3])
    vlines.sort(key=lambda x : x[3])

    intersections = []

    for l1 in hlines:
        myIntersections = []
        for l2 in vlines:
            point = l1[0].intersection(l2[0])
            myIntersections.append((point.x, point.y))
        intersections.append(myIntersections)

    return intersections

#draw the intersection points onto the image
def drawIntersectionPoints(intersections, image):
    for a in intersections:
        for b in a:
            cv2.line(image, (int(b[0]), int(b[1])), (int(b[0]), int(b[1])), (0, 0, 255), 5)

#Now that we have the intersection points defining the edges of each square, we read each square
#returns the final expected board state
def constructBoard(threshed, intersections):
    board = [[0 for x in range(9)] for y in range(9)]

    for i in range(9):
        for j in range(9):
            
            x1, y1 = intersections[i][j]
            x2, y2 = intersections[i+1][j+1]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            #crops out the relevant square
            square = threshed[y1:y2, x1:x2]

            #if we don't think there is a number in the square, then write a 0 into the board position and move on
            if not hasChar(square, x1, x2, y1, y2):
                board[j][i] = 0
                #save this square's image for debugging purposes
                cv2.imwrite("logs/"+str(i)+"-"+str(j)+".jpg", square)
                continue

            #when the square was cropped above, some of the grid lines may have been included
            #tesseract really can't stand this so we have to remove them
            square = cropLinesAtEdges(square, x1, x2, y1, y2)
            
            #saves this square's image for debugging purposes
            cv2.imwrite("logs/"+str(i)+"-"+str(j)+".jpg", square) 

            #runs the tesseract OCR on the square, --psm 10 specifies that the image contains a single character
            t = pytesseract.image_to_string(square, config="--psm 10")

            if len(t) == 1 and t.isnumeric(): #if valid number
                board[j][i] = int(t)
            else:
                fixed = False
                #sometimes just has a '.' appended for some reason. This check saves us in that case
                if len(t) > 1:
                    for c in t:
                        if c.isnumeric():
                            board[j][i] = int(c)
                            fixed = True

                #if we couldnt get a guess, say the square is blank and pray
                if not fixed:
                    print(t, i, j)
                    board[j][i] = 0

    return board

#takes the average colour of the centre of the image
#The square is black on a white background (and binary)
#so if the average is too high then there cannot be a black character there
def hasChar(square, x1, x2, y1, y2):
    o = int((y2-y1)/4)
    
    centre = square[o:y2-y1-o, o:x2-x1-o]
    avg_color_per_row = np.average(centre, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return np.max(avg_color) <= 220

#very crude, there must be a better way to do this but idk how
#walk in from the edges, flood filling any black squares we find until we can't detect the character any more
#then rollback one iteration and we have the cropped image
def cropLinesAtEdges(square, x1, x2, y1, y2):
    copy = square.copy()
    prev = copy.copy()
    k = 0
    while hasChar(copy, x1, x2, y1, y2):
        prev = copy.copy()
        for l in range(y2-y1):
            if copy[l][k] == 0:
                cv2.floodFill(copy, np.zeros((y2-y1+2, x2-x1+2), np.uint8), (k, l), 255)
            if copy[l][x2-x1-k-1] == 0:
                cv2.floodFill(copy, np.zeros((y2-y1+2, x2-x1+2), np.uint8), (x2-x1-k-1, l), 255)
        for l in range(x2-x1):
            if copy[k][l] == 0:
                cv2.floodFill(copy, np.zeros((y2-y1+2, x2-x1+2), np.uint8), (l, k), 255)
            if copy[y2-y1-k-1][l] == 0:
                cv2.floodFill(copy, np.zeros((y2-y1+2, x2-x1+2), np.uint8), (l, y2-y1-k-1), 255)
        k += 1

    return prev

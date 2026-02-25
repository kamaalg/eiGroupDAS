def generateMatrix(n:int):
    if type(n) != int or n <= 0:
        raise ValueError("n must be a positive integer ")
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    curr = 1
    if n%2 != 0:
        x = n//2
        y = n//2
    else:
        x = n // 2-1
        y = n // 2-1

    matrix[x][y] = curr
    step = 1
    target = n*n
    while curr < target:
        #right
        for _ in range(step):
            if curr >= target: break

            y+=1
            curr+=1

            matrix[x][y] = curr
        #down
        for _ in range(step):

            if curr >= target: break
            x+=1
            curr+=1
            matrix[x][y] = curr
        step+=1
        #left
        for _ in range(step):
            if curr >= target: break
            y-=1
            curr+=1
            matrix[x][y] = curr
        #up
        for _ in range(step):
            if curr >= target: break
            x-=1
            curr+=1
            matrix[x][y] = curr
        step+=1
    return matrix


def pretty_print(matrix: list):
    n = len(matrix)
    width = len(str(n * n))

    for row in matrix:
        print(" ".join(f"{num:>{width}}" for num in row))


def first_diagonal(matrix: list):
    return sum(matrix[i][i] for i in range(len(matrix)))
def second_diagonal(matrix: list):
    n = len(matrix)
    return sum(matrix[i][n-i-1] for i in range(n))


if __name__ == "__main__":
    matrix = generateMatrix(12)
    pretty_print(matrix)
    print("First diagonal matrix sum: ", first_diagonal(matrix))
    print("Second diagonal matrix sum: ", second_diagonal(matrix))
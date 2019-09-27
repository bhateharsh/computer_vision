def possibleAbsoluteDifference(digit, K):
    possibility1 = digit - K
    possibility2 = digit + K
    if possibility1 < 0:
        return [possibility2]
    if possibility2 > 9:
        return [possibility1]
    return [possibility1, possibility2]

def solver(numArray, K, last):
    if len(numArray) == 1:
        return possibleAbsoluteDifference(numArray[0], last)
    else:
        allPoss = []
        nos = possibleAbsoluteDifference(last, K)
        print ("Solver")
        print ("numArray: ", numArray)
        print ("Nos:", nos)
        print ("Last", last)
        print ("len(numArray) = ", len(numArray))
        for num in nos:
            numArray[0] = num
            allPoss=[num]+solver(numArray[1:], K, num)
        return allPoss
def numsSameConsecDiff(N, K):
    """
    :type N: int
    :type K: int
    :rtype: List[int]
    """
    numbers = []
    numArray = [0 for i in range(N)]
    numArray[0] = 1
    last = 1
    print (numArray)
    numbers = [numArray[0]]+solver(numArray[1:], K, last)
    return numbers

a = numsSameConsecDiff(3,7)
print (a)

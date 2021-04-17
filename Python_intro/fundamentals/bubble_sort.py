# make a bubble sort:

def bubble(arr):
    for x in range(0, len(arr),1):
        print(arr[x])
        for y in range(1, len(arr)-x, 1):
            if arr[y] < arr[y-1]:
                arr[y], arr[y-1] = arr[y-1], arr[y]
    return arr



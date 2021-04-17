# Insertion Sort

def insertion_sort(arr):
    for x in range(0,len(arr),1):
        for y in range(x+1,0,-1):
            if y == len(arr):
                break
            if arr[y] < arr[y-1]:
                arr[y], arr[y-1] = arr[y-1], arr[y]
            else:
                break
    return arr

print(insertion_sort([5,7,3,8,1,2,4,6,9,15,17,14,13,16,12,11,19,10,18,0]))
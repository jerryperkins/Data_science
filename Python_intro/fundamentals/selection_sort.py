# Selection Sort

def selection_sort(arr):
    for x in range(0,len(arr), 1):
        min = arr[x]
        temp = -1
        for y in range(x,len(arr), 1):
            # print(min)
            if min > arr[y]:
                temp = y
                min = arr[y]
        if temp != -1:
            arr[x], arr[temp] = arr[temp], arr[x]
        # print("here is x", x)
        # print("Here is temp", temp)
        # print("here is arr", arr)
        # print("end of inner loop")
    return arr




print(selection_sort([17,15,5,2,-6,1,3,9,3,0,1-3,4]))
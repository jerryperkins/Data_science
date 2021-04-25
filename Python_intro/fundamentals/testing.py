# arr = [1,2,3,4,5]
# print(len(arr) / 2)
# for i in range(0, len(arr) /2, 1):
#     print(arr[i])


# print(True == False)

import timeit

empty_tuple = ()
print(empty_tuple)
empty_tuple = tuple()
print(empty_tuple)
z = (3,7,4,2) # this is tuple packing
print(z)
z = 3,7,4,2
print(z)
tup1 = ('Jerry',)
print(tup1)
not_tuple = ('Jerry')
print(not_tuple)
print(z[-1])
print(z[1:4])
print(z[3:])

tup1 = ('Python', 'SQL')
tup2 = ('R',)
print(tup1, tup2)
new_tuple = tup1 + tup2
print(new_tuple)
a, b, c, d = (3, 7, 4, 2)
print(a,b,c,d)

friends = ('Steve', 'Rachel', 'Michael', 'Monica')
for index, friend in enumerate(friends):
    print(index, friend)

animals = ('lama', 'sheep', 'lama', 48)
print(animals.index('lama'))
print(animals.count('lama'))

print(timeit.timeit('x=(1,2,3,4,5,6,7,8,9,10,11,12)', number=1000000))
print(timeit.timeit('x=[1,2,3,4,5,6,7,8,9,10,11,12]', number=1000000))


graphic_designer = {('this', 'is'), ('is', 'a'), ('a', 'sentence')}
print(graphic_designer)

webstersDict = {'person': 'a human being',
                'marathon': 'a running race that is about 26 miles',
                'resist': 'to remain strong against the force',
                'run': 'to move with haste; act quickly'}
print(webstersDict['marathon'])
webstersDict['shoe'] = 'an external covering for the human foot'
print(webstersDict)
webstersDict['marathon'] = '26 mile race'
print(webstersDict)
del webstersDict['resist'] 
print(webstersDict)
webstersDict.update({'ran': 'past tense of run', 'shoes': 'plural of shoe'})
print(webstersDict)

storyCount = {'is': 100,
            'the': 90,
            'Michael': 12,
            'runs': 5}

print(storyCount.get('Michael'))
print(storyCount.get('chicken', "you done messed up"))
print(storyCount.pop('the'))
print(storyCount)
print(type(storyCount.keys()))
key_list = list(storyCount.keys())
print(key_list)
values_list = list(storyCount.values())
print(values_list)
print(webstersDict.items())
print(list(webstersDict.items()))

dataScienceTools = ['Python', 'R', 'SQL', 'Git']
for index, tool in enumerate(dataScienceTools):
    print(index, tool)

def testing(arr):
    for x in range(0,10,1):
        print(x)
testing([3,2,5,1,45])
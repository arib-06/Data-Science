# Question 2.1

my_list = [2, 4, 6, 7, 3]

print(my_list[0])
print(my_list[-1])

print(my_list[1:4])
print(my_list[:3])

my_list[3] = 9

my_list.append(6)
my_list.insert(2, 10)
my_list.extend([7, 8])

my_list.remove(3)
my_list.pop()
my_list.pop(1)
del my_list[2]

if 2 in my_list:
    print(my_list.index(2))
    print(my_list.count(2))
else:
    print("2 not found in the list")

my_list.sort()
print("Ascending:", my_list)

my_list.sort(reverse=True)
print("Descending:", my_list)

my_list.reverse()
print("Reversed:", my_list)

my_list.clear()
print("List after clearing:", my_list)
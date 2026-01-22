# Question 2.3

s = {1, 5, 32, 54, 5, 5, 5, "Arib"}
print(s) # it doesnt print repeated items multiple times

s.add(566)
print(s)        
print(type(s))

s.remove(1)
print(s)

# s.pop()  
s.clear()   
s1 = {1, 45, 6, 78}
s2 = {7, 8, 1, 78}

print(s1.union(s2))
print(s1.intersection(s2))
print(s1.difference(s2))
print({78}.issubset(s1))
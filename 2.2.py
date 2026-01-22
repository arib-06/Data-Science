# Question 2.2

marks = { "Arib" : 91, "Sanyam" : 78, "Ankit"  : 92, "Ravinder" : 87 }

print(marks, type(marks))
print(marks["Arib"])

print(marks.items())
print(len(marks))

print(marks.keys())
print(marks.values())

marks.update({"Arib": 98})
marks.update({"Manav": 69})
print(marks)

marks1 = marks.copy()

if "Arib" in marks:
    marks.pop("Arib")

marks.clear()
print(marks)
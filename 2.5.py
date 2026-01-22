class Queue:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            print("underflow")
        else:
            print(self.items.pop(0))

    def top(self):
        if not self.items:
            print("underflow")
        else:
            print(self.items[0])

q = Queue()
q.push(1)
q.push(2)
q.push(3)
q.pop()   # prt 1
q.top()   # prt 2

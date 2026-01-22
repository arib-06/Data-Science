class MyStack:
    def __init__(self):
        self.size = 100
        self.stack = [0] * self.size #create
        self.top = -1

    def push(self, value):
        if self.top == self.size - 1:
            print("Stack is full")     #add
        else:
            self.top = self.top + 1
            self.stack[self.top] = value

    def pop(self):
        if self.top == -1:
            print("Stack is empty") #remove
        else:
            print(self.stack[self.top])
            self.top = self.top - 1

    def peek(self):
        if self.top == -1:
            print("Stack is empty")
        else:
            print(self.stack[self.top])

s = MyStack()
s.push(10)
s.push(20)
s.push(30)
s.pop()
s.peek()

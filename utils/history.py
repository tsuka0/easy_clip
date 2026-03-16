class History:

    def __init__(self):

        self.undo=[]
        self.redo=[]

    def push(self,state):

        self.undo.append(state)
        self.redo.clear()

    def undo_state(self,current):

        if not self.undo:
            return current

        self.redo.append(current)

        return self.undo.pop()

    def redo_state(self,current):

        if not self.redo:
            return current

        self.undo.append(current)

        return self.redo.pop()

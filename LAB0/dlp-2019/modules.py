def echo_print(word):
    print(word)

class ResponseNoInit:
    def response(self, word):
        echo_print(word)

class ResponseInit:
    def __init__(self, data):
        self.data = data
    def response(self, word):
        if not word=='':
            echo_print(word)
        else:
            echo_print(self.data)

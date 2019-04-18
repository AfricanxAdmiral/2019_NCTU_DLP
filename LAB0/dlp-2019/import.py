import modules
#from modules import echo_print

notInit = modules.ResponseNoInit()
print("No init class's response: ")
notInit.response('Hello World')
print("No init class's response for Empty: ")
notInit.response('')

init = modules.ResponseInit('Input Empty')
print("Init class's response: ")
init.response('Hello World')
print("Init class's response for Empty: ")
init.response('')

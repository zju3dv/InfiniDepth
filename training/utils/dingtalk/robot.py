import os
try: import requests
except: os.system('pip install requests'); import requests
try: from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
except: os.system('pip install DingtalkChatbot'); from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=56e7dc92fddd78bbc4d457a08d037e6d866664fd90ddf7100ee3a03882f07153'
secret = 'SEC1f896436f9c0fffa73a95d973d65f68d0dc9092a6c6e4d016b44a8175bf61c18'
robot = DingtalkChatbot(webhook, secret=secret)

def send_message(message):
    try: robot.send_text(msg=message)
    except: print('Failed to send message')
if __name__ == '__main__':
    send_message('test')
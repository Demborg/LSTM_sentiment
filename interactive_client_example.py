""" This is a simple example of interacting with the live sentiment server. """

import socket
import settings

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", int(settings.args.port)))

text = input("Give me some text, and I'll send it for you baby. \n")
s.send(text.encode("utf-8"))
rec = s.recv(1024)
print("Yeah baby, I've got something for you: {}".format(rec.decode("utf-8")))



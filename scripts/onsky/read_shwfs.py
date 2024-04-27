# %%
import socket
import struct

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(4)
port = 8888
remote_ip = socket.gethostbyname('real.ucolick.org')

def recv_all(s,nbytes):
    b = b''
    nrcv = 0
    while (nrcv < nbytes):
        b = b + s.recv(nbytes-nrcv)
        nrcv = len(b)
    return b

b_count = recv_all(s,8)
r_count = int(struct.unpack('1q',b_count)[0]) # was '1l'
# %%

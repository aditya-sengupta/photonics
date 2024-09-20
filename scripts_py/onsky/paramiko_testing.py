# %%
import os
import select
import sys

import paramiko
import Xlib.support.connect as xlib_connect


local_x11_display = xlib_connect.get_display(os.environ['DISPLAY'])
local_x11_display = list(local_x11_display)
local_x11_display[1] = None
local_x11_socket = xlib_connect.get_socket(*local_x11_display[:4])


ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect('karnak.ucolick.org', username='user', password='yam != spud')
transport = ssh_client.get_transport()
session = transport.open_session()
session.request_x11(single_connection=True)
session.exec_command('xterm')
x11_chan = transport.accept()

session_fileno = session.fileno()
x11_chan_fileno = x11_chan.fileno()
local_x11_socket_fileno = local_x11_socket.fileno()

poller = select.poll()
poller.register(session_fileno, select.POLLIN)
poller.register(x11_chan_fileno, select.POLLIN)
poller.register(local_x11_socket, select.POLLIN)
while not session.exit_status_ready():
    poll = poller.poll()
    if not poll: # this should not happen, as we don't have a timeout.
        break
    for fd, event in poll:
        if fd == session_fileno:
            while session.recv_ready():
                sys.stdout.write(session.recv(4096))
            while session.recv_stderr_ready():
                sys.stderr.write(session.recv_stderr(4096))
        if fd == x11_chan_fileno:
            local_x11_socket.sendall(x11_chan.recv(4096))
        if fd == local_x11_socket_fileno:
            x11_chan.send(local_x11_socket.recv(4096))

print('Exit status:', session.recv_exit_status())
while session.recv_ready():
    sys.stdout.write(session.recv(4096))
while session.recv_stderr_ready():
    sys.stdout.write(session.recv_stderr(4096))
session.close()
# %%

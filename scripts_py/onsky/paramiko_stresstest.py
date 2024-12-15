# %%
import paramiko
from time import sleep

# Update the next three lines with your
# server's information

host = "karnak.ucolick.org"
username = "user"
password = "yam != spud"

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)
client.get_transport().set_keepalive(60) # send an empty packet to keep the connection alive every 60 seconds; hopefully this fixes my crash issue
# %%
for _ in range(1000):
    _stdin, _stdout,_stderr = client.exec_command("date")
    sleep(1)
# %%
client.close()
# %%

# %%
import paramiko

# Update the next three lines with your
# server's information

host = "karnak.ucolick.org"
username = "user"
password = "yam != spud"

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)
# %%
_stdin, _stdout,_stderr = client.exec_command("date")
print(_stdout.read().decode())
# %%
client.close()
# %%

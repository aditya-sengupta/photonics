# %%
import paramiko

# Define SSH connection parameters
hostname = 'karnak.ucolick.org'
port = 22
username = 'user'
password = 'yam != spud'

# Create an SSH client
ssh_client = paramiko.SSHClient()

# Automatically add the server's host key
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the server
ssh_client.connect(hostname, port, username, password)

# Request X11 forwarding
channel = ssh_client.get_transport().open_session()
channel.get_pty()
channel.invoke_shell()
# %%
# Example: execute a command and print the output
channel.send('ls -a \n')
output = ''
while not channel.recv_ready():
    continue
while channel.recv_ready():
    output += channel.recv(1024).decode('utf-8')
print(output)
# %%
# Close the SSH connection
channel.close()
ssh_client.close()

# %%

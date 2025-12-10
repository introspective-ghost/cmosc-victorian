import paramiko
from scp import SCPClient

class LocalNetworkPicTransfer:
    def __init__(self, host, user="cmosc", port=22, keyFile="/home/cmosc/.ssh/id_ed25519"):
        """
        Initialize the SCP transfer class.
        :param host: Hostname or IP of target Pi
        :param user: SSH username. Default "cmosc"
        :param port: SSF port. Default 22
        :param keyFile: Path to private key file. Default "~/.ssh/id_ed25519"
        """
        self.host = host
        self.user = user
        self.port = port
        self.keyFile = keyFile
        self.ssh = None
        
    def connect(self):
        """ Establish SSH connection using key authentication"""
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            self.host,
            port=self.port,
            username=self.user,
            key_filename=self.keyFile
        )
    
    def sendFile(self, localPath, remotePath):
        """Send a file from local Pi to remote Pi"""
        if not self.ssh:
            self.connect()
        with SCPClient(self.ssh.get_transport()) as scp:
            scp.put(localPath, remotePath)
            
    def close(self):
        """Close SSH connection"""
        if self.ssh:
            self.ssh.close()
            self.ssh = None
import subprocess

class LocalNetworkPicTransfer:
    def __init__(self, host, user="cmosc", port=22, keyFile="/home/cmosc/.ssh/id_ed25519"):
        """
        Initialize the rsync transfer class.
        :param host: Hostname or IP of target Pi
        :param user: SSH username. Default "cmosc"
        :param port: SSH port. Default 22
        :param keyFile: Path to private key file. Default "~/.ssh/id_ed25519"
        """
        self.host = host
        self.user = user
        self.port = port
        self.keyFile = keyFile

    def sendFile(self, localPath, remotePath):
        """Send a file from local Pi to remote Pi using rsync over SSH"""
        remote = f"{self.user}@{self.host}:{remotePath}"
        cmd = [
            "rsync",
            "-av", # archive mode (preserves permissions, group, owners, etc.), verbose
            "--no-times",
            "-e", f"ssh -i {self.keyFile} -p {self.port}",  # specify SSH key and port
            localPath,
            remote
        ]
        subprocess.run(cmd, check=True)

    def close(self):
        """No persistent connection to close with rsync"""
        pass

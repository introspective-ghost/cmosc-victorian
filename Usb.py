import subprocess
import os

def get_usb_mount_point():
    try:
        # This command lists all mounted filesystems and their targets
        output = subprocess.check_output(['findmnt', '-l', '-o', 'SOURCE,TARGET,FSTYPE', '-t', 'vfat,ext4,ntfs,exfat']).decode('utf-8')
        lines = output.strip().split('\n')[1:] # Skip header
        
        for line in lines:
            parts = line.split()
            source = parts[0]
            target = parts[1]
            fstype = parts[2]
            
            # Look for devices typically associated with USB sticks
            if source.startswith('/dev/sd') and (target.startswith('/media/') or target.startswith('/run/media/')):
                return target
        return None
    except subprocess.CalledProcessError:
        print("Error running findmnt command. Ensure it's installed and accessible.")
        return None

mount_point = get_usb_mount_point()
if mount_point:
    print(f"USB stick likely mounted at: {mount_point}")
else:
    print("No USB stick mount point found.")
    
if mount_point:
    # List contents of the USB stick
    print(f"Contents of {mount_point}:")
    for item in os.listdir(mount_point):
        print(item)

    # Create a new file on the USB stick
    file_path = os.path.join(mount_point, "my_data.txt")
    with open(file_path, "w") as f:
        f.write("This is some data written to the USB stick.")
    print(f"File created: {file_path}")

    # Read from a file on the USB stick
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()
        print(f"Content of {file_path}:\n{content}")

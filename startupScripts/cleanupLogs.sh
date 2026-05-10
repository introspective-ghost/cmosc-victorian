#!/bin/bash

LogsDir="$HOME/cmosc-victorian/logs"

# search for LogsDir, if it exists, remove any entries that 
# have not been modified in the last 2 weeks

if [ -d "$LogsDir" ]; then
    find "$LogsDir" -maxdepth 1 -mtime +14 -exec rm -rf {} +
fi

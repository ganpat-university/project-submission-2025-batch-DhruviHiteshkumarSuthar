#!/bin/bash

# Define the remote server and port
SERVER="$1"
PORT="$2"

# Set default port to 22 if not provided
if [ -z "$PORT" ]; then
    PORT=22
fi

# Run ssh-keyscan to check if the server is reachable
echo "Attempting to reach $SERVER on port $PORT..."
ssh-keyscan -p "$PORT" "$SERVER" > /dev/null 2>&1

# Capture the result of ssh-keyscan and store the message
if [ $? -eq 0 ]; then
    echo "Server $SERVER is reachable on port $PORT."
else
    TITLE="Critical! The Server is down"
    MESSAGE="The SSH server at $SERVER on port $PORT is not reachable."
    DESCRIPTION="All operations on the server are impacted (Deployments). Please check urgently."
    SOURCE="https://github.com/DhruviHiteshkumarSuthar/g20_fer/blob/main/deployment/check_ssh_server.sh"
    deployment/send_slack_notification.sh "<$SOURCE|check_ssh_server.sh>" "failure" "$TITLE" "$MESSAGE" "$DESCRIPTION" "🔥"
    exit 1
fi
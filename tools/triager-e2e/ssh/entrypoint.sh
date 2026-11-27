#!/bin/sh
# Host keys live in the keys volume so the client's known_hosts stays valid across
# restarts; the analyst's authorized key is the public test key mounted read-only.
set -eu
if [ ! -f /keys/ssh_host_ed25519_key ]; then
    ssh-keygen -q -t ed25519 -N '' -f /keys/ssh_host_ed25519_key
fi
install -d -m 700 -o analyst -g analyst /home/analyst/.ssh
install -m 600 -o analyst -g analyst /client/id_ed25519.pub /home/analyst/.ssh/authorized_keys
install -d -o analyst -g analyst /data /data/runs
exec /usr/sbin/sshd -D -e -h /keys/ssh_host_ed25519_key

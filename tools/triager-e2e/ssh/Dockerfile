# A Linux analysis host for the triager SSH acceptance (task 3.2): OpenSSH with key
# login for one user, and a Rust toolchain that builds l2d from the read-only source
# mounted at /src into a volume. Nothing of the workspace is copied into the image.
FROM rust:1-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && useradd --create-home --shell /bin/bash analyst \
 && sed -i -e 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' \
           -e 's/^#\?KbdInteractiveAuthentication .*/KbdInteractiveAuthentication no/' \
           -e 's/^#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config

COPY entrypoint.sh /usr/local/bin/triager-ssh-entrypoint
RUN chmod 0755 /usr/local/bin/triager-ssh-entrypoint

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/triager-ssh-entrypoint"]

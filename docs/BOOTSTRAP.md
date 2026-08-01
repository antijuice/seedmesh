# Running a bootstrap peer

A swarm needs at least one **publicly reachable** machine. This is the part people
underestimate, so it's worth stating plainly:

> A home laptop on wifi and a Colab notebook are both behind NAT. Neither can accept an
> incoming connection. A swarm made only of those **cannot form** — there is nowhere for
> peers to find each other.

The fix is one cheap always-on box with a public IP. It is the *rendezvous*, not the
compute: a bootstrap peer relays discovery metadata and needs **no GPU**.

## What to buy

Requirements are genuinely small: 1 vCPU, 1 GB RAM, a public IPv4, and one open TCP port.
Any of these work — check current pricing, it moves.

| Provider | Rough cost | Notes |
| --- | --- | --- |
| **Hetzner Cloud** (CX22) | ~€4/mo | Best value. EU + US (Ashburn, Hillsboro) locations. |
| **Oracle Cloud Always Free** | **$0** | 4 ARM cores, 24 GB RAM, free indefinitely. Capacity in the free tier is often exhausted and signup can be fussy — worth trying first anyway. |
| **DigitalOcean** | ~$6/mo | Simplest UX if you have not done this before. |
| **Vultr / Linode** | ~$5–6/mo | Equivalent. |

Pick a region **near your friends**, not near you — latency to the bootstrap only affects
discovery, but if you later host blocks there it matters more.

Avoid Fly.io / Railway / Render for this: they are built around ephemeral containers and
constrained inbound ports, which is the opposite of what a stable rendezvous needs.

## Provision

Ubuntu 24.04, smallest size.

Do all the root-level work **first, as root**. The `seedmesh` service account is created
with `--disabled-password`, which means it has no password to type — so `sudo` from that
account can never succeed. Swap and the firewall have to be in place before you drop into
it. (Don't "fix" this by giving the account a password and sudo rights: a passwordless,
non-privileged account is the point. It runs a daemon exposed to the internet.)

```bash
ssh root@YOUR_IP

apt update && apt install -y python3-venv python3-pip git
```

**Swap, before anything downloads torch.** A 1 GB box OOMs during the install without it,
and the `fstab` line is what makes it survive a reboot:

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile
mkswap /swapfile && swapon /swapfile
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
free -h        # confirm a non-zero Swap row
```

**Firewall.** Allow SSH *before* enabling, or you lock yourself out of the box.
`--force` skips the "may disrupt existing ssh connections" prompt:

```bash
ufw allow OpenSSH
ufw allow 31337/tcp
ufw --force enable
ufw status     # confirm both rules are listed
```

Now create the unprivileged account and install as that user:

```bash
adduser --disabled-password --gecos "" seedmesh
su - seedmesh

git clone <this repo> ~/seedmesh && cd ~/seedmesh
python3 -m venv ~/.venv && ~/.venv/bin/pip install -e .
~/.venv/bin/seedmesh setup
```

`setup` clones Petals, installs dependencies, applies the Seedmesh port, and verifies it —
in that order, and the order matters: the codemod checks its symbol mapping against the
*installed* hivemind, so it cannot run before the install. Run it from the checkout, since
that is where `tools/port_petals.py` lives.

Seeing no GPU, it installs **CPU-only torch** (a few hundred MB rather than ~2.5 GB of CUDA
wheels a bootstrap peer would never use). Pass `--cpu-torch` to force that on a machine that
does have a GPU. The download is still the slow part — this is what the swap above is for.

A successful run ends with `7/7 checks passed` and `ready.` — anything else means stop and
read the output rather than continuing to the next step.

## The provider firewall is a separate thing

`ufw` above only configures the firewall *inside* the machine. If your provider has its own
(Hetzner, Oracle and AWS all do, and Oracle's is **on by default and blocks everything**),
open 31337/tcp there too, in their web console. This is the single most common reason a
bootstrap "starts fine" and nobody can reach it.

## Start it

A bootstrap peer hosts no blocks — `--num-blocks 0`. That is a real, tested configuration:
the server announces itself, reserves 0 GiB of attention cache and downloads no weights (it
still reads the model's `config.json`, so the model name must be one it can fetch).

```bash
~/.venv/bin/seedmesh serve \
  --model JackFram/llama-160m \
  --num-blocks 0 \
  --host-maddrs /ip4/0.0.0.0/tcp/31337 \
  -- --announce_maddrs /ip4/YOUR_PUBLIC_IP/tcp/31337
```

`--host-maddrs 0.0.0.0` binds every interface; `--announce_maddrs` is what it *tells* other
peers, which must be the public IP — on most VPSs the machine only sees its private address
and would otherwise advertise something unroutable.

It prints:

```
Running a server on ['/ip4/YOUR_PUBLIC_IP/tcp/31337/p2p/QmXh3hVoj...']
```

**That whole string is the bootstrap address.** Send it to your friends verbatim; it is what
they pass to `--initial-peers`.

The peer id is derived from `--identity-path` (default `~/.seedmesh/`), so it stays stable
across restarts as long as you keep that file. Losing it means a new address and everyone
re-pasting it.

## Keep it running

```bash
sudo tee /etc/systemd/system/seedmesh-bootstrap.service >/dev/null <<'EOF'
[Unit]
Description=Seedmesh bootstrap peer
After=network-online.target

[Service]
User=seedmesh
WorkingDirectory=/home/seedmesh/seedmesh
ExecStart=/home/seedmesh/.venv/bin/seedmesh serve --model JackFram/llama-160m --num-blocks 0 \
  --host-maddrs /ip4/0.0.0.0/tcp/31337 -- --announce_maddrs /ip4/YOUR_PUBLIC_IP/tcp/31337
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now seedmesh-bootstrap
sudo systemctl status seedmesh-bootstrap
journalctl -u seedmesh-bootstrap -f
```

## Check it from outside

From your own machine, not the VPS:

```bash
nc -vz YOUR_PUBLIC_IP 31337        # should connect
seedmesh chat --model JackFram/llama-160m --initial-peers <the address>
```

If `nc` fails, it is the firewall — provider-level before OS-level, in that order of
likelihood.

## What your friends do

Nothing special. Behind NAT is fine for *them*:

```bash
seedmesh serve --model JackFram/llama-160m \
  --initial-peers /ip4/YOUR_IP/tcp/31337/p2p/Qm... \
  --public-name "alice"
```

Petals detects that a peer is not directly reachable and routes it through relays
automatically — that is exactly what the bootstrap is for. See
[NAT-AND-RELAYS.md](NAT-AND-RELAYS.md).

## Cost and honesty

One box, a few dollars a month, and it is what decouples the swarm's existence from any one
person's laptop being open. It is also the single point whose loss takes the swarm down
until a new address is circulated — which is why the design treats bootstrap peers as
*entry points, not authority*: they hold no reputation state and cannot censor routing.
Running two, in different providers, removes even that.

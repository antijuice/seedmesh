# Running a bootstrap peer

A swarm needs at least one **publicly reachable** machine. This is the part people  
underestimate, so it's worth stating plainly:

> A home laptop on wifi and a Colab notebook are both behind NAT. Neither can accept an  
> incoming connection. A swarm made only of those **cannot form** — there is nowhere for  
> peers to find each other. 

The fix is one cheap always-on box with a public IP. It is the _rendezvous_, not the  
compute: a bootstrap peer relays discovery metadata and needs **no GPU**.

## What to buy

Requirements are genuinely small: 1 vCPU, 1 GB RAM, a public IPv4, and one open TCP port.  
Any of these work — check current pricing, it moves.  
Provider  
Rough cost  
Notes  
**Hetzner Cloud** (CX22)  
~€4/mo  
Best value. EU + US (Ashburn, Hillsboro) locations.  
**Oracle Cloud Always Free**  
**$0**  
4 ARM cores, 24 GB RAM, free indefinitely. Capacity in the free tier is often exhausted and signup can be fussy — worth trying first anyway.  
**DigitalOcean**  
~$6/mo  
Simplest UX if you have not done this before.  
**Vultr / Linode**  
~$5–6/mo  
Equivalent.

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
    
    `\`\\`ssh root@YOUR_IP      
      
    apt update && apt install -y python3-venv python3.12-venv python3-pip git      
    \\`    
    \`  
    `

**Swap, before anything downloads torch.** A 1 GB box OOMs during the install without it,  
and the `fstab` line is what makes it survive a reboot: 
    
    `\`\\`fallocate -l 2G /swapfile && chmod 600 /swapfile      
    mkswap /swapfile && swapon /swapfile      
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab      
    free -h        # confirm a non-zero Swap row      
    \\`    
    \`  
    `

**Firewall.** Allow SSH _before_ enabling, or you lock yourself out of the box.  
`--force` skips the "may disrupt existing ssh connections" prompt: 
    
    `\`\\`ufw allow OpenSSH      
    ufw allow 31337/tcp      
    ufw --force enable      
    ufw status     # confirm both rules are listed      
    \\`    
    \`  
    `

Now create the unprivileged account and install as that user: 
    
    `\`\\`adduser --disabled-password --gecos "" seedmesh      
    su - seedmesh      
      
    git clone <this repo> ~/seedmesh && cd ~/seedmesh      
    python3 -m venv ~/.venv && ~/.venv/bin/pip install -e .      
    ~/.venv/bin/seedmesh setup      
    \\`    
    \`  
    `

`setup` clones Petals, installs dependencies, applies the Seedmesh port, and verifies it —  
in that order, and the order matters: the codemod checks its symbol mapping against the  
_installed_ hivemind, so it cannot run before the install. Run it from the checkout, since  
that is where `tools/port_petals.py` lives.

Seeing no GPU, it installs **CPU-only torch** (a few hundred MB rather than ~2.5 GB of CUDA  
wheels a bootstrap peer would never use). Pass `--cpu-torch` to force that on a machine that  
does have a GPU. The download is still the slow part — this is what the swap above is for.

A successful run ends with `7/7 checks passed` and `ready.` — anything else means stop and  
read the output rather than continuing to the next step.

## Additional bootstrap peers (droplets 2, 3, 4)

A swarm needs **at least four** publicly reachable peers before NAT'd volunteers can hole-punch
to direct connections — see [NAT-AND-RELAYS.md](NAT-AND-RELAYS.md) for why (go-libp2p only
accepts an observed public address after four *distinct* peers report it). Below is the whole
provision as one paste, to avoid the ordering trap: everything here is root-level, and the
`seedmesh` account is deliberately passwordless, so `sudo` from it can never work.

Run as **root**, substituting the public IP and the first droplet's bootstrap address:

```bash
PUBLIC_IP=<this droplet's public IPv4>
PEER1=<the first droplet's /ip4/.../p2p/... address>

apt update && apt install -y python3-venv python3.12-venv python3-pip git

fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

ufw allow OpenSSH && ufw allow 31337/tcp && ufw --force enable

adduser --disabled-password --gecos "" seedmesh
sudo -u seedmesh bash -lc "
  git clone https://github.com/antijuice/seedmesh ~/seedmesh &&
  python3 -m venv ~/.venv &&
  ~/.venv/bin/pip install -q -e ~/seedmesh &&
  ~/.venv/bin/seedmesh setup
"

tee /etc/systemd/system/seedmesh-bootstrap.service >/dev/null <<EOF
[Unit]
Description=Seedmesh bootstrap peer
After=network-online.target

[Service]
User=seedmesh
WorkingDirectory=/home/seedmesh/seedmesh
ExecStart=/home/seedmesh/.venv/bin/seedmesh bootstrap --port 31337 --announce-ip ${PUBLIC_IP} --initial-peers ${PEER1}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable --now seedmesh-bootstrap
journalctl -u seedmesh-bootstrap -f
```

`--initial-peers ${PEER1}` matters: without it each droplet starts its *own* isolated DHT and
they never form one swarm.

### Then point volunteers at all four

This is the step that is easy to miss and makes the whole exercise pointless if skipped. The
four-observer threshold counts peers that have actually *seen* your server, so a host must
connect to all four:

```bash
seedmesh serve --model <m> --initial-peers <addr1> <addr2> <addr3> <addr4> --public-name "name"
```

Connecting to only one bootstrap gives one observer, and hole punching stays dormant exactly
as it does today.

## The provider firewall is a separate thing

`ufw` above only configures the firewall _inside_ the machine. If your provider has its own  
(Hetzner, Oracle and AWS all do, and Oracle's is **on by default and blocks everything**),  
open 31337/tcp there too, in their web console. This is the single most common reason a  
bootstrap "starts fine" and nobody can reach it.

## Start it

A bootstrap peer is a **DHT node**, not a server hosting zero blocks. It takes no `--model`  
at all — a DHT node relays discovery for whatever swarm forms on top of it, so one bootstrap  
serves any model. 
    
    `~/.venv/bin/seedmesh bootstrap --port 31337 --announce-ip YOUR_PUBLIC_IP  
    `

Substitute your real public IPv4 — `curl -4 ifconfig.me` prints it. The command refuses a  
placeholder or a private address rather than starting up unreachable.

`--announce-ip` is what the peer _tells_ others to dial. It must be the public address: most  
VPSs only see their private one locally and would otherwise advertise something unroutable.

> **Corrected 2026-08-01\.** This section previously said to run  
> `seedmesh serve --num-blocks 0` and called it a tested configuration. It is neither.  
> With no blocks, Petals builds a `ModuleAnnouncerThread` from an empty uid list and dies on  
> `module_uids[0]` with `IndexError: list index out of range` — but only _after_ about a  
> minute of throughput measurement, so it prints `Running a server on /ip4/...` and looks  
> healthy first. The test that "verified" it ran 30 seconds and stopped before the failing  
> path executed. `seedmesh serve --num-blocks 0` now refuses immediately and points here.
> 
> Verified for the replacement: `seedmesh bootstrap` ran 180s with zero tracebacks and 12  
> status reports; a block-hosting server then joined it, announced `blocks [0]`, reached  
> `Started`, and both processes were still alive 60s later with the bootstrap's routing  
> table showing **2 DHT nodes and 2 keys**. 

It prints its address: 
    
    `To connect other peers to this one, use --initial_peers /ip4/159.89.52.179/tcp/31337/p2p/QmTG981oPjsPFX5WWNegdXmkiNUgWqjUvK9spieg4hSi1h  
    `

**That whole `/ip4/.../p2p/...` string is the bootstrap address.** Send it to your friends  
verbatim; it is what they pass to `--initial-peers`.

The peer id comes from `--identity-path` (default `~/.seedmesh/bootstrap.key`), so it stays  
stable across restarts as long as you keep that file. Losing it means a new address and  
everyone re-pasting it.

## Keep it running

As **root** (the `seedmesh` account has no sudo — see Provision above): 
    
    `tee /etc/systemd/system/seedmesh-bootstrap.service >/dev/null <<'EOF'  
    [Unit]  
    Description=Seedmesh bootstrap peer  
    After=network-online.target  
      
    [Service]  
    User=seedmesh  
    WorkingDirectory=/home/seedmesh/seedmesh  
    ExecStart=/home/seedmesh/.venv/bin/seedmesh bootstrap --port 31337 --announce-ip YOUR_PUBLIC_IP  
    Restart=always  
    RestartSec=10  
      
    [Install]  
    WantedBy=multi-user.target  
    EOF  
      
    systemctl enable --now seedmesh-bootstrap  
    systemctl status seedmesh-bootstrap  
    journalctl -u seedmesh-bootstrap -f  
    `

## Check it from outside

From your own machine, not the VPS: 
    
    `nc -vz YOUR_PUBLIC_IP 31337  
    `

If `nc` fails, it is a firewall — provider-level before OS-level, in that order of  
likelihood.

Then, with someone hosting blocks: 
    
    `seedmesh chat --model JackFram/llama-160m --initial-peers <the address>  
    `

A bootstrap alone serves no model. Until at least one peer hosts blocks, a client will  
connect and find nothing to route through.

## What your friends do

Nothing special. Behind NAT is fine for _them_: 
    
    `seedmesh serve --model JackFram/llama-160m --initial-peers /ip4/YOUR_IP/tcp/31337/p2p/Qm... --public-name "alice"  
    `

Everyone must use the **identical** `--model` string — it sets the DHT prefix, so a mismatch  
puts people in separate swarms with no error message.

Petals detects that a peer is not directly reachable and routes it through relays  
automatically — that is exactly what the bootstrap is for. See  
[NAT-AND-RELAYS.md](NAT-AND-RELAYS.md).

## Cost and honesty

One box, a few dollars a month, and it is what decouples the swarm's existence from any one  
person's laptop being open. It is also the single point whose loss takes the swarm down  
until a new address is circulated — which is why the design treats bootstrap peers as  
_entry points, not authority_: they hold no reputation state and cannot censor routing.  
Running two, in different providers, removes even that. 
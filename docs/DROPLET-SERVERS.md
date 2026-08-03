# Running block servers on the bootstrap droplets

Two directly-reachable servers is what unblocks verification. Every peer so far is either a
bootstrap peer (hosts no blocks) or behind a NAT, and a relay-only peer cannot be placed on a
network, so the sampler correctly refuses to pair anyone. Droplets have public IPs and no NAT,
so they are the shortest path to a working verification pair.

## Pick two droplets, and host the SAME blocks on both

Not a split. Verification replays one server's work on another, and
`eligible_verifiers` requires the verifier to cover the subject's block range — so two
servers hosting blocks 0–6 and 6–12 give you coverage and **zero** verification pairs. Both
must host the full range.

Your four droplets sit in four different /16s, so any two of them count as independent
clusters. Worth being honest about what that means: they are all your DigitalOcean account,
and an ASN-aware resolver would put them in one cluster and refuse the pair. This exercises
the machinery correctly; it is not genuine third-party independence. Your friend's machine is.

## Memory is the constraint

A 1 GB droplet already running a bootstrap peer has roughly 700–800 MB free, and a CPU server
for llama-160m needs most of that:

| | |
| --- | --- |
| torch + petals runtime | ~400 MB |
| 12 blocks at fp32 | ~220 MB |
| attention cache, default | ~145 MB |
| model on disk | 0.61 GiB |

That is too close to the line. Two settings bring it down, and the cache one matters most:

```bash
--quant none --device cpu --attn-cache-tokens 2048
```

`--quant none` because bitsandbytes' NF4 path wants CUDA; on a CPU droplet it buys nothing.
`--attn-cache-tokens 2048` cuts the cache to about a third — it bounds concurrent session
length, not context or quality, and a two-person swarm has no concurrency to speak of.

**Add swap before starting.** An OOM here does not just kill the server, it kills the
bootstrap peer sharing the box — and dropping below four bootstrap peers is what caused the
whole relay saga. As root:

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
free -h
```

## Set it up

On **two** of the four droplets. As root first:

```bash
su - seedmesh -c 'cd ~/seedmesh && git pull -q && ~/.venv/bin/pip install -qe .'
```

**Open a port for the block server first.** This is the whole reason for hosting on a
droplet, and leaving it out silently defeats the exercise: without a fixed, reachable port
Petals' startup reachability check fails, the server falls back to relays, and it then
advertises a *circuit* address whose IP belongs to the relay. A client cannot place such a
peer on a network, so it is refused as a verification partner — which is exactly the thing
these servers exist to provide.

```bash
ufw allow 31338/tcp
```

Then a service, so it survives a reboot and logs somewhere findable. Note the three flags
that make it directly dialable — a pinned port, and its own public address to announce:

```bash
tee /etc/systemd/system/seedmesh-server.service >/dev/null <<'EOF'
[Unit]
Description=Seedmesh block server
After=network-online.target seedmesh-bootstrap.service

[Service]
User=seedmesh
WorkingDirectory=/home/seedmesh/seedmesh
ExecStart=/home/seedmesh/.venv/bin/seedmesh serve --num-blocks 12 --quant none --device cpu --attn-cache-tokens 2048 --host-maddrs /ip4/0.0.0.0/tcp/31338 --public-ip PUBLIC_IP_HERE --public-name droplet-1
Restart=on-failure
RestartSec=30
# Keep the server from taking the bootstrap peer down with it.
MemoryMax=650M

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload && systemctl enable --now seedmesh-server
journalctl -u seedmesh-server -f
```

Change `--public-name` per droplet so the monitor can tell them apart. The first start
downloads the model and takes a few minutes.

`MemoryMax=650M` is deliberate: if the server exceeds it, systemd kills **only the server**
and restarts it, rather than letting the kernel's OOM killer pick a victim — which might be
the bootstrap peer.

## Check it worked

```bash
seedmesh monitor
```

You want both droplets listed with no `+relay` marker, and coverage showing at least 2 hosts
per block. Then, from anywhere:

```bash
seedmesh chat
```

The summary at exit should stop saying `skipped: no independent verifier` and start reporting
verified requests. That is the first time the trust layer will have run end to end on a real
swarm.

## If it does not work

**Server killed within a minute** — memory. Check `journalctl -u seedmesh-server | grep -i
kill`. Drop to `--num-blocks 6` on both (still overlapping, still verifiable) or add more swap.

**Bootstrap peer died too** — `MemoryMax` was missing or too high. `systemctl status
seedmesh-bootstrap` and restart it; the swarm needs all four.

**Monitor shows them but `chat` still skips verification** — check the two servers' block
ranges actually overlap. `seedmesh monitor` prints the range per server; a verifier must
cover the subject's whole span.

## Fixing droplet servers that came up relay-only

If `seedmesh chat` still reports `skipped: no independent verifier`, and `seedmesh monitor`
shows `+relay` next to the droplet servers, they were started without the port and announce
address above. Run this on each, substituting that droplet's own public IP:

**Look before patching.** A `sed` that matches part of the existing line does nothing when
that part differs, and silently — the service restarts, reports `active`, and is unchanged.
Print the current state first:

```bash
ssh root@DROPLET_IP 'grep ^ExecStart /etc/systemd/system/seedmesh-server.service; ufw status | head -5; ss -ltnp | grep -c 31338'
```

Then replace the **whole** `ExecStart` line rather than a fragment of it, substituting the
droplet's own IP in both places:

**Let the droplet supply its own IP.** An earlier version of this command had the address in
two places for you to substitute, and one droplet ended up announcing the *other* one's
address — which fails in a genuinely confusing way: the port is open, TCP connects, and
libp2p still refuses with `all dials failed ... dial backoff`, because the peer id answering
there is not the one the record claims. `curl` removes the possibility:

```bash
ssh root@DROPLET_IP 'IP=$(curl -4 -s ifconfig.me); ufw allow 31338/tcp; sed -i "s|^ExecStart=.*|ExecStart=/home/seedmesh/.venv/bin/seedmesh serve --num-blocks 12 --quant none --device cpu --attn-cache-tokens 2048 --host-maddrs /ip4/0.0.0.0/tcp/31338 --public-ip ${IP} --public-name $(hostname)|" /etc/systemd/system/seedmesh-server.service && systemctl daemon-reload && systemctl restart seedmesh-server && sleep 30 && grep ^ExecStart /etc/systemd/system/seedmesh-server.service'
```

It prints the resulting `ExecStart`, so you can see the address it chose rather than trusting
that a substitution landed.

**Restart order matters.** Petals decides relay-vs-direct with a reachability check at
*startup*. Opening the firewall after the server is already running changes nothing until it
restarts, so `ufw allow` and `systemctl restart` belong in that order, in one command.

If `ufw status` says `inactive`, the host firewall is not what is blocking you — check the
provider's own firewall (DigitalOcean cloud firewalls are configured in the control panel,
not on the box, and `ufw allow` has no effect on them).

Two servers cannot share port 31338 on one host, so if you ever run a second block server on
the same droplet give it a different port and open that one too.

Confirm it worked — the `+relay` marker should be gone:

```bash
seedmesh monitor
```

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

Then a service, so it survives a reboot and logs somewhere findable:

```bash
tee /etc/systemd/system/seedmesh-server.service >/dev/null <<'EOF'
[Unit]
Description=Seedmesh block server
After=network-online.target seedmesh-bootstrap.service

[Service]
User=seedmesh
WorkingDirectory=/home/seedmesh/seedmesh
ExecStart=/home/seedmesh/.venv/bin/seedmesh serve --num-blocks 12 --quant none --device cpu --attn-cache-tokens 2048 --public-name droplet-1
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

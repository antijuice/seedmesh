# Running the public monitor on a bootstrap droplet

A swarm needs somewhere always-on and publicly reachable to publish its status page. Your
bootstrap droplets already are both, so they are the natural host — and `seedmesh monitor`
is deliberately built so this costs almost nothing.

## What this does and does not need

**Does not need `seedmesh setup`.** Monitoring reads DHT keys. It needs `hivemind` and
nothing else — no Petals, no transformers, no model weights. The DHT prefix and block count
come from the model's `config.json` over plain HTTP, and the per-block announcements are
decoded directly. A droplet provisioned only as a bootstrap peer can serve the dashboard.

Measured peak RSS for one refresh against the live swarm:

| | |
| --- | --- |
| with Petals importable | 390 MiB |
| without | **279 MiB** |

Smaller a gap than you would expect, because hivemind itself depends on torch — torch is
resident either way. The saving is real but the stronger argument is that the second column
needs no backend install at all.

**Verified equivalent, not assumed.** The no-Petals path was A/B'd against the Petals path on
the live swarm with the `petals` import blocked: identical reports, field by field. This
matters because the failure mode is silent — a wrong DHT prefix reads an *empty* namespace
and renders a healthy swarm as dead rather than raising.

## Cost on a $5 droplet

The refresh is a short-lived process, not a resident service: it starts a client-mode DHT,
reads, writes a file, exits. ~280 MiB for a few seconds every five minutes, alongside a
bootstrap peer that is already running. On a 1 GiB droplet that is comfortable. Do not run it
every 30 seconds — each run pays a fresh DHT bootstrap, and the data does not change that
fast.

## Setup

All of this runs as root on **one** droplet. Pick one; you do not want four identical pages.

```bash
apt-get update -qq && apt-get install -y -qq nginx
su - seedmesh -c 'cd ~/seedmesh && git pull -q && ~/.venv/bin/pip install -qe .'
```

A systemd timer rather than cron, so failures land in the journal where you can find them:

```bash
tee /etc/systemd/system/seedmesh-monitor.service >/dev/null <<'EOF'
[Unit]
Description=Render the Seedmesh swarm status page
After=network-online.target

[Service]
Type=oneshot
User=seedmesh
WorkingDirectory=/home/seedmesh/seedmesh
ExecStart=/home/seedmesh/.venv/bin/seedmesh monitor --html /var/www/seedmesh/index.html
TimeoutStartSec=300
EOF

tee /etc/systemd/system/seedmesh-monitor.timer >/dev/null <<'EOF'
[Unit]
Description=Refresh the Seedmesh status page every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

mkdir -p /var/www/seedmesh && chown seedmesh:seedmesh /var/www/seedmesh
systemctl daemon-reload && systemctl enable --now seedmesh-monitor.timer
systemctl start seedmesh-monitor && systemctl status seedmesh-monitor --no-pager
```

Serve it. Static files only — there is no application here, and nothing to POST to:

```bash
tee /etc/nginx/sites-available/seedmesh >/dev/null <<'EOF'
server {
    listen 80;
    server_name _;
    root /var/www/seedmesh;
    index index.html;
    location / { try_files $uri $uri/ =404; }
}
EOF
ln -sf /etc/nginx/sites-available/seedmesh /etc/nginx/sites-enabled/seedmesh
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

Then open `http://<that droplet's IP>/`.

## Checks

```bash
systemctl list-timers seedmesh-monitor --no-pager
journalctl -u seedmesh-monitor -n 30 --no-pager
```

The page is self-contained — no external CSS, fonts, or scripts — so it renders on a bare
static host and works offline. It also means nothing on it phones home about whoever views it.

## A note on what the page publishes

Peer ids, self-reported names and throughputs, compute profiles, and observed reliability.
Peer ids and volunteer names are already public to anyone who joins the swarm, so this
exposes nothing new — but it does make it *convenient*, which is a different thing. If a
volunteer would rather not appear by name, they can omit `--public-name` when serving and
show as a truncated peer id instead.

## Exit codes

`seedmesh monitor` exits 0 when every block has a host and 1 when the model is not usable, so
it works as a health check without parsing anything:

```bash
seedmesh monitor >/dev/null || echo "swarm has a coverage gap"
```

The `--html` and `--watch` forms keep that behaviour.

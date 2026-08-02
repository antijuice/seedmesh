# Making a Windows + WSL2 machine directly reachable

Being directly dialable is what unlocks three things at once: verification (a relayed peer
cannot be shown to be an independent operator, so it is refused as a verification partner),
the routing gate (which needs verification verdicts to have anything to act on), and reliable
large-model hosting (a relayed connection is severed at 128 KiB).

Values below are from a real machine — swap in your own. Find yours with:

```powershell
(Find-NetRoute -RemoteIPAddress 1.1.1.1 | Select-Object -First 1).IPAddress
Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name,MacAddress
```

## Step 0 — check whether you need the WSL step at all

```powershell
Get-Content "$env:USERPROFILE\.wslconfig"
```

If it contains `networkingMode=mirrored`, WSL shares the Windows host's network interfaces
directly. **There is no second NAT and no `netsh interface portproxy` needed** — which is the
advice you will find in most guides, and it is wrong for this configuration. Confirm from
inside WSL: `ip -4 -o addr show` should list your real LAN addresses, not a `172.x` private
one on `eth0`.

If it is *not* mirrored, add it (Windows 11 with WSL ≥ 2.0):

```powershell
Add-Content "$env:USERPROFILE\.wslconfig" "`n[wsl2]`nnetworkingMode=mirrored"
wsl --shutdown
```

## Step 1 — the Hyper-V firewall (the one that silently drops everything)

Mirrored mode routes inbound traffic through a **separate firewall from the normal Windows
one**, and it defaults to blocking. A correct router forward plus a correct Windows Firewall
rule will still fail here, with nothing logged and nothing to see.

Check it:

```powershell
Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore | Select-Object Name,DefaultInboundAction
```

`DefaultInboundAction : Block` means you need a rule. **Add a rule for the one port — do not
set `DefaultInboundAction Allow`**, which would expose every service running in WSL to your
whole local network. Run as Administrator:

```powershell
New-NetFirewallHyperVRule -Name "SeedMesh-31338" -DisplayName "Seedmesh block hosting" -Direction Inbound -VMCreatorId "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" -Protocol TCP -LocalPorts 31338 -Action Allow
```

That `VMCreatorId` is WSL's, and is the same on every machine.

## Step 2 — the Windows Firewall

Separate from the above, and both must allow the traffic. Check first — you may already have
a rule:

```powershell
Get-NetFirewallRule -Enabled True -Direction Inbound | ForEach-Object { $p = $_ | Get-NetFirewallPortFilter; if ($p.LocalPort -eq 31338) { $_ | Select-Object DisplayName,Action,Profile } }
```

If nothing comes back, add one as Administrator. `-Profile Any` matters: home Wi-Fi is often
categorised **Public**, and a Private-only rule then does nothing.

```powershell
New-NetFirewallRule -DisplayName "Seedmesh block hosting" -Direction Inbound -Protocol TCP -LocalPort 31338 -Action Allow -Profile Any
```

## Step 3 — pin the machine's LAN address

Port forwarding sends traffic to a fixed IP, so the IP must stop moving. In your router's
DHCP settings, reserve the address against the **MAC of the adapter actually carrying
traffic** — not any adapter, the one `Find-NetRoute` named in Step 0.

Worth checking if you have more than one Wi-Fi adapter: a USB dongle and a built-in card both
get addresses, and unplugging the dongle silently moves you to the other one and breaks the
forward.

## Step 4 — forward the port on the router

Browse to your gateway (`10.0.0.1` for most home routers; confirm with
`Get-NetRoute -DestinationPrefix '0.0.0.0/0'`). Find **Port Forwarding** / **Virtual Server**
/ **NAT** and add:

| field | value |
| --- | --- |
| external / WAN port | `31338` |
| internal / LAN port | `31338` |
| protocol | TCP |
| internal IP | the reserved address from Step 3 |

## Step 5 — serve on the forwarded port and check

```bash
seedmesh serve --host-maddrs /ip4/0.0.0.0/tcp/31338
```

```bash
seedmesh doctor --port 31338
```

`doctor` listens exactly as a server does and reports whether other peers can observe a
public address for you. `=== reachable ===` means done.

## If it does not work

**`doctor` says symmetric-nat.** Your router assigns a different external port per
destination, so no two peers agree on your address. A forward is the fix, so if you have done
one and still see this, check the forward is actually active — some routers silently disable
rules when UPnP is on, or when the internal IP no longer matches a connected device.

**Nothing reaches you at all.** Check for a VPN. A connected VPN takes over the default route,
and inbound connections to your home router then never reach the machine. Check with:

```powershell
Get-NetRoute -DestinationPrefix '0.0.0.0/0' | Select-Object InterfaceAlias,NextHop
```

If a tunnel adapter is listed there, disconnect it while hosting.

**You are behind CGNAT.** Some ISPs share one public address across many customers, and no
forward can work. You can rule this in or out: compare your router's WAN address to the
public address `seedmesh doctor` reports. If they differ, you are behind carrier NAT, and
hosting needs a different network or an ISP static-IP option.

## What changes once you are reachable

- `seedmesh doctor` reports `reachable`
- `seedmesh monitor` stops showing `+relay` next to your server
- Inline verification can pair you with another server, so `chat` stops reporting
  `skipped: no independent verifier`
- The routing gate gains verdicts to act on
- Large models stop hitting the 128 KiB relay budget

#!/usr/bin/env bash
# Churn test: kill a serving node mid-swarm and confirm the client reroutes.
#
# Seedmesh's whole design assumes churn is normal and survivable -- routing has failover,
# reputation forgives it, the DHT expires dead peers. None of that had ever been observed.
#
# Three servers, with the tail half REDUNDANT:
#   server 1: blocks 0-5
#   server 2: blocks 6-11
#   server 3: blocks 6-11   <- the spare
# Killing the sole host of a range proves nothing except that inference stops, so the
# redundancy is the point: there must be somewhere to reroute TO.
#
#   bash tools/churn_test.sh

set -u
PY=${PY:-$HOME/hmspike/bin/python3}
MODEL=${MODEL:-JackFram/llama-160m}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

cleanup() {
  echo
  echo "--- shutting down ---"
  pkill -f run_server 2>/dev/null
  sleep 2
}
trap cleanup EXIT

pkill -f run_server 2>/dev/null
sleep 2
rm -f /tmp/srv*.log /tmp/srv*.id /tmp/srv1.addr /tmp/srv*.pid
cd /tmp

start() {   # start <n> <host> <port> <blocks> [initial_peers]
  local n=$1 host=$2 port=$3 blocks=$4 peers=${5:-}
  local extra="--new_swarm"
  [ -n "$peers" ] && extra="--initial_peers $peers"
  setsid nohup "$PY" -m petals.cli.run_server "$MODEL" \
    $extra --block_indices "$blocks" --device cpu --torch_dtype float32 --throughput 1 \
    --host_maddrs "/ip4/$host/tcp/$port" --identity_path "/tmp/srv$n.id" \
    > "/tmp/srv$n.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    grep -q "\[INFO\] Started" "/tmp/srv$n.log" 2>/dev/null && break
    grep -qE "Traceback" "/tmp/srv$n.log" 2>/dev/null && { echo "server $n FAILED"; tail -12 "/tmp/srv$n.log"; exit 1; }
    sleep 3
  done
  # Record pid and peer id so the test can map "the peer serving these blocks" to a process.
  pgrep -f "identity_path /tmp/srv$n.id" | head -1 > "/tmp/srv$n.pid"
  grep -o "/ip4/$host/tcp/$port/p2p/[A-Za-z0-9]*" "/tmp/srv$n.log" | tail -1 \
    | sed 's|.*/p2p/||' > "/tmp/srv$n.peer"
  echo "server $n up: $host:$port blocks $blocks, pid $(cat /tmp/srv$n.pid), peer ...$(tail -c 9 /tmp/srv$n.peer)"
}

# Each server binds a DIFFERENT loopback address. The whole 127.0.0.0/8 range is local on
# Linux, and distinct addresses are what let peer-address resolution be *observed* -- with
# every server on 127.0.0.1 the resolver works but every peer looks identical.
echo "=== bringing up 3 servers (tail half redundant, distinct addresses) ==="
start 1 127.0.0.1 31337 0:6
ADDR=$(grep -o "/ip4/127.0.0.1/tcp/31337/p2p/[A-Za-z0-9]*" /tmp/srv1.log | tail -1)
echo "$ADDR" > /tmp/srv1.addr
start 2 127.0.0.2 31338 6:12 "$ADDR"
start 3 127.0.0.3 31339 6:12 "$ADDR"

echo
# CLIENT is overridable so the same three-server swarm can host other clients --
# tools/backend_demo.py in particular, which needs the redundant tail to form a
# verification pair.
CLIENT=${CLIENT:-$SCRIPT_DIR/churn_test.py}
echo "=== client: $(basename "$CLIENT") ==="
"$PY" "$CLIENT"

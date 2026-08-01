#!/usr/bin/env bash
# Bring up a private two-server Petals swarm and run one inference through it.
#
# Servers host complementary halves of a 12-layer model, so a successful generation
# requires the client to build a pipeline that spans BOTH hosts -- which is the point.
# Everything runs on localhost; --new_swarm means it never touches the public network.
#
# Runs everything in one shot deliberately: block announcements carry a TTL, so a client
# started minutes later can find the swarm already expired.
#
# Requires: a ported Petals checkout (tools/port_petals.py) and hivemind, so Linux/WSL.
#   bash tools/swarm_demo.sh

set -u

PY=${PY:-$HOME/hmspike/bin/python3}
MODEL=${MODEL:-JackFram/llama-160m}
NLAYERS=${NLAYERS:-12}
HALF=$((NLAYERS / 2))
# Resolved before any `cd` -- the servers are started from /tmp.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CLIENT=${CLIENT:-$SCRIPT_DIR/swarm_client.py}

cleanup() {
  echo
  echo "--- shutting down servers ---"
  pkill -f run_server 2>/dev/null
  sleep 2
}
trap cleanup EXIT

pkill -f run_server 2>/dev/null
sleep 2
rm -f /tmp/srv1.log /tmp/srv2.log /tmp/srv1.id /tmp/srv2.id /tmp/srv1.addr

wait_for_started() {
  local log=$1 name=$2
  for _ in $(seq 1 60); do
    if grep -q "\[INFO\] Started" "$log" 2>/dev/null; then
      echo "$name up"
      return 0
    fi
    if grep -qE "Traceback|Error" "$log" 2>/dev/null; then
      echo "$name FAILED:"
      tail -15 "$log"
      return 1
    fi
    sleep 3
  done
  echo "$name timed out"
  tail -15 "$log"
  return 1
}

echo "=== server 1: blocks 0-$((HALF - 1)) ==="
cd /tmp
setsid nohup "$PY" -m petals.cli.run_server "$MODEL" \
  --new_swarm --block_indices "0:$HALF" --device cpu --torch_dtype float32 \
  --throughput 1 --host_maddrs /ip4/127.0.0.1/tcp/31337 --identity_path /tmp/srv1.id \
  > /tmp/srv1.log 2>&1 < /dev/null &
wait_for_started /tmp/srv1.log "server 1" || exit 1

ADDR=$(grep -o "/ip4/127.0.0.1/tcp/31337/p2p/[A-Za-z0-9]*" /tmp/srv1.log | tail -1)
echo "$ADDR" > /tmp/srv1.addr
echo "bootstrap: $ADDR"

echo
echo "=== server 2: blocks $HALF-$((NLAYERS - 1)) ==="
setsid nohup "$PY" -m petals.cli.run_server "$MODEL" \
  --initial_peers "$ADDR" --block_indices "$HALF:$NLAYERS" --device cpu --torch_dtype float32 \
  --throughput 1 --host_maddrs /ip4/127.0.0.1/tcp/31338 --identity_path /tmp/srv2.id \
  > /tmp/srv2.log 2>&1 < /dev/null &
wait_for_started /tmp/srv2.log "server 2" || exit 1

echo
echo "=== client ==="
"$PY" "$CLIENT"

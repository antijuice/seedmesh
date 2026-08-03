"""`seedmesh gateway` -- a local web UI over the swarm.

Everything Seedmesh does is currently reachable only by running a CLI inside WSL. That is a
real barrier for the thing this project is for: someone donating a GPU, or using the swarm,
should not have to live in a terminal. This is the front door.

**It shows the trust layer, not just the chat.** A ChatGPT-shaped box that hides where the
answer came from would throw away the entire point. Every reply carries which servers
computed it, and the sidebar shows coverage, what this client has measured first-hand, and
what verification has and has not been able to check. "Not yet verified" is displayed as
prominently as a verdict, because a swarm that cannot verify must never look like one that
verified and found nothing wrong.

**Deliberately local-only, with a token.** It binds 127.0.0.1 and prints a URL containing a
random token that the page sends back on every request. Two reasons, and neither is
paranoia: there is no authentication, so binding publicly would hand anyone an open
inference endpoint *and* the ability to publish reputation signed with this node's identity.
And any web page open in the browser can POST to a localhost port, so without a token an
unrelated site could drive this one. `--host` exists for people who know what they are doing
and says so loudly.

**One request at a time.** Petals inference sessions are stateful and not thread-safe, and
the reputation observer wraps a shared class -- so generation is serialised behind a lock.
For a personal gateway that is correct rather than limiting; a multi-user deployment would
need a different design, and pretending otherwise would produce corrupted sessions under the
first concurrent click.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Optional

# Long enough that guessing is hopeless, short enough to paste.
TOKEN_BYTES = 24


class GatewayState:
    """Everything the HTTP layer needs, with the backend injected so it can be tested.

    `generate` and `status` are supplied by the caller rather than built here: the real ones
    need Petals and a live swarm, and the routing, locking, error handling and payload
    shaping are worth testing without either.
    """

    def __init__(
        self,
        generate: Callable[[str, int], str],
        *,
        status: Optional[Callable[[], Dict[str, Any]]] = None,
        model: str = "",
        max_new_tokens: int = 64,
    ) -> None:
        self._generate = generate
        self._status = status
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []
        self.busy = False

    def chat(self, prompt: str) -> Dict[str, Any]:
        """Run one prompt. Returns a payload the page can render, error or not."""
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "empty prompt"}

        # Refused rather than queued: a queued request behind a 30s generation looks like a
        # hung UI, and the user has no way to tell which of their clicks is running.
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "error": "busy", "detail": "a request is already running"}

        began = time.perf_counter()
        try:
            self.busy = True
            text = self._generate(prompt, self.max_new_tokens)
            entry = {
                "ok": True,
                "prompt": prompt,
                "reply": text,
                "elapsed_ms": (time.perf_counter() - began) * 1000.0,
            }
        except Exception as exc:
            entry = {
                "ok": False,
                "prompt": prompt,
                "error": f"{type(exc).__name__}",
                "detail": str(exc)[:400],
                "elapsed_ms": (time.perf_counter() - began) * 1000.0,
            }
        finally:
            self.busy = False
            self._lock.release()

        self.history.append(entry)
        return entry

    def status(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "busy": self.busy,
                                   "requests": len(self.history)}
        if self._status is None:
            return payload
        try:
            payload.update(self._status() or {})
        except Exception as exc:
            # A broken status panel must not take the chat down with it.
            payload["status_error"] = f"{type(exc).__name__}: {exc}"
        return payload


def route(state: GatewayState, method: str, path: str, body: Optional[dict]) -> tuple:
    """(status code, payload) for an API call. Pure, so the routing is testable."""
    if method == "GET" and path == "/api/status":
        return 200, state.status()
    if method == "POST" and path == "/api/chat":
        if not isinstance(body, dict):
            return 400, {"ok": False, "error": "expected a JSON object"}
        result = state.chat(body.get("prompt", ""))
        return (200 if result.get("ok") else 409 if result.get("error") == "busy" else 200), result
    return 404, {"ok": False, "error": "not found"}


def build_status(reputation, monitor_report: Callable[[], Any]) -> Callable[[], Dict[str, Any]]:
    """Status callable combining swarm coverage with what this client has measured."""

    def status() -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        try:
            report = monitor_report()
            payload["swarm"] = {
                "covered": report.covered,
                "blocks": report.n_blocks,
                "min_replicas": report.min_replicas,
                "missing": report.missing_blocks[:12],
                "servers": [
                    {
                        "label": row.label,
                        "peer": row.short_id,
                        "blocks": f"{row.start}-{row.end}",
                        "profile": row.profile,
                        "throughput": row.throughput,
                        "reliability": row.reliability,
                        "observers": row.observers,
                    }
                    for row in report.servers
                ],
            }
        except Exception as exc:
            payload["swarm_error"] = f"{type(exc).__name__}: {exc}"

        if reputation is not None:
            observer = reputation.observer
            payload["measured"] = {
                "requests": observer.stats.requests,
                "ok": observer.stats.ok,
                "failed": observer.stats.timeout + observer.stats.error,
                "servers": len(observer.subjects),
            }
            verifier = getattr(reputation, "verifier", None)
            if verifier is not None:
                tally: Dict[str, int] = {}
                for outcome in verifier.outcomes:
                    tally[outcome.verdict.value] = tally.get(outcome.verdict.value, 0) + 1
                payload["verification"] = {
                    "checked": verifier.checked,
                    "skipped": verifier.skipped_no_verifier,
                    # Surfaced separately because the remedy differs: "nobody else hosts
                    # these blocks" needs a volunteer, "nobody has a direct address" needs
                    # a reachable one.
                    "skipped_unlocated": verifier.skipped_unlocated,
                    "verdicts": tally,
                }
            gate = getattr(reputation, "gate", None)
            if gate is not None:
                payload["excluded"] = sorted(gate.ever_blocked)
        return payload

    return status


def make_handler(state: GatewayState, token: str, page: str):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: Any, content_type: str = "application/json"):
            body = (
                payload.encode("utf-8")
                if isinstance(payload, str)
                else json.dumps(payload).encode("utf-8")
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # No external anything, and no framing by another page.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; frame-ancestors 'none'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            supplied = self.headers.get("X-Seedmesh-Token")
            if supplied is None:
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                for part in query.split("&"):
                    if part.startswith("token="):
                        supplied = part[6:]
            # compare_digest: a plain == leaks the token a character at a time to anything
            # that can time the response, and this token authorises publishing under this
            # node's signing identity.
            return supplied is not None and secrets.compare_digest(supplied, token)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if not self._authorised():
                return self._send(403, {"error": "bad or missing token"})
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            code, payload = route(state, "GET", path, None)
            return self._send(code, payload)

        def do_POST(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if not self._authorised():
                return self._send(403, {"error": "bad or missing token"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
            except Exception:
                return self._send(400, {"ok": False, "error": "malformed JSON"})
            code, payload = route(state, "POST", path, body)
            return self._send(code, payload)

        def log_message(self, *args):  # keep the console for Seedmesh's own output
            return

    return Handler


def serve(state: GatewayState, *, host: str, port: int, page: str) -> None:
    from http.server import ThreadingHTTPServer

    token = secrets.token_urlsafe(TOKEN_BYTES)
    server = ThreadingHTTPServer((host, port), make_handler(state, token, page))
    actual = server.server_address[1]

    print(f"\n  seedmesh gateway  ->  http://{host}:{actual}/?token={token}\n")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: bound to a non-local address. There is no authentication beyond")
        print("  the token above, and anyone who reaches this port can spend your swarm")
        print("  capacity and publish reputation signed with your identity.\n")
    print("  Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def cmd_gateway(args) -> int:
    """Start the local UI over a live swarm.

    Reuses `chat`'s stack wholesale -- same warm-up wait, same retry-with-a-fresh-session,
    same reputation and verification -- so the two front ends cannot drift into disagreeing
    about how the swarm behaves.
    """
    import torch

    from seedmesh.cli.main import (
        CHAT_WARMUP_TIMEOUT,
        _backend_missing,
        _refuse_on_windows,
        _start_reputation,
        explain_inference_failure,
        generate_with_retry,
        wait_until_routable,
    )
    from seedmesh.cli.gateway_page import PAGE
    from seedmesh.cli.swarm import resolve_peers

    if _refuse_on_windows("The gateway") or _backend_missing("The gateway"):
        return 2

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note:
        print(note)
    if not peers:
        print("no bootstrap peers; pass --initial-peers or --swarm-file")
        return 2

    from hivemind.dht import DHT
    from transformers import AutoTokenizer

    from petals import AutoDistributedModelForCausalLM

    # DHT before the model, as everywhere else -- hivemind allocates a shared-memory buffer
    # on first construction and it fails inside transformers' init context.
    dht = DHT(initial_peers=peers, client_mode=True, start=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=peers, torch_dtype=torch.float32,
        request_timeout=args.timeout, max_retries=1,
    )

    print("waiting for the swarm to be routable...")
    probe_ids = tokenizer("hello", return_tensors="pt")["input_ids"]

    def probe():
        with torch.inference_mode():
            model.generate(probe_ids, max_new_tokens=1, do_sample=False)

    try:
        if not wait_until_routable(probe, timeout=CHAT_WARMUP_TIMEOUT):
            print(explain_inference_failure(RuntimeError("routing: not found")))
            dht.shutdown()
            return 1
    except Exception as exc:
        print(explain_inference_failure(exc))
        dht.shutdown()
        return 1

    reputation = None
    if not args.no_reputation:
        try:
            args._model_object = model
            args.no_verify = getattr(args, "no_verify", False)
            args.no_routing_gate = getattr(args, "no_routing_gate", False)
            args.gossip_interval = getattr(args, "gossip_interval", 300.0)
            reputation = _start_reputation(args, dht, peers)
        except Exception as exc:
            print(f"  (reputation disabled: {type(exc).__name__}: {exc})")

    def generate(prompt: str, max_new_tokens: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

        def run():
            with torch.inference_mode():
                return model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)

        outputs = generate_with_retry(run, args.attempts)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if reputation is not None:
            # Between requests, exactly as in chat: verification and gossip must not
            # compete with the request they are measuring.
            reputation.maybe_sync()
        return text

    status = None
    if reputation is not None or True:
        from seedmesh.cli.monitor import collect

        prefix = "seedmesh"
        try:
            from seedmesh.cli.swarm import load_swarm

            swarm = load_swarm(args.swarm_file)
            if swarm is not None and swarm.name:
                prefix = f"seedmesh/{swarm.name}"
        except Exception:
            pass
        status = build_status(reputation, lambda: collect(dht, args.model, prefix=prefix))

    state = GatewayState(
        generate, status=status, model=args.model, max_new_tokens=args.max_new_tokens
    )
    try:
        serve(state, host=args.host, port=args.port, page=PAGE)
    finally:
        if reputation is not None:
            reputation.close()
            for line in reputation.describe():
                print(line)
        dht.shutdown()
    return 0

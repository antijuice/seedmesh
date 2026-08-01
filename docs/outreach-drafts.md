# Outreach drafts

Drafts for the two emails that gate the MIT-dependent parts of the plan, plus the reasoning
behind how each is framed. **Read the framing notes before sending** — in both cases the ask
changed once the project's actual shape became clear, and sending the original version would
ask for more than is needed and invite a harder review.

Nothing here has been sent. These are drafts for you to edit and send yourself.

---

## 1. MIT ORCD — cluster use

**To:** `orcd-help-engaging@mit.edu`
**Gates:** any use of Engaging GPUs for Seedmesh work.

### Why the ask is now narrower than the original spec assumed

The original plan wanted cluster GPUs to host blocks for a swarm that outside volunteers
would connect to. That is the version most likely to run into trouble: a permanent,
internet-facing service serving anonymous members of the public is a materially different
thing from a research computing job, even with identical code, and university acceptable-use
policies are generally scoped to research and instruction.

That is no longer what the cluster is needed for. The trust layer is built and needs no GPU
at all. What actually needs datacenter hardware is bounded and unambiguously research-shaped:

1. **Numerical calibration.** Measuring how far apart two *honest* servers' outputs land on
   genuinely different GPUs (A100 vs H100 vs consumer cards, bf16 vs fp16 vs int8). Every
   verification threshold in the system is currently fitted to *simulated* floating-point
   noise. This needs real heterogeneous hardware and cannot be faked.
2. **Multi-node private swarm testing.** A handful of Slurm jobs talking to each other over
   the cluster's internal network, to confirm block sharding and failover behave under real
   churn. Cluster-internal only.

Both are short-lived batch jobs, both are internal, and neither serves outside traffic. Ask
for that. If you later want the cluster to carry public traffic, that is a separate
conversation and should be asked separately.

### Draft

> Subject: Request for guidance — GPU use for an open-source distributed systems research project
>
> Hello,
>
> I'm an MIT student working on an open-source research project in distributed systems, and
> I'd like to check in advance whether my intended use of Engaging is appropriate before I
> start, rather than after.
>
> The project is a trust and verification layer for distributed model inference — the
> problem of detecting whether a remote node in a peer-to-peer inference network actually
> performed the computation it claims to have performed. The software is MIT-licensed and
> developed in the open.
>
> There are two things I would like to use cluster GPUs for:
>
> 1. **Numerical characterization.** I need to measure how much two *honest* GPUs disagree
>    when computing the same transformer layers, across different architectures and
>    precisions (e.g. A100 bf16 vs L40S fp16). This calibrates the statistical thresholds
>    that distinguish ordinary floating-point nondeterminism from a node returning incorrect
>    results. These are short batch jobs that run a few forward passes and record output
>    distances.
>
> 2. **Multi-node testing.** A small number of concurrent Slurm jobs communicating over the
>    cluster's internal network, to test how the system handles nodes joining and leaving
>    mid-computation. I would deliberately cancel jobs mid-request to confirm the failure
>    handling works.
>
> To be explicit about what I am **not** asking for: these jobs would be
> cluster-internal only, would not accept connections from outside MIT, would not serve
> traffic for anyone outside the project, and would not run persistently. I understand a
> permanently-running internet-facing service would be a different request, and I am not
> making it.
>
> Is this an appropriate use of Engaging resources? If there are constraints I should work
> within — particular partitions, walltime limits, or restrictions on inter-node networking
> — I would rather design around them now.
>
> Thank you,
> [name, MIT affiliation, course/lab if relevant]

### Notes

- The explicit "what I am not asking for" paragraph is doing real work. It shows you
  understand the distinction the AUP cares about, which makes it much easier to say yes.
- If they say no to (2) but yes to (1), that is still a good outcome — calibration data is
  the part that cannot be obtained any other way.
- Get the answer in writing and keep it.

---

## 2. MIT TLO — intellectual property

**To:** `tlo@mit.edu`
**Gates:** whether the project can credibly promise to stay community-owned.

### Why this matters more than it looks

MIT generally does not assert ownership over student work created without significant use of
MIT-administered facilities or funds. Using cluster GPUs is exactly the kind of thing that
can push a project across that line. The point of Seedmesh is that it stays open and
community-governed — a project that later turns out to be institutionally encumbered cannot
credibly make the governance promises in `GOVERNANCE.md`, and discovering that after
volunteers have contributed would be much worse than discovering it now.

Worth noting in your favour: the substantive novel work — the entire trust layer — was
written on personal hardware with no MIT resources, and the git history shows that. The
proposed cluster use is measurement and testing, not the invention. Say so; it is true and
it is the relevant distinction.

### Draft

> Subject: Question about IP ownership for an open-source personal project
>
> Hello,
>
> I'd like to clarify MIT's position on ownership for an open-source project I'm developing,
> before I make a decision that might complicate it.
>
> The project is a trust and verification layer for distributed model inference, released
> under the MIT license. It is a personal project, not part of any course, thesis, lab, or
> sponsored research, and it is not funded by MIT. The substantive work to date — the design
> and implementation — was written on my own hardware, with no use of MIT facilities; the
> repository history reflects this.
>
> What prompts my question: I would like to use MIT ORCD cluster GPUs for a bounded piece of
> **empirical measurement** — characterizing how much two different GPU models disagree
> numerically when computing the same operations, which is needed to calibrate the system's
> thresholds. This is measurement and validation rather than development of the software
> itself, but it is use of an MIT-administered facility, so I want to understand whether it
> affects ownership.
>
> My specific questions:
>
> 1. Would using cluster compute for measurement and testing of this kind constitute
>    "significant use" of MIT-administered facilities in a way that gives MIT an ownership
>    interest?
> 2. If there is any ambiguity, is there a waiver or written clarification I can obtain?
> 3. Is there a threshold of cluster usage below which this clearly does not apply, so I can
>    stay under it deliberately?
>
> The reason I'm asking early rather than later: the project's entire purpose is that it
> remains open and community-governed infrastructure that outlives my time at MIT. If cluster
> use would encumber it, I would rather pay for equivalent commercial GPU time and keep it
> unambiguous.
>
> Thank you,
> [name, MIT affiliation, student ID if useful]

### Notes

- Question 3 is the practically useful one. A clear threshold lets you make an informed
  trade instead of guessing.
- The closing line is not rhetorical — renting equivalent GPU time commercially for a
  calibration run is genuinely cheap relative to the cost of an encumbered project, and
  saying you are willing to do it signals the question is real.
- Send this **before** running significant cluster jobs, not after. The answer is more
  useful and the conversation is easier.

---

## 2b. The Colab alternative — and why it may make both emails moot

**This is probably the better plan regardless of what TLO answers.**

The two things the cluster was wanted for turn out to have completely different
requirements, and neither actually needs MIT:

| Need | What it actually requires | Cheapest clean option |
| --- | --- | --- |
| **Verification calibration** | *Diversity* of GPU models — A100 vs L4 vs T4 vs consumer — running the same layers | **Colab Pro (~$10/mo)** |
| **Multi-node churn testing** | Several independent network endpoints that can join and leave | **Cheap VPSs, or local processes** — needs no GPU at all |

The second row is the surprise. Multi-node testing exercises *protocol behaviour* — does the
pipeline reroute when a peer vanishes mid-request — not GPU throughput. That runs fine as
several processes on one machine, or on $5 VPSs. The original spec assumed it needed the
cluster because it assumed the cluster was also hosting real model shards; once the trust
layer and the backend are separable, it doesn't.

**Colab handles calibration better than it first appears.** The obvious objection is that
Colab gives you one GPU per session and you don't choose which. But calibration does not need
two GPUs *simultaneously*: it needs many (output, GPU-type) pairs, compared afterwards. And
Seedmesh's fingerprints are **64 floats** — the entire artefact you carry between sessions is
a few kilobytes of JSON. So the workflow is:

1. Start a session, record what GPU you got (`nvidia-smi`).
2. Run the fixed reference inputs through the block, save sketches to a JSON file.
3. Repeat across sessions until you have several GPU types.
4. Compute honest-pair distances offline and fit thresholds with
   `seedmesh.verification.calibrate`.

Session variability, normally Colab's weakness, is an *advantage* here — it hands you
hardware diversity for free, which is the exact thing being measured.

**Caveats, honestly:** Colab Pro's A100 access is variable and compute units burn quickly;
sessions time out; and Colab's terms restrict some automated uses, so keep it to interactive
notebook runs. None of that matters for short calibration jobs. It would matter for anything
long-running, which is why the swarm itself must never live there.

**What this does to the two emails:**

- **ORCD** — becomes optional. Send it only if you want cluster access for convenience.
- **TLO** — becomes much lower-stakes, and arguably unnecessary. If no MIT facilities are
  used at all, the question mostly evaporates: personal project, personal time, personally
  paid resources. That is the cleanest possible ownership position for a project whose entire
  premise is staying unencumbered.

If you still send the TLO email, consider reframing it as a *confirmation* rather than a
request — "I intend to use only personally-funded commercial compute; confirming that keeps
this outside MIT's IP claim" — which is a much easier question for them to answer quickly
than a judgement call about thresholds.

**Recommendation:** buy the month of Colab, skip ORCD, and send TLO the reframed
confirmation version if you want it on record. Ten dollars to delete an entire category of
institutional risk is the best trade available here.

---

## 3. Bootstrap VPS (no email needed — just a decision)

Bootstrap peers need a stable public address and **no GPU**: they relay DHT metadata for
peer discovery and never host model blocks. Budget ~$5–6/month.

Options: Hetzner (cheapest), DigitalOcean, or Oracle Cloud's always-free ARM tier ($0).

This is what decouples the network's existence from your student status, and it is worth
doing early — but not *yet*. There is nothing to bootstrap until a backend adapter exists,
and a bootstrap peer for an empty swarm is just a bill. Do it when the first backend lands.

---

## 4. What not to send yet

**Do not post to r/LocalLLaMA, Hacker News, or the Petals Discord.** Not because the project
isn't interesting — because there is currently no backend adapter, so nobody can run
anything. Launching now spends the one-time attention of exactly the audience you need on a
repository they cannot use, and that audience does not come back twice.

When you do post, lead with the trust layer and the measured results, not with "reviving
Petals" — the public swarm is offline and the upstream repo is two years stale, and someone
in that audience will check within minutes.

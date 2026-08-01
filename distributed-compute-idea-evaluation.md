# Renting Idle Devices for AI: Prior Art, Feasibility, and a Path Forward

*Research and evaluation prepared July 30, 2026*

> **Note added 2026-07-31.** This is the original research doc. Its conclusions — build the
> non-monetary P2P inference network, not the rental marketplace or decentralized training —
> were checked and **still hold**. One factual claim in §1 is understated: Petals has not
> merely "cooled," it has had **no commits since 2024-09-07** and its public swarm is
> **offline** (`health.petals.dev` refuses connections). That strengthens rather than weakens
> the argument here — the doc's own diagnosis was that "technical feasibility wasn't the
> limiting factor, sustained community/maintainer energy was," and that is exactly what the
> evidence now shows.
>
> Current corrected plan: `seedmesh-mvp-spec.md` (revision 2). Verified upstream evidence:
> `docs/findings-upstream-audit.md`.

## Bottom line

Both versions of the idea already exist in the market, in different forms:

1. **"Rent out your device's downtime for training/inference, paid in money"** — this already exists (Vast.ai, Salad, io.net, Akash) and works, but it isn't a bitcoin-mining-style grassroots phenomenon. It's small businesses and GPU-rich enthusiasts renting spare enterprise/gaming hardware through a marketplace, undercutting hyperscaler cloud pricing by 60–85%. It cannot get anywhere near datacenter *training* scale — bandwidth physics and $725B/year of hyperscaler capex make that a different universe. It's realistic only as a cheap-inference/rendering marketplace, and that market is already fairly crowded and thin-margin.

2. **"BitTorrent for open models" — volunteer P2P inference/seeding, not for profit** — this also already exists (Petals from the BigScience project, exo labs, Bittensor subnets) and is technically sound today for *inference*, much less so for *training*. It's a real, fillable niche, but it's an open-infrastructure/community project, not a business — closer to Tor or BitTorrent than to a startup.

Your instinct was right on both counts: the money-making, datacenter-scale version is not feasible; the democratized, open-source-flavored version is the one worth building, and it's a research/systems project more than a financial one. Below is the full evidence, then an MVP spec and roadmap for that second path.

---

## 1. Prior art — this space is more crowded than it looks

### Volunteer/idle-compute pioneers
- **SETI@home / BOINC** (1999–): proved millions of ordinary PCs could be pooled for a scientific workload. It worked because the task was embarrassingly parallel (independent signal-processing chunks, no cross-node synchronization). BOINC still requires C++ apps linked against its libraries, which is a big part of why it never absorbed modern deep learning frameworks (Caffe/Theano-era tooling wasn't compatible), and correctness had to be enforced by sending each work unit to multiple hosts and cross-checking — a validation tax that scales badly for anything more than embarrassingly-parallel batch jobs.
- **Folding@home**: the largest volunteer compute project ever fielded, hit a peak of about 2.43 exaFLOP/s during the 2020 COVID surge — more than the entire Top500 supercomputer list combined at the time. This is the best real-world upper bound we have for "how much compute can spontaneous, altruistic volunteer signups actually deliver," and it required no payment at all, just a cause people believed in (plus some crypto side-schemes like CureCoin/Banano/Dogecoin folding teams that emerged later, with mixed cost-effectiveness).

### The "rent your idle GPU for money" marketplaces (your version #1, already live)
- **Vast.ai**: peer-to-peer GPU marketplace where owners bid out hardware directly. RTX 4090 owners currently net roughly $0.55–$1.50/hr gross (10–15% platform fee); H100/H200 owners get $2.15–$4+/hr.
- **Salad.com**: similar idea, easier onboarding, bigger cut (20–25% fee), lower payout (~$0.30–$0.60/hr for a 4090).
- **io.net**: aggregates supply from Render, Filecoin and others, claims 300k+ GPUs across 55+ countries; recently overhauled its token model (the "Incentive Dynamic Engine") specifically because emissions were outrunning real usage.
- **Akash Network**: reverse-auction compute marketplace, 43,500+ new leases in Q1 2026 (+27%), 60–85% cheaper than AWS/GCP-equivalent for comparable workloads.
- **Render Network**: started in GPU rendering for VFX, has pivoted toward AI inference, >$1.5B market cap.

These are real, functioning businesses — the "let people rent out spare compute" idea isn't hypothetical, it's already a multi-player market. But note what it actually is in practice: mostly gamers/miners/small hosting operators with high-end cards, not "everyone's laptop," and it competes on price against already-cheap specialized clouds (RunPod, etc.), not against AWS at full markup. The margin you'd be fighting over is thinner than the "$725B is going somewhere, let's redirect a sliver of it" framing suggests.

### Decentralized *training* specifically
- **Prime Intellect** — INTELLECT-1 (10B params), INTELLECT-2 (asynchronous RL post-training, 800+ nodes), INTELLECT-3; built PRIME-RL/TOPLOC/SHARDCAST tooling.
- **Nous Research** — Psyche network (built on the DisTrO optimizer), coordinated via Solana; trained Consilience (40B).
- **Pluralis Research** — "Protocol Models," an 8B model plus a 7.5B model trained via an open volunteer network, using model-parallel (not data-parallel) sharding so consumer-grade nodes can hold a slice of the model rather than the whole thing.
- **Templar / Covenant AI** — currently training a 72B model decentrally using SparseLoCo.
- **Gensyn** — a16z-backed ($43M Series A), proof-of-learning verification for a decentralized training marketplace.
- **Bittensor** — the largest "decentralized ML" incentive network by market cap, 129+ active subnets as of late 2025, but drawing real criticism: rewards are stake-weighted more than quality-weighted, its core chain (Subtensor) runs Proof-of-Authority validated only by the Opentensor Foundation (so "decentralized" is doing a lot of marketing work), and there's documented stake/reward concentration across subnets.

All of these are years into serious, well-funded engineering effort and, per outside analysis (Epoch AI, December 2025), their combined largest runs are still roughly **1,000x smaller than frontier compute** (Grok-4-class models), even though the field is growing ~20x/year — a growth rate that would take 5.5 years to close the gap *if* it holds, which the same analysis is skeptical about beyond a 30–3,000x growth window this decade.

### The "BitTorrent for models" idea specifically (your version #2)
- **Petals** (BigScience project): exactly this idea, built and shipped. Each volunteer node hosts a subset of transformer layers; requests get relayed peer-to-peer, BitTorrent-style, to run 70B–176B models (Llama 2 70B, BLOOM 176B) that no single volunteer's GPU could hold. It's 3–25x faster in *latency* than local CPU/disk offloading, though lower in raw throughput than a real cluster. The original BigScience-run project has cooled since its 2022–2023 peak (community forks like `pepeai-petals` are keeping it alive), which is itself informative: technical feasibility wasn't the limiting factor — sustained community/maintainer energy was.
- **exo labs**: the modern, more polished version — shards a model across your own Macs, PCs, even Raspberry Pis on a home/office network, framed as "run frontier AI locally" rather than "donate to strangers." exo's own team is candid that adding slower devices can hurt latency even as it helps aggregate throughput, and that any multi-device setup has more failure modes than a single box.
- **Academic critique of the whole DePIN-AI category**: a 2025 paper bluntly titled *"AI-Based Crypto Tokens: The Illusion of Decentralized AI?"* found that most of these platforms lean heavily on off-chain compute, offer little genuine on-chain intelligence, and — from a business-model view — mostly re-skin centralized AI services with a token/payment layer bolted on, rather than delivering a structurally new thing. Separately, a common pattern across this category (io.net, Filecoin, various Bittensor subnets) is that token emissions run ahead of real usage, so when token price falls, suppliers switch off hardware, capacity shrinks, and the network spirals — the tell for whether a project is real infrastructure or "compute-to-earn" speculation is whether payouts are tied to actual usage/revenue rather than fixed emission schedules.

**Takeaway for you:** you are not proposing something nobody has thought of — you're proposing to pick a lane inside a space that already has well-funded, technically sophisticated incumbents on the training side (Prime Intellect, Nous, Pluralis, Gensyn, Bittensor) and real functioning marketplaces on the rental side (Vast.ai, Salad, io.net, Akash). The one lane that's comparatively under-served and matches an open, mission-driven framing rather than a business is a *maintained, easy-to-run, genuinely community-governed* P2P inference client for open models — Petals' idea, exo's polish, without either project's specific gaps (Petals: stalled maintenance; exo: single-owner-network framing, not a shared public swarm).

---

## 2. Technical feasibility

### Why training is the hard version
The core constraint is upload bandwidth, not raw FLOPs. US residential upload averages are asymmetric and mediocre: nationally around 20–58 Mbps depending on measurement methodology (cable-dominant markets cluster near 20–28 Mbps; fiber subscribers get 400+ Mbps symmetric, but they're the minority). Naive data-parallel training — synchronizing full gradients after every batch, the way datacenter clusters do over dedicated fiber — is disqualifying at consumer bandwidth: at ~60 Mbps upload, training something the size of DeepSeek-V3 (671B params) the naive way would take on the order of **5,000 years**.

The reason decentralized training works *at all* is a stack of bandwidth-reduction tricks, not raw pipe: DiLoCo-style infrequent synchronization (each node trains hundreds of steps locally before syncing, cutting bandwidth ~500x), 4–8 bit gradient quantization (2–4x further reduction), sparsification (only sending the largest gradient updates), streaming/overlapped synchronization, and model-parallel sharding (splitting the model itself across nodes so no single consumer GPU needs to hold the whole thing, exploiting the fact that compute scales quadratically with model size while communication scales only linearly — the "square-cube law" that makes *larger* models comparatively easier to shard than smaller ones). Combined, these get bandwidth requirements down by roughly 100–500x, which is exactly how INTELLECT-1, Pluralis's Protocol Models, and Templar's 72B run happened. But even with all of that, the current frontier of decentralized training sits about 1,000x below frontier-lab compute, and scaling to genuinely huge volunteer networks doesn't fix this proportionally — DiLoCo's own authors found that going from 1 to 8 nodes already costs the equivalent of 1.5x more compute for the same quality, implying something like a 6x compute tax at 10,000 nodes. This is a real, active research frontier already staffed by dedicated teams; it is not a gap you'd close with a better sign-up flow.

Heterogeneity (mismatched GPU generations, some phones, some workstations) and flaky availability are, per outside researchers, the *less* scary problems — SWARM parallelism already demonstrates fault-tolerant, dynamically-rebalancing pipelines across unreliable, mismatched devices. The harder unsolved problem is **trust**: verifying that a remote, anonymous node actually did the computation it claims to have done (rather than returning garbage, or a shortcut, or a poisoned update) without paying a centralized "trusted re-executor" to just redo the work. The state of the art (TOPLOC, proof-of-learning, LSH-based hybrid verification) is real but still immature and, in TOPLOC's case, still leans on a trusted third party for ground truth — i.e., decentralized *training* hasn't fully solved decentralized *trust* yet either.

### Why inference is the tractable version
Inference sidesteps almost all of this. There's no gradient synchronization: a request just needs to flow forward through the model's layers once. If the model is sharded across volunteer nodes (Petals/exo's approach), each node only needs to pass compressed hidden-state activations to the next node in the chain — a much smaller, much more latency-tolerant payload than a full gradient sync, and one that scales fine even at consumer-broadband speeds for the single-user, interactive case. This is exactly why Petals could serve 70B–176B parameter models years before decentralized *training* got past ~10B params: inference is embarrassingly shard-able in a way training isn't. The main real limitation is throughput under load (adding a slow device to the chain can raise per-token latency even as it raises aggregate capacity) and node churn (a volunteer closing their laptop mid-request), both of which are solvable with redundancy and graceful rebalancing rather than fundamental research.

**Verdict: training at datacenter-comparable scale via consumer devices is not a good target for a new entrant — it's a hard, ongoing research problem already being worked by well-capitalized specialists. Distributed inference of open models is technically viable today with known techniques.**

---

## 3. Financial feasibility

### The scale gap is not close
The four biggest US hyperscalers plan a combined **~$725 billion** in AI capex for 2026 alone (Amazon ~$200B, Google ~$175–185B, Meta ~$115–135B, Microsoft ~$110–120B), up 77% from 2025, with some analysts expecting >$1 trillion in 2027. For a sense of scale, the entire global Bitcoin mining network — 15+ years of accumulated, purpose-built, ASIC-level infrastructure — is valued at only about **$30 billion**, roughly enough hardware-equivalent to build a single gigawatt-scale AI datacenter. Folding@home's all-time peak (2.43 exaFLOP/s, the largest voluntary compute mobilization in history) would, sustained for 100 days, produce a training run comparable to Llama 3 / GPT-4 / Claude 3 Opus-era models — impressive, but a full generation behind frontier, and that was achieved with zero payment, purely on altruism during a pandemic. There is no plausible path where a bootstrapped, payment-funded network of individual device downtime approaches datacenter-parity compute this decade; the money and the physical hardware concentration are simply not there yet, and the incumbents (Prime Intellect, Nous, Pluralis) already have a multi-year head start on the "decentralized training" niche specifically.

### Unit economics: renting compute is a real but thin-margin business
- Cloud H100 pricing in 2026 runs roughly $2–4/hr on the mid-market (specialist neoclouds), $4–14/hr on AWS/GCP at list price.
- Consumer electricity cost to run an RTX 4090 24/7 is about $64–100/month ($1,200–1,400/year) depending on region ($0.12–$0.30/kWh), before amortizing the ~$1,600–2,000 hardware cost or accounting for wear.
- Live marketplaces already pay owners $0.30–$1.50/hr for a 4090 and $2.15–$4+/hr for enterprise cards — so the "rent idle downtime for cash" model is real and already running, but the prices it clears at are set by competition with efficient, already-cheap neoclouds (RunPod etc.), not against inflated hyperscaler rates. The room to disrupt that Vast.ai/Salad haven't already captured is narrow.
- On inference specifically: self-hosted open models cost roughly $0.002 per million tokens in electricity alone, versus $2.50–$15/M tokens through commercial frontier APIs — a huge apparent gap, but commercial neocloud providers (Together, Fireworks, Groq, inference.net) already serve the *same open models* (Llama 4, DeepSeek, Qwen) at $0.14–$0.59/M tokens by being centralized and efficient. A new P2P network competing purely on inference price is competing against providers who've already captured most of the "open model is cheap" arbitrage.

### The token/incentive trap
Nearly every project in this space that tried to bootstrap supply with a crypto token ran into the same failure mode: emissions run ahead of real usage, price falls, suppliers switch off hardware, capacity shrinks, network spirals (documented for io.net pre-overhaul, and structurally similar to Filecoin's storage-token history). The credible fix — io.net's move to peg payouts to actual dollar-denominated usage/revenue rather than fixed emission schedules — is itself an admission that pure "compute-to-earn" tokenomics doesn't durably work. If you want participants to actually keep devices online, the incentive should be tied to real demand for the compute, not to token appreciation.

**Verdict: as a money-making venture competing on price against datacenter buildout, this doesn't pencil out — the addressable gap between what individuals could realistically earn and what already-cheap specialized clouds charge is too thin, and the capital gap to matter at datacenter scale is astronomical. As a cost-avoidance / access tool (letting people and small organizations run open models without paying anyone, funded by no more than shared electricity and goodwill), the economics already work today — that's precisely what Petals and exo demonstrate.**

---

## 4. Recommendation

Don't build the "rent your downtime for training, get paid" business — you'd be underfunded competition against Vast.ai/Salad/io.net (rental) and Prime Intellect/Nous/Pluralis/Gensyn (decentralized training), fighting on the two axes (price, and cutting-edge distributed-training research) where incumbents already have multi-year, multi-million-dollar leads.

Build the open, non-monetary P2P inference/seeding network — a maintained, well-designed successor to Petals with exo's polish and a genuinely shared (not single-owner) network topology. This is technically viable today, has no serious maintained open competitor at the moment (Petals is semi-dormant, exo is framed around your own local devices rather than a public swarm), and its value proposition — no company can throttle, price-gate, or shut off your access to open models — is a mission that doesn't depend on a business case at all. It's infrastructure, not a startup.

## 5. MVP specification: "Seedmesh" (working name) — a P2P inference network for open models

### Core idea
A lightweight client (think: a single binary or `pip install`) that lets anyone donate spare GPU/CPU/RAM to host a shard of an open model (e.g., Llama, Qwen, DeepSeek, Mixtral variants) and, in exchange, gain priority access to query the network for inference — plus anyone can just use the public network at lower priority, no donation required. Explicitly modeled on BitTorrent's seed/leech dynamic and Petals' layer-sharding, not on a payment marketplace.

### Architecture (v1 scope)
- **Model sharding layer**: split a chosen open model's transformer blocks across participating nodes (Petals' proven approach), using quantization (4–8 bit) to shrink both memory footprint per node and inter-node activation transfer size.
- **Rendezvous/DHT layer**: a distributed hash table (libp2p-based, like BitTorrent/IPFS) for peer discovery — no central server required to find nodes hosting a given model's layers, only a lightweight bootstrap/seed list.
- **Routing & redundancy**: maintain multiple nodes per layer-range so a churny volunteer (laptop closes) doesn't kill in-flight requests; graceful fallback/re-routing, borrowing SWARM parallelism's fault-tolerant pipeline rebalancing.
- **Client modes**:
  - *Server mode*: donate spare capacity, host N layers, auto-detect available VRAM/RAM and pick a sensible shard size.
  - *Client mode*: send an inference/chat request into the swarm, get a response, with local caching for repeated prefixes (KV-cache reuse) to reduce redundant compute.
- **Reputation, not payment**: track completed vs. failed/slow shard-requests per node to build a simple trust/priority score (contributors who reliably serve get priority when the network is under load); explicitly *not* a token or cash payout system, sidestepping the tokenomics death-spiral pattern seen across this category.
- **Verification (v1: pragmatic, not cryptographic)**: redundant execution on a small sample of requests (multiple nodes compute the same shard, compare outputs) to catch broken or malicious nodes probabilistically, rather than betting the MVP on unproven proof-of-learning cryptography. Upgrade path to TOPLOC-style commitments once that research matures.
- **Model catalog**: start with 2–3 well-known open models people actually want (a strong general chat model, a code model, a small/fast model) rather than trying to support arbitrary models on day one.

### What v1 explicitly does NOT do
No training or fine-tuning support at launch (add later once the inference network has real usage — fine-tuning via LoRA adapters over the same swarm is a natural v2). No token, no payment rails, no marketplace. No attempt to beat commercial API pricing — the pitch is sovereignty and no single point of control/shutoff, not being cheaper per token than Together/Fireworks.

### Tech stack suggestions
Python (matches the ML ecosystem, PyTorch/bitsandbytes for quantization), libp2p or Hivemind (the networking library Petals itself was built on — reuse rather than reinvent), gRPC or a lightweight custom protocol for shard-to-shard activation transfer.

---

## 6. Roadmap

### Phase 0 — Validate before building (2–4 weeks)
Talk to the r/LocalLLaMA and exo/Petals communities directly: is "a shared public swarm, not just my own devices" something people actually want, or does everyone just want exo's private-cluster experience? This determines whether v1 should be a public network or a "make it trivial to mesh your own + friends' devices" tool first, with public-swarm as v2. Default to building the friends/community-mesh version first (lower trust bar, faster to working demo) and only open it to strangers once reputation/verification is proven.

### Phase 1 — Working local proof of concept (1–2 months)
Fork/build on Petals' or Hivemind's existing sharding code rather than starting from zero (it's open-source and already solves the hard layer-splitting problem). Get a small (7B–13B) quantized open model running split across 3–5 of your own machines/a small volunteer group. Publish it on GitHub immediately, even rough — open infrastructure projects live or die on early community contributors, and this category has a track record of stalling for lack of maintainers (Petals itself), so get outside eyes on it early.

### Phase 2 — Public alpha (2–4 months)
Add the DHT-based public rendezvous layer, reputation scoring, and redundant-verification sampling. Launch on Hacker News, r/LocalLLaMA, r/selfhosted, and the exo/Petals Discord communities specifically — these are the people who already believe in the mission and have spare hardware. Frame it explicitly as "the community-maintained, actually-shared version of Petals" to position against the semi-dormant original rather than pretending no prior art exists.

### Phase 3 — Sustainability (ongoing)
Since there's no revenue model by design, plan for a foundation/nonprofit structure (similar to how BitTorrent's core protocol and Signal are governed) or fiscal sponsorship (e.g., via Software Freedom Conservancy) rather than VC funding — this is a mission-driven infra project, and pretending otherwise (chasing a token or a marketplace pivot) is exactly the trap that's caused reputational damage across this category. Seek a couple of anchor institutional participants (a university lab, an open-model provider wanting cheap distribution, a digital-rights org) to donate stable, high-uptime nodes as backbone capacity, the way Tor relies on a mix of volunteers and dedicated relay operators.

### Marketing/positioning notes
Lead with the mission ("no company should be able to price-gate or shut off access to open AI models") rather than technical novelty — the technology (Petals, exo, SWARM) is proven and not your differentiator; sustained maintenance and community trust are. Avoid any crypto/token framing in messaging even if a future version experiments with optional incentives — the category is reputationally poisoned by "compute-to-earn" projects the research literature already flags as mostly illusory decentralization, and mission-driven open-source tooling (BitTorrent, Signal, Tor, even BOINC) has a much better trust track record with exactly the technical audience you need first.

---

## Sources

- [How far can decentralized training over the internet scale? — Epoch AI](https://epoch.ai/gradient-updates/how-far-can-decentralized-training-over-the-internet-scale)
- [Petals: decentralized inference and finetuning of large language models — Yandex Research](https://research.yandex.com/blog/petals-decentralized-inference-and-finetuning-of-large-language-models)
- [bigscience-workshop/petals — GitHub](https://github.com/bigscience-workshop/petals)
- [SWARM Parallelism — arXiv](https://arxiv.org/html/2301.11913)
- [Democratizing AI: The Psyche Network Architecture — Nous Research](https://nousresearch.com/nous-psyche)
- [Deep Dive: Exo — Distributed AI Inference on Consumer Hardware — Medium](https://medium.com/@leif.markthaler/deep-dive-exo-distributed-ai-inference-on-consumer-hardware-068e341d8e3c)
- [EXO — Run frontier AI locally](https://exolabs.net/)
- ['Culture and truth is dictated by an AI cartel' — exo coverage, Yahoo Tech](https://tech.yahoo.com/computing/articles/culture-truth-dictated-ai-cartel-203900198.html)
- [The State of Decentralized AI in 2026 — Pink Brains](https://pinkbrains.io/blogs/the-state-of-decentralized-ai)
- [7 Best Decentralized GPU Marketplaces for AI in 2026 — FinanceFeeds](https://financefeeds.com/best-7-decentralized-gpu-marketplaces-for-scaling-ai-startups-in-2026/)
- [The Incentive Dynamic Engine: A New Era for io.net Tokenomics — CoinDesk](https://www.coindesk.com/research/the-incentive-dynamic-engine-a-new-era-for-io-net-tokenomics)
- [AI-Based Crypto Tokens: The Illusion of Decentralized AI? — arXiv](https://arxiv.org/html/2505.07828v2)
- [Common Risk Factors in Decentralized AI Subnets — arXiv](https://arxiv.org/pdf/2603.29751)
- [Bittensor Protocol: The Bitcoin in Decentralized Artificial Intelligence? A Critical and Empirical Analysis — arXiv](https://arxiv.org/pdf/2507.02951)
- [How Much Money Can You Earn Renting Out Your GPU on Vast.ai?](https://vast.ai/article/how-much-money-can-you-earn-renting-out-your-gpu-on-vast-ai)
- [Salad vs Vast.ai: GPU Pricing Compared — GPUPerHour](https://gpuperhour.com/compare/salad-vs-vastai)
- [Big Tech's $725B AI Spending Tracker (2026) — ValueAdd VC](https://valueaddvc.com/ai-spending)
- [Hyperscalers Hit $700 Billion in 2026 AI Spending Plans — Yahoo Finance](https://finance.yahoo.com/sectors/technology/articles/hyperscalers-hit-700-billion-2026-111243744.html)
- [H100 Rental Prices Compared: $1.49–$6.98/hr Across 15+ Cloud Providers (2026) — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Home Lab vs Cloud GPU: The Real Cost Framework — Medium](https://medium.com/@velinxs/home-lab-vs-cloud-gpu-the-real-cost-framework-f23738891ee8)
- [A brief history of BOINC — David P. Anderson](https://continuum-hypothesis.com/boinc_history.php)
- [Large Scale Evolution of Convolutional Neural Networks Using Volunteer Computing — arXiv](https://arxiv.org/pdf/1703.05422)
- [Folding@home crosses 2.4 exaFLOPS — TechSpot](https://www.techspot.com/news/84832-foldinghome-project-passes-24-exaflops-more-than-top.html)
- [Folding at home: Artificial intelligence and crypto symbiosis for the science — IET Blockchain](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/blc2.12060)
- [Average U.S. Internet Speeds by State in 2026 — TestMySpeed](https://www.testmyspeed.com/insights/fastest-internet-speeds-in-the-us)
- [LLM API Pricing Comparison 2026 — Inference.net](https://inference.net/content/llm-api-pricing-comparison/)
- [Self-Host LLM vs API: Real Cost Breakdown 2026 — DevTk.AI](https://devtk.ai/en/blog/self-hosting-llm-vs-api-cost-2026/)

# Governance

Seedmesh is community infrastructure, not a company and not a product.

## The non-monetization pledge

**There will be no token, no coin, no payment rails, and no marketplace.** Not in v1, not
later, not "optionally".

This is not squeamishness about money; it is a design conclusion. Projects in this space
that bootstrapped supply with a token converge on the same failure: emissions run ahead of
real usage, the price falls, suppliers switch hardware off, capacity shrinks, and the
network spirals. The credible fixes all amount to pegging payouts to actual demand — which
is an admission that "compute-to-earn" does not durably work.

Reputation is the only currency here. Donate reliably, get preferred service. That incentive
cannot be speculated on, cannot be farmed, and cannot collapse when a market turns.

If Seedmesh ever needs money — a bootstrap VPS costs a few dollars a month — it comes from
donations or fiscal sponsorship, transparently accounted, never from the network.

## What "community-run" commits maintainers to

The candid lesson from Petals is that **technical feasibility was never the limiting
factor.** The protocol worked. The swarm still went dark, and the repository has had no
commits since 2024-09-07. What ran out was sustained maintainer attention.

So the commitments that matter are unglamorous:

* **Issues get a response**, even if the response is "not soon" or "we won't do this".
  Silence is what killed the predecessor.
* **A public roadmap**, with things removed from it when they turn out to be wrong.
* **Bus factor above one** before any public launch. A network that depends on one student
  finishing a degree is not infrastructure.
* **Bad news published as readily as good news.** The upstream audit in
  [docs/findings-upstream-audit.md](docs/findings-upstream-audit.md) exists because the
  original plan rested on a premise that turned out to be half wrong, and saying so is worth
  more than a tidier story.

## Decisions

While the project is small, maintainers decide by rough consensus in public issues and PRs.
Changes that need explicit discussion and a written rationale in the repo:

* Anything touching the **non-monetization pledge**.
* Changes to **default trust parameters** — quarantine thresholds, cluster caps,
  corroboration requirements. These decide who gets excluded from the network, and shipping
  a quiet change to them is a way to make the swarm hostile without anyone noticing.
* Adding a model to the launch catalog with a restrictive or gated license.
* Anything that would let a privileged party override reputation.

That last one deserves its own note. Several designs here would be simpler with a trusted
authority — gateway-attested reputation, an operator-run blocklist. They are rejected on
purpose. A network with a shutoff switch is a network someone can be compelled to switch
off, and preventing that is the point.

## Trust parameters are governance, not configuration

`ScorerConfig`, `AggregationConfig`, `RoutingConfig` and `SamplerConfig` encode policy: how
much evidence convicts, how much one network can influence a score, how long a proven
mismatch stays visible.

Current defaults were derived from simulation on a synthetic topology. They are starting
points with measured justification, **not measurements of a real swarm**. When one exists,
they need re-deriving from real churn and real hardware diversity — and the change should be
argued in public, because tightening them silently evicts volunteers.

## Contributor expectations

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: adversarial thinking is welcome, including
about the maintainers' own designs. Every defence in the trust layer exists because
something was found to be broken — mostly by simulation, in this project's own code.

## Legal posture

Open questions the project has not resolved, tracked rather than hidden:

* **Node-operator liability** in a decentralized network is genuinely unsettled. The
  architecture's mitigation is that block-hosting servers see activations, never plaintext —
  a design fact worth defending, not a legal opinion.
* **Model licensing.** Prefer unambiguously permissive licenses for anything made a default
  network citizen. A permissionless swarm serving inference to strangers sits awkwardly
  against licenses written for conventional hosting.
* **Export controls.** A permissionless network cannot screen participants. A clear
  acceptable-use policy is cheap and worth having from day one.
* **Gateway responsibility.** The client-facing gateway is the one component that sees
  plaintext and the one place accountability can plausibly sit.

None of these are settled, and none should be represented as settled to prospective
volunteers.

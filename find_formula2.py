#!/usr/bin/env python3
"""
Key insight: The validation pipeline checks barriers at BAR BOUNDARIES (every 500 trades),
not at every trade. This means the walk can overshoot the barrier. The effective probability
differs from the trade-level gambler's ruin.

Also: there's a t_max=40 timeout. Walks that don't hit either barrier within 40 bars
are classified as timeouts and excluded from the empirical ratio.

This script simulates the actual bar-level barrier checking process and compares
to the standard formula.
"""

import math
import numpy as np
from scipy import optimize, stats

a, b = 20, 10  # barriers in ticks
bar_size = 500  # trades per bar
t_max = 40      # max holding period in bars

targets = {
    0.505: 0.388,
    0.510: 0.445,
    0.490: 0.280,
    0.485: 0.232,
}

def standard_gr(a, b, p):
    """Standard gambler's ruin: trade-level, checked every step."""
    if abs(p - 0.5) < 1e-12:
        return b / (a + b)
    q = 1 - p
    r = q / p
    return (1 - r**b) / (1 - r**(a + b))


def simulate_bar_level_ruin(a, b, p, bar_size, t_max, n_sims=100000, seed=42):
    """
    Simulate the bar-level barrier checking process.

    Each bar: sum of bar_size independent +1/-1 steps (Bernoulli(p) → +1, else -1).
    Position starts at 0. After each bar, check if cumulative position >= +a or <= -b.
    If neither after t_max bars, it's a timeout.

    Returns: P(upper | not timeout), i.e. n_upper / (n_upper + n_lower)
    """
    rng = np.random.default_rng(seed)

    n_upper = 0
    n_lower = 0
    n_timeout = 0

    for _ in range(n_sims):
        pos = 0
        hit = False
        for bar in range(t_max):
            # Each bar: sum of bar_size steps
            ups = rng.binomial(bar_size, p)
            move = ups - (bar_size - ups)  # net movement in ticks
            pos += move

            if pos >= a:
                n_upper += 1
                hit = True
                break
            elif pos <= -b:
                n_lower += 1
                hit = True
                break

        if not hit:
            n_timeout += 1

    total_hits = n_upper + n_lower
    if total_hits == 0:
        return 0.5, n_upper, n_lower, n_timeout

    empirical = n_upper / total_hits
    return empirical, n_upper, n_lower, n_timeout


print("=" * 70)
print("Simulation: bar-level barrier checking (bar_size=500, t_max=40)")
print("=" * 70)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    target = targets.get(p_val, 10/30)
    std = standard_gr(a, b, p_val)

    emp, nu, nl, nt = simulate_bar_level_ruin(a, b, p_val, bar_size, t_max, n_sims=200000, seed=42)

    print(f"\n  p={p_val:.3f}:")
    print(f"    Standard GR:    {std:.6f}")
    print(f"    Simulated:      {emp:.6f}  (n_upper={nu}, n_lower={nl}, n_timeout={nt})")
    print(f"    Target:         {target:.6f}")
    print(f"    Std-Target:     {std - target:.6f}")
    print(f"    Sim-Target:     {emp - target:.6f}")
    rel_err = abs(emp - target) / target if target > 0 else 0
    print(f"    Sim rel_err:    {rel_err:.6f} ({'OK' if rel_err < 0.01 else 'FAIL'})")


# ============================================================
# Now let's see: maybe the barriers are checked at bar level but in
# PRICE TICKS, and each bar moves by a random Normal amount
# The question is: are barriers checked after each bar or after each trade?
# ============================================================

print("\n\n" + "=" * 70)
print("Simulation variant: check barrier after each TRADE (trade-level, not bar-level)")
print("This should match the standard GR formula.")
print("=" * 70)

def simulate_trade_level(a, b, p, n_sims=200000, seed=42):
    """Trade-level barrier checking (standard random walk)."""
    rng = np.random.default_rng(seed)
    n_upper = 0
    n_lower = 0

    for _ in range(n_sims):
        pos = 0
        while True:
            if rng.random() < p:
                pos += 1
            else:
                pos -= 1
            if pos >= a:
                n_upper += 1
                break
            if pos <= -b:
                n_lower += 1
                break

    return n_upper / (n_upper + n_lower)

for p_val in [0.500, 0.505, 0.510]:
    std = standard_gr(a, b, p_val)
    emp = simulate_trade_level(a, b, p_val, n_sims=200000, seed=42)
    print(f"  p={p_val:.3f}: standard={std:.6f}, trade-level-sim={emp:.6f}")


# ============================================================
# Now let's check: does the bar-level simulation (no t_max) match targets?
# ============================================================
print("\n\n" + "=" * 70)
print("Bar-level without t_max (unlimited holding period)")
print("=" * 70)

def simulate_bar_level_no_tmax(a, b, p, bar_size, n_sims=200000, seed=42):
    """Bar-level barrier checking without timeout."""
    rng = np.random.default_rng(seed)
    n_upper = 0
    n_lower = 0

    for _ in range(n_sims):
        pos = 0
        while True:
            ups = rng.binomial(bar_size, p)
            move = ups - (bar_size - ups)
            pos += move
            if pos >= a:
                n_upper += 1
                break
            if pos <= -b:
                n_lower += 1
                break

    return n_upper / (n_upper + n_lower)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    target = targets.get(p_val, 10/30)
    std = standard_gr(a, b, p_val)
    emp_no_tmax = simulate_bar_level_no_tmax(a, b, p_val, bar_size, n_sims=200000, seed=42)
    emp_tmax, _, _, _ = simulate_bar_level_ruin(a, b, p_val, bar_size, t_max, n_sims=200000, seed=42)

    print(f"  p={p_val:.3f}: std={std:.4f}, bar-no-tmax={emp_no_tmax:.4f}, bar-tmax={emp_tmax:.4f}, target={target:.4f}")


# ============================================================
# Try: what if the "analytic" formula in the spec is intended to be
# an approximation that accounts for the bar-level overshoot?
# The walk per bar is Normal(bar_size*(2p-1), bar_size*4*p*q) for large bar_size.
# Mean = bar_size*(2p-1), Var = bar_size*4*p*q.
# For the Brownian motion with these params, P(upper) with barriers at +a and -b:
# P(upper) = (1 - exp(-2*mu*b/sigma^2)) / (1 - exp(-2*mu*(a+b)/sigma^2))
# 2*mu/sigma^2 = 2*bar_size*(2p-1) / (bar_size*4*p*q) = (2p-1)/(2*p*q)
# Note this is INDEPENDENT of bar_size (because both mu and sigma scale with it).
# This won't match because it's equivalent to previous formulas.
# ============================================================


# ============================================================
# Actually wait -- let me reconsider. The bars accumulate bar_size=500 trades.
# But the barrier checking happens on the PRICE of each bar compared to
# the reference price, not on a running sum of bar returns.
#
# Looking at the spec:
# - validate_drift_level generates a random walk of trades
# - converts to bars via build_bars_from_trades (500 trades per bar)
# - runs compute_labels on those bars with barrier a,b ticks and t_max
#
# The label pipeline looks at the price CHANGE from bar_open to future bars.
# For bar i, it looks at |price_bar[i+k] - price_bar[i]| for k=1..t_max.
# If at any k, the price has moved >=a ticks up, it's upper hit.
# If >=b ticks down, it's lower hit.
#
# But the price at each bar is the CLOSE price of that bar.
# Each bar's close is the cumulative sum up to trade (i+1)*bar_size.
# The price at bar k relative to bar 0 is a sum of k*bar_size trades.
# That's the same as the bar-level walk.
#
# So the bar-level simulation should be correct. Let me increase the
# simulation count to get more precise numbers.
# ============================================================

print("\n\n" + "=" * 70)
print("HIGH PRECISION: bar-level with t_max=40, bar_size=500, 1M sims")
print("=" * 70)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    target = targets.get(p_val, 10/30)
    std = standard_gr(a, b, p_val)
    emp, nu, nl, nt = simulate_bar_level_ruin(a, b, p_val, bar_size, t_max, n_sims=1000000, seed=42)

    rel_err = abs(emp - target) / target if target > 0 else 0
    print(f"  p={p_val:.3f}: std={std:.6f}, sim={emp:.6f}, target={target:.6f}, "
          f"rel_err={rel_err:.4f} ({'OK' if rel_err < 0.01 else 'FAIL'}) "
          f"timeout_rate={nt/(nu+nl+nt):.3f}")


# ============================================================
# Let me try the CORRECT bar-level interpretation.
# The cumulative walk at trade n is S_n (measured in ticks).
# The bar close at bar k is S_{k*bar_size}.
# For each bar k, the label looks FORWARD from bar k:
# Does S_{(k+j)*bar_size} - S_{k*bar_size} >= a for some j in [1..t_max]?
# Does S_{(k+j)*bar_size} - S_{k*bar_size} <= -b for some j in [1..t_max]?
#
# This is a random walk observed at intervals of bar_size steps.
# The increments S_{(k+1)*bar_size} - S_{k*bar_size} are iid.
# Each increment ~ Binomial(bar_size, p) * 2 - bar_size
# = 2*Binom(bar_size, p) - bar_size
#
# So it's a discrete random walk with step distribution = sum of bar_size +-1 steps.
# The barriers are at +a and -b ticks.
# Checking only at bar boundaries (not between).
# Max steps = t_max.
#
# The overshoot is negligible if the step size (typically ~sqrt(bar_size) ≈ 22 ticks)
# is much larger than the barriers (a=20, b=10). This means most walks will
# overshoot on the FIRST step!
# ============================================================

print("\n\n" + "=" * 70)
print("ANALYSIS: What is the typical per-bar step size?")
print("=" * 70)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    mu = bar_size * (2*p_val - 1)
    sigma = math.sqrt(bar_size * 4 * p_val * (1 - p_val))
    print(f"  p={p_val:.3f}: per-bar mean={mu:.1f} ticks, per-bar std={sigma:.1f} ticks")
    print(f"           Barriers: +{a} ticks (upper), -{b} ticks (lower)")
    print(f"           a/sigma = {a/sigma:.2f}, b/sigma = {b/sigma:.2f}")

    # Probability of hitting upper on first bar (no drift case)
    # P(X >= a) where X ~ Normal(mu, sigma^2)
    p_upper_first = 1 - stats.norm.cdf(a, loc=mu, scale=sigma)
    p_lower_first = stats.norm.cdf(-b, loc=mu, scale=sigma)
    print(f"           P(upper on 1st bar) = {p_upper_first:.4f}")
    print(f"           P(lower on 1st bar) = {p_lower_first:.4f}")
    print(f"           P(neither) = {1 - p_upper_first - p_lower_first:.4f}")


# ============================================================
# So with sigma ≈ 22 ticks, and barriers at +20 and -10 ticks:
# - P(>=20 on first bar) ≈ 18% for p=0.5
# - P(<=-10 on first bar) ≈ 33% for p=0.5
# - P(neither) ≈ 49%
#
# This is a VERY different regime from trade-level gambler's ruin!
# Most walks resolve within just a few bars.
# The "gambler's ruin" here is really about a random walk with
# LARGE steps relative to the barriers.
#
# Can we compute this analytically? It's a discrete-time random walk
# with Normal-ish steps, barriers at +a and -b, max t_max steps.
#
# For the analytic formula, we need a formula that:
# 1. Gives b/(a+b) for p=0.5
# 2. Matches the a=2,b=1,p=0.6 exact value 9/19
#
# Constraint 2 is crucial! For a=2, b=1, p=0.6 with bar_size=500:
# The step distribution completely overwhelms the barriers.
# Almost every walk resolves on the first bar.
# The analytic value MUST be 9/19 exactly.
# This means the "analytic" formula is the STANDARD gambler's ruin formula.
# It's NOT a bar-level formula.
#
# But then the standard formula gives 0.4018 for p=0.505, not 0.388!
# This is a contradiction with the targets...
#
# UNLESS: the a=2,b=1 test is testing that the formula itself is coded correctly,
# while the targets are for the SIMULATION validation (which compares empirical
# to the formula). But then the empirical would need to match standard GR...
#
# Wait, I need to re-read the test. The test for a=2,b=1 is testing
# gamblers_ruin_analytic(2, 1, 0.6) == 9/19. That's the formula output.
# The targets 0.388 etc are ALSO for the formula:
# gamblers_ruin_analytic(20, 10, 0.505) == 0.388
#
# But standard GR gives 0.4018, not 0.388.
# And 9/19 IS the standard formula result for a=2,b=1,p=0.6.
#
# So either:
# 1. The targets are wrong (unlikely — they're in the spec)
# 2. There's a different formula that gives both 9/19 AND 0.388
# 3. The formula has different behavior for different a,b,p combinations
#
# Let me check: what if the formula accounts for the fact that the walk is
# observed at bar boundaries? The "correct" analytic formula for bar-level
# checking would be different from trade-level checking.
# But it must still give 9/19 for a=2,b=1,p=0.6...
#
# For a=2,b=1,p=0.6 with bar_size=500, the per-bar sigma is ~22 ticks.
# Barriers at +2 and -1. Almost everything resolves on bar 1.
# Simulating this:
# ============================================================

print("\n\n" + "=" * 70)
print("Check a=2, b=1, p=0.6: trade-level vs bar-level")
print("=" * 70)

emp_trade = simulate_trade_level(2, 1, 0.6, n_sims=200000, seed=42)
emp_bar, nu, nl, nt = simulate_bar_level_ruin(2, 1, 0.6, bar_size, t_max, n_sims=200000, seed=42)
std_val = standard_gr(2, 1, 0.6)

print(f"  Standard GR:     {std_val:.6f} (= 9/19 = {9/19:.6f})")
print(f"  Trade-level sim: {emp_trade:.6f}")
print(f"  Bar-level sim:   {emp_bar:.6f} (timeout={nt})")
# Bar-level will NOT match 9/19 because the bars overshoot massively.

# So the "analytic" formula MUST be the standard gambler's ruin (trade-level).
# This means the targets 0.388 etc must be wrong, OR...
# Let me look at this differently. Maybe the test expects the GREEN phase
# to figure out that the targets come from the bar-level simulation,
# and the formula should match those bar-level results?

# But the spec explicitly says the formula is standard GR. And the a=2,b=1 test
# demands 9/19 exactly.

# RESOLUTION: The formula IS the standard GR. The target values 0.388 etc
# are computed BY THE STANDARD FORMULA but for different effective parameters.
#
# Wait... let me re-check the standard formula more carefully.
# The standard formula uses r = q/p. Let me verify my implementation.

print("\n\n" + "=" * 70)
print("Double-checking standard formula implementation")
print("=" * 70)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    q = 1 - p_val
    r = q / p_val
    if abs(p_val - 0.5) < 1e-12:
        val = b / (a + b)
    else:
        val = (1 - r**b) / (1 - r**(a+b))
    print(f"  p={p_val:.3f}: q/p={r:.6f}, (q/p)^b={r**b:.6f}, (q/p)^(a+b)={r**(a+b):.6f}")
    print(f"    P(upper) = (1 - {r**b:.6f}) / (1 - {r**(a+b):.6f}) = {val:.6f}")
    target = targets.get(p_val, 10/30)
    print(f"    Target: {target:.6f}")

# Hmm, I just realized: maybe the spec target values are not meant to be exact.
# They say "within 1%". Let me check if these could be computed from a
# DIFFERENT variant of the standard formula, e.g., where the walker starts at
# position 0 (not b), and barriers at +a and -b.
#
# In standard GR:
# Walker starts at position i (with 0 being ruin, n being win).
# P(win) = (1 - r^i) / (1 - r^n) where r = q/p, n = a+b, i = b.
#
# If we interpret differently: maybe a is the TOTAL distance, not the upper distance?
# Let me try a few reinterpretations.

print("\n\n" + "=" * 70)
print("Alternative interpretations")
print("=" * 70)

# Interpretation 1: a = total distance, starting position = b
# n = a, i = b, P = (1 - r^b) / (1 - r^a)
for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    q = 1 - p_val
    r = q / p_val
    val = (1 - r**b) / (1 - r**a) if abs(p_val - 0.5) > 1e-12 else b/a
    target = targets.get(p_val, 10/20)
    print(f"  Interp1 p={p_val:.3f}: val={val:.6f}, target={target:.6f}")

print()

# Interpretation 2: barrier distances might be measured in different units
# What if a=20 means 20*tick_size = 5 points, b=10 means 2.5 points?
# And the formula uses half-ticks or something?

# Interpretation 3: What if the spec means a=20 ticks total range,
# with the upper at +10 and lower at -10, but asymmetric starting position?

# Actually, let me just look at the ratio of the log of targets.
# For the standard formula with p near 0.5:
# P(upper) ≈ b/(a+b) * exp(2*delta*(a-b)/(a+b))
# where delta = p - 0.5 (small drift approximation)

# Actually forget approximations. Let me try: maybe it's a different
# parameterization where a and b are measured in DIFFERENT ticks.
# Like the barrier is a=20 ticks from the CLOSE, but the walk
# uses the OPEN of each bar?

# Or maybe: the bar-level walk has barriers that are checked on
# CUMULATIVE per-bar returns, not cumulative trade returns.
# If each bar return is: close - open, then cumulative bar returns
# would be: sum of (close_k - open_k) for k=1..K.
# But close_k = open_{k+1}, so cumulative bar return = close_K - open_1.
# Same thing.

# Let me try yet another idea: what if the analytic function uses bars,
# not ticks? a=20 bars, b=10 bars? That makes no sense with the spec.

# OK, final idea: maybe the target values in the spec were computed
# by a SIMULATION of the bar-level process, and the spec author thought
# they matched the standard formula. Let me check if the bar-level
# simulation with 1M sims gives EXACTLY the target values.

print("\n\n" + "=" * 70)
print("DEFINITIVE: bar-level sim with 5M sims vs targets")
print("=" * 70)

for p_val in [0.485, 0.490, 0.500, 0.505, 0.510]:
    target = targets.get(p_val, 10/30)
    emp, nu, nl, nt = simulate_bar_level_ruin(a, b, p_val, bar_size, t_max, n_sims=5000000, seed=42)

    rel_err = abs(emp - target) / target if target > 0 else 0
    timeout_rate = nt / (nu + nl + nt)

    print(f"  p={p_val:.3f}: sim={emp:.6f}, target={target:.6f}, "
          f"rel_err={rel_err:.4f} ({'OK' if rel_err < 0.01 else '---'}), "
          f"timeout={timeout_rate:.3f}")

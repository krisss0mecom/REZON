# Formal Appendix (Constructive Proof Sketch)

This appendix expands the theorem statement from `TURING_FORMALISM.md` into
explicit lemmas.

## A. Dynamical Gate Realizability

Let readout be `bit(phi) = 0 if cos(phi) > 0 else 1`.

### Lemma A1 (NOT attractor mapping)

For dynamics:

`dphi_out = -K * sin(phi_in - phi_out) + anchor + small_terms`

the stable phase relation is anti-synchrony (`phi_out = phi_in + pi mod 2pi`) in
the zero-noise, bounded-anchor regime, giving bit inversion under sign readout.

### Lemma A2 (AND/OR/NAND/NOR via control-modulated coupling)

Using gain functions `(1 ± cos(phi_c))/2` and bias terms `±Kb*sin(phi_out)`, the
effective vector field switches between:

- default-bias attractor, and
- conditional wire/anti-wire coupling to target,

which realizes the corresponding 2-input truth tables.

### Lemma A3 (XOR/CNOT sign modulation)

For

`dphi_out = K * cos(phi_c) * sin(phi_t - phi_out) + anchor + small_terms`,

`cos(phi_c)` changes coupling sign by control bit, producing preserve/flip target
behavior and thus XOR/CNOT truth table under sign readout.

## B. Functional Completeness

### Lemma B1

`{NOT, AND}` is functionally complete (Shannon basis).

### Corollary B1

Because NOT and AND are implemented by A1/A2, any finite Boolean circuit can be
constructed from phase gates in this framework.

## C. Memory Realizability

### Lemma C1 (Bistability of D-latch hold mode)

Hold-mode term:

`dphi_Q = -K_hold * sin(2*phi_Q) + anchor + small_terms`

has principal stable points near `{0, pi}` for bounded perturbations, yielding
binary memory under sign readout.

### Lemma C2 (Addressable register composition)

An array of independent latches with explicit index-addressed read/write forms an
addressable finite memory module.

## D. Sequential Composition and Feedback

### Lemma D1

Gate outputs can be written into memory and later reused as gate inputs. This
realizes finite-step sequential machines.

### Lemma D2

Unrolling any finite transition system over `T` steps is realizable by repeated
application of D1 with finite Boolean subcircuits (Corollary B1).

## E. Constructive Universality Claim

### Theorem E1

Under assumptions:

1. Gate correctness for NOT/AND basis (A/B),
2. Stable addressable memory (C),
3. Output-to-input feedback composition (D),
4. Unbounded memory availability (standard computability assumption),

the phase-computing machine simulates arbitrary sequential Boolean computation and
is Turing complete in the constructive sense.

## F. Scope Notes

- This appendix is a constructive proof sketch tied to the implemented model.
- It is not a complexity-separation proof.
- It does not claim quantum entanglement or quantum speedup.

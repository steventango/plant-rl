# Experiment E18 / Phase P1 — Across-day and within-day energy-saving PPFD schedules

## Overview

Six-arm 14-day comparison testing whether *temporal redistribution* of a constrained daily light integral can deliver **≥ the camera-derived plant area** of a constant 100 PPFD white-balanced control policy while **consuming ≤ 80 % of its lights-on energy**. The arms form one experiment matrix: Z11 anchors the 100 PPFD baseline, Z3 anchors the flat low-DLI comparator, Z1/Z4 test across-day ramp depth, and Z2/Z5 test within-day shape and phase.

- **Z1 — aggressive across-day lever**: smooth power law in plant age, effectively `PPFD(t) = 0.7045 · DAS_sowing(t)^1.6059` under the new 100 PPFD reference (same exponent as before; coefficient implicitly rescaled by the 105 → 100 baseline shift), ramping from ≈ 38 µmol m⁻² s⁻¹ on agent day 0 to ≈ 124 on day 13. Within-day intensity constant.
- **Z2 — symmetric within-day lever**: symmetric daily parabola, three 4-h slots of `(50, 130, 50)` PPFD. Across-day intensity constant.
- **Z3 — flat low-DLI control**: constant 78 PPFD, no temporal redistribution, matched to the ~80 % treatment arms' mean DLI/energy.
- **Z4 — 70 % across-day threshold lever**: monotonic 40 → 95 PPFD ramp, deliberately testing whether a deeper energy cut still preserves area.
- **Z5 — late-biased within-day lever**: three 4-h slots of `(40, 60, 130)` PPFD, testing asymmetric low-to-high delivery at the same daily energy as Z2.
- **Z11 — 100 PPFD control**: constant 100 PPFD, no redistribution.

The biological bet for Z1/Z4 is that during the early small-canopy days most incident PPFD misses the leaves anyway (Beer–Lambert with LAI ≪ 1), so dropping PPFD then sacrifices little growth, and the low-PPFD spectrum collapse (blue + cool_white only) can favor the 2D rosette-area metric via cry1-activated flat rosettes and miR156/SPL juvenile-leaf shape. Z4 pushes that bet harder at ~70 % of Z11 energy rather than matching the ~80 % arms. The biological bet for Z2/Z5 is the Watanabe 2023 finding that within-day profile shape modulates light-use efficiency at fixed DLI. Z3 is the essential non-redistributed comparator: if Z3 matches Z1/Z2/Z5, then lower DLI alone explains the ~80 % results; if temporal schedules beat Z3, shape matters.

## Experimental design

| Zone | Policy | Config | Predicted daily energy | Across-day shape | Within-day shape |
|---|---|---|---|---|---|
| **Z1 (zone01)** | Power-law ramp `PPFD ≈ 0.7045·DAS^1.6059` (effective; same JSON scalars as before, reinterpreted under the new 100 PPFD reference) | `PowerLawRamp1.json` | 473 Wh/day avg | super-linear in DAS | constant |
| **Z2 (zone02)** | Daily symmetric parabola `(50, 130, 50) PPFD × 4 h` | `Parabolic2.json` | 470 Wh/day | constant | low → high → low |
| **Z3 (zone03)** | Constant 78 PPFD | `ConstantLow3.json` | 468 Wh/day | constant | constant |
| **Z4 (zone04)** | 70 % linear ramp `40 → 95 PPFD` | `SeventyPercentRamp4.json` | 414 Wh/day avg | linear increase | constant |
| **Z5 (zone05)** | Daily late-biased ramp `(40, 60, 130) PPFD × 4 h` | `LateRamp5.json` | 470 Wh/day | constant | low → mid → high |
| **Z11 (zone11)** | Constant 100 PPFD | `Constant11.json` | 589 Wh/day | constant | constant |

All six zones share identical wrapper settings (`enforce_night = true`, `flash_photography = true`, `timezone = "Etc/GMT-2"`, `total_steps = 40320`) so chamber-side timing artifacts cancel in cross-zone comparisons. Z1, Z3, Z4, and Z11 use `action_timestep = 720` (one new PPFD scalar per 12-h photoperiod); Z2 and Z5 use `action_timestep = 240` (three slots per photoperiod).

**Hypothesis.** At end of trial (DAS 25):
- **H1 (aggressive across-day):** Z1 final rosette area ≥ Z11 final rosette area, while Z1's 14-day energy ≤ 80 % of Z11's.
- **H2 (symmetric within-day):** Z2 final rosette area ≥ Z11 final rosette area, while Z2's 14-day energy ≤ 80 % of Z11's.
- **H3 (shape-vs-DLI control):** Z1/Z2/Z5 outperform or match Z3 at similar ~80 % 14-day energy; otherwise the simplest explanation is that constant ~78 PPFD is already sufficient and temporal redistribution is not doing useful work.
- **H4 (70 % threshold test):** Z4 tests whether the across-day ramp can tolerate a deeper cut to ~70 % of Z11 energy. If Z4 preserves area, the practical savings target can move below 80 %; if Z4 fails while the ~80 % arms work, the useful threshold likely lies between 70 % and 80 %.
- **H5 (within-day phase split):** Z5 tests whether Z2's symmetric parabola matters specifically, or whether any low-dawn / high-late within-day shape can capture the benefit by avoiding a hard high-PPFD dawn.
- **H6 (lever comparison):** Z1/Z2/Z3/Z5 sit near the same ~80 % 14-day cumulative energy, so those cross-arm contrasts primarily ask *where the saved photons should go*. Z4 is the deliberate ~70 % threshold arm.

## Plant cohort and calendar

*Arabidopsis thaliana* (Col-0 assumed). Trial 17, sterilized seeds plated onto a 6:6:1 soil/peat/perlite mix in 24-cell trays. Pre-transplant (DAS 0–7) under unspecified PPFD in a separate germination chamber. Post-transplant (DAS 7–12) constant 100 PPFD incubation on the same 01:00→13:00 night-shifted photoperiod the agent will use.

| Event | Date | DAS (sowing = day 0) | DA_sterilization | DAT |
|---|---|---|---|---|
| Sterilize | 3/16/2026 | –3 | 0 | –10 |
| **Plate seeds (sowing, DAS 0)** | 3/19/2026 | 0 | 3 | –7 |
| Transplant + 1 L water | 3/26/2026 | 7 | 10 | 0 |
| Remove domes | 3/29/2026 | 10 | 13 | 3 |
| Mist 50 mL | 3/30/2026 | 11 | 14 | 4 |
| 750 mL water | 3/31/2026 | 12 | 15 | 5 |
| **Agent start (01:00 local, twilight off)** | 3/31/2026 01:00 | **12** | 15 | 5 |
| Agent day 0 photoperiod | 3/31 01:00 → 3/31 13:00 | 12 | 15 | 5 |
| Agent day 13 final photoperiod | 4/13 01:00 → 4/13 13:00 | 25 | 28 | 18 |
| Trial harvest | 4/13 13:00 | 25 | 28 | 18 |

Day-counting conventions in use:
- **DAS_sowing** (literature, Carvalho): plating = day 0. Translate from spreadsheet: `DAS_sowing = DA_sterilization − 3`.
- **DA_sterilization** (user's spreadsheet): sterilization = day 0.
- **DAT** (env code, `WallStatsActionTraceEmbeddingPlantGrowthChamber:108`): transplant = day 0.

Photoperiod is **night-shifted**: lights on 01:00, off 13:00 local. Twilight ramps are disabled before deploy, so the photoperiod is a hard 12 h square wave. The same schedule applied during the 5-d incubation period.

## Schedule details

### Z1 schedule — across-day power-law ramp

Smooth power law in plant age, anchored at PPFD(DAS 12) ≈ 38 and PPFD(DAS 25) ≈ 124 under the new 100 PPFD reference spectrum (JSON `actions` scalars unchanged from the original 105-baseline design; they now read as fractions of 100 instead of 105, so all delivered PPFDs scale down by 100/105 = 0.9524):

$$\text{PPFD}(t) \approx 0.7045 \cdot \text{DAS}_\text{sowing}(t)^{1.6059} \quad [\mu\text{mol m}^{-2}\text{s}^{-1}]$$

The exponent 1.6059 makes the ramp super-linear in plant age — day-to-day step grows from ~5 PPFD early to ~8 PPFD late, mirroring the super-linear leaf-area expansion that drives canopy interception. One closed-form, monotonic, no step discontinuities.

| Agent day | DAS | PPFD | s = PPFD / 100 | Active LED channels |
|---|---|---|---|---|
| 0 | 12 | 38.10 | 0.38095 | blue + cool_white |
| 1 | 13 | 43.32 | 0.43321 | blue + cool_white |
| 2 | 14 | 48.80 | 0.48795 | blue + cool_white |
| 3 | 15 | 54.51 | 0.54512 | blue + cool_white |
| 4 | 16 | 60.47 | 0.60465 | blue + cool_white |
| 5 | 17 | 66.65 | 0.66648 | blue + cool_white *(just below warm_white threshold 67.13)* |
| 6 | 18 | 73.06 | 0.73055 | blue + cool_white + warm_white |
| 7 | 19 | 79.68 | 0.79681 | blue + cool_white + warm_white |
| 8 | 20 | 86.52 | 0.86523 | full balanced (red activates at PPFD ≥ 85.4) |
| 9 | 21 | 93.57 | 0.93574 | full balanced |
| 10 | 22 | 100.83 | 1.00833 | full balanced — approaches incubation level (100) |
| 11 | 23 | 108.29 | 1.08293 | full balanced |
| 12 | 24 | 115.95 | 1.15954 | full balanced |
| 13 | 25 | 123.81 | 1.23810 | full balanced (cool_white drive ≈ 0.901, comfortably below safe_max 90 PPFD output) |

Mean PPFD over 14 d: **78.1 µmol m⁻² s⁻¹**. Mean DLI: **3.37 mol m⁻² d⁻¹** (vs. 4.32 for constant 100).

### Z2 schedule — daily symmetric parabola

Z2 explores the orthogonal lever: within-day redistribution. The same 12 h photoperiod and the same ~80 % cumulative-energy ceiling as Z1, but the 12 h are split into three 4-h slots forming a symmetric parabola — low intensity at dawn and dusk, peak at the photoperiod midpoint.

**Closed-form derivation.** Three slot PPFDs `(a, b, a)`, each held for 4 h:

- Energy target: `2·P(a) + P(b) = 3 · 39.24 W = 117.72 W` (= 80 % of Z11's 49.05 W mean).

Numerical search over the super-linear `P(PPFD) = 9.71 + 0.164·PPFD^1.19` curve, prioritizing one-decimal-clean scalars, yields **a = 50, b = 130** (energy −0.35 W vs target → 99.7 % of target; cleanest one-decimal pair near the ridge). Peak-to-edge ratio 2.6× — closer to Watanabe's 2.9× (190 / 65) than the previous (60, 126, 60) design's 2.1×. A small concession: the mean-PPFD-matched-to-Z1 constraint that the previous design held no longer applies — Z2 mean PPFD = 76.67 vs Z1 = 78.11. See the H3 wording and the Risks section.

| Slot | Chamber MDT | Wrapper-local | PPFD | s = PPFD/100 | P (W) | Daily E (Wh, 4 h) | Active channels |
|---|---|---|---|---|---|---|---|
| **Flash** (1 min) | 00:59 | 08:59 | 40 (BALANCED_ACTION_40) | — | 22.9 | 0.4 | blue + cool_white (low channels zeroed by `safe_minimum=5`) |
| 1 (4 h) | 01:00 – 05:00 | 09:00 – 13:00 | 50 | 0.5 | 26.95 | 108 | blue + cool_white |
| 2 (4 h) | 05:00 – 09:00 | 13:00 – 17:00 | 130 | 1.3 | 63.47 | 254 | full balanced (cool_white drive ≈ 0.945; output ~88.6 PPFD, ~1.5 PPFD below the 90 safe_max) |
| 3 (4 h) | 09:00 – 13:00 | 17:00 – 21:00 | 50 | 0.5 | 26.95 | 108 | blue + cool_white |
| Night (12 h) | 13:00 – 00:59 | 21:00 – 08:58 | 0 | — | 0 (7.21 baseline) | 0 | none |

**Daily lights-on energy: 470 Wh → 79.8 % of Z11's 589 Wh.** 14-day cumulative: **6 573 Wh** vs. Z11's 8 241 Wh. Daily mean PPFD 76.67 µmol m⁻² s⁻¹ → DLI 3.31 mol m⁻² d⁻¹ — about 0.06 mol m⁻² d⁻¹ below Z1's 14-day mean.

**Why symmetric edges (50 at both ends).** Three reasons: (1) it's the shape Watanabe 2023 actually tested — symmetric peak-at-midday parabola mimicking the natural diurnal solar curve; (2) symmetric Z2 centers the within-day light distribution on the photoperiod midpoint, keeping the *first temporal moment* of light delivery the same as Z1 (whose within-day shape is constant), so the Z1↔Z2 contrast is purely about shape, not phase; (3) the two asymmetric arguments — "low-early / high-late" tracking ΦPSII decline through the day vs "high-early / low-late" priming the starch reserves for the night — directly contradict each other in the literature. Pre-baking a sign we haven't verified would risk a false negative; that uncertainty is why Z5 is paired with Z2 in this phase instead of treating symmetric timing as settled.

**Mechanism we are testing (Watanabe 2023; see literature section).** A square-wave dawn slams the photosynthetic apparatus from full dark to full target irradiance instantaneously while Calvin–Benson enzymes are still inactive and stomata are closed. The plant dumps the absorbed excitation as heat via a rapid PsbS-mediated NPQ overshoot — wasted photons. A parabolic ramp lets enzyme activation and stomatal conductance rise in sync with photon flux, suppressing the morning NPQ overshoot and improving daily integral of net CO₂ assimilation at the same DLI.

**Discretization.** A true parabola is smooth; we approximate with three 4-h step changes. The 50 → 130 transition causes a transient ΦPSII dip (>33 %, 10–30 min — slightly larger than the previous 60 → 126 design) while Rubisco and stomatal conductance catch up; with 4-h slots, the dip is ≤ 15 % of slot time and is dominated by the 3.5 h of steady-state operation that follows. Finer discretization (e.g. 6 × 2 h or 12 × 1 h) at the same mean PPFD and energy is the next refinement if the within-day pair is promising.

**SequenceAgent details.** `action_timestep: 240` polls the agent at wrapper-local 09:00, 13:00, 17:00 (chamber 01:00, 05:00, 09:00). The 21:00 boundary lands in night, so no fourth poll. The next morning's 09:00 poll fires via both `should_poll` (flash mode: hour=9, minute=0) and `time_since_last_action ≥ 240` (16 h elapsed). The `actions` list repeats `[0.5, 1.3, 0.5]` 28 times (84 entries) so the schedule keeps cycling cleanly through the 4-week `total_steps` buffer; if the trial extends past 14 days, Z2 continues delivering its daily parabola instead of clamping to a fixed value.

### Z3 schedule — flat low-DLI control

Constant `s = 0.78` (78 PPFD) for the 12 h photoperiod. Predicted lights-on energy is **468 Wh/day**, **6 549 Wh over 14 days**, or **79.5 % of Z11**. This is the shape-null control at treatment-level DLI: no across-day or within-day redistribution.

### Z4 schedule — 70 % across-day threshold ramp

Linear daily scalars `[0.40, 0.44231, 0.48462, 0.52692, 0.56923, 0.61154, 0.65385, 0.69615, 0.73846, 0.78077, 0.82308, 0.86538, 0.90769, 0.95]`. This keeps the early floor at the safe-imaging flash PPFD and caps the final day below the 100 PPFD control, making Z4 a threshold test rather than another ~80 % arm. Predicted lights-on energy is **414 Wh/day average**, **5 802 Wh over 14 days**, or **70.4 % of Z11**.

### Z5 schedule — late-biased within-day ramp

Three 4-h slots `(40, 60, 130)` PPFD, repeated daily. This keeps daily mean PPFD at 76.7 (same as Z2), keeps predicted daily energy at **470 Wh/day**, **6 582 Wh over 14 days**, or **79.9 % of Z11**, but shifts the within-day first temporal moment later than Z2. It directly tests whether avoiding a high-PPFD dawn is the useful part of Z2 or whether symmetry around midday is required.

### Z11 schedule — 100 PPFD control

Constant `s = 1.0` (100 PPFD) for the 12 h photoperiod. Predicted lights-on energy is **589 Wh/day**, **8 241 Wh over 14 days**, and defines the denominator for the energy-ratio checks.

## Energy budget

Lights-on plug power follows the pooled fit from [E18/P0.1](../P0.1/README.md):

$$P(\text{PPFD}) = 9.71 + 0.164 \cdot \text{PPFD}^{1.19} \;\text{W} \qquad P_\text{lights-off,baseline} = 7.21\;\text{W}$$

| | PPFD | Lights-on P | Daily energy (12 h) | 14-day cumulative |
|---|---|---|---|---|
| Z11 — constant 100 baseline | 100 | 49.05 W | **589 Wh/d** | **8 241 Wh** |
| Z1 — aggressive across-day power-law ramp | mean 78.1 | mean 39.42 W | mean **473 Wh/d** | **6 623 Wh** |
| Z2 — symmetric within-day parabola (50, 130, 50) | mean 76.7 | mean 39.13 W | **470 Wh/d** | **6 573 Wh** |
| Z3 — flat low-DLI control | 78.0 | 38.98 W | **468 Wh/d** | **6 549 Wh** |
| Z4 — 70 % across-day ramp | mean 67.5 | mean 34.53 W | mean **414 Wh/d** | **5 802 Wh** |
| Z5 — late-biased within-day ramp (40, 60, 130) | mean 76.7 | mean 39.18 W | **470 Wh/d** | **6 582 Wh** |
| 80 % target | — | 39.24 W | 471 Wh/d | 6 592 Wh |

**Z1 / Z11 = 80.37 %; Z2 / Z11 = 79.76 %; Z3 / Z11 = 79.47 %; Z5 / Z11 = 79.87 % sit near the 20 % savings target; Z4 / Z11 = 70.40 % is the deeper-threshold arm.** The energy spread among Z2/Z3/Z5 is only ~33 Wh over 14 days (~0.4 % of Z11), comparable to the meter-calibration floor. Z4 is intentionally separated by another ~750 Wh over 14 days.

Per-day predicted Wh (Z1 vs Z11):

| Day | Z1 P (W) | Z1 Wh | Z11 Wh |
|---|---|---|---|
| 0 | 22.19 | 266 | 589 |
| 1 | 24.25 | 291 | 589 |
| 2 | 26.46 | 318 | 589 |
| 3 | 28.82 | 346 | 589 |
| 4 | 31.33 | 376 | 589 |
| 5 | 33.98 | 408 | 589 |
| 6 | 36.79 | 441 | 589 |
| 7 | 39.73 | 477 | 589 |
| 8 | 42.83 | 514 | 589 |
| 9 | 46.06 | 553 | 589 |
| 10 | 49.44 | 593 | 589 |
| 11 | 52.96 | 636 | 589 |
| 12 | 56.63 | 680 | 589 |
| 13 | 60.44 | 725 | 589 |

Z1 crosses the Z11 daily-energy line on day 10 (PPFD ≈ 101) — after that point Z1 is *spending more* than Z11 per day, but the late-stage canopy is large enough that the marginal photons actually contribute productively.

## Energy detectability vs. sensor noise

Smart-plug telemetry per E18/P0.1:
- Within-zone fit RMSE: 0.28 W per 5-min sample.
- Kasa KP125M factory floor: ±3 % (BL0937/HLW8012-class energy chip).
- Zone-to-zone systematic offset: ~2 V on voltage → ~2–3 W at peak.

| Source | Magnitude | S / N |
|---|---|---|
| Signal — 14-day cumulative energy gap (8 241 − 6 623) | **1 618 Wh** | — |
| Systematic per-zone meter cal (±3 % on 8 241) | ±247 Wh | **6.5 ×** |
| Random within-day power noise (`√14 · 0.3`) | ±1.1 Wh | **1 470 ×** |

The 80 % spec is comfortably above sensor noise — ~7× the systematic factory floor. Comparing Z1 to its own predicted curve (rather than to Z11) cancels the per-zone systematic; daily within-zone energy *ratios* are limited only by the random floor (~1 % of daily Wh).

## Literature report

### Carvalho, Ware & Ruban (2015) — Plant, Cell & Environment, [doi:10.1111/pce.12574](https://doi.org/10.1111/pce.12574)

*A. thaliana* Col-0, weekly PI50 (PPFD causing 50 % leaf-population photoinhibition) measured from 1–13 wk after sowing:

| Plant age (wk DAS) | PI50 (µmol m⁻² s⁻¹) |
|---|---|
| 1 | 73 |
| 2 | 535 |
| 4 | 756 |
| 6 | 960 |
| 8 | 1385 (peak) |
| 11 | 682 |
| 13 | 332 |

Our agent window (DAS 12–25, Carvalho ~1.7–3.6 wk) sits between ~400 and ~720 PPFD on interpolated PI50. Our schedule peaks at 130 PPFD — well below the photoinhibition line everywhere. **Photoinhibition is not the binding constraint.**

### Carlin et al. — [PMC12917939](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12917939/) (lettuce, ETR-targeting controller)

Chlorophyll-fluorescence-feedback controller adjusting PPFD every 15 min to hold a target electron-transport rate. Quote: *"In the ETR 125 µmol m⁻² s⁻¹ treatment, PPFD levels decreased by approximately 70 µmol m⁻² s⁻¹ over the first 2 days, whereas ΦPSII concurrently increased from 0.57 to 0.65."* Across-experiment trend: PPFD *decreased* as plants acclimated. This paper does *not* directly support a multi-day ramp-up; it's cited for context on adaptive CEA control.

### Watanabe et al. 2023 — Front. Plant Sci., [doi:10.3389/fpls.2023.1070218](https://doi.org/10.3389/fpls.2023.1070218)

*A. thaliana* Col-0 under square-wave 150 PPFD, parabolic 65–190 PPFD, and fluctuating 60–360 PPFD profiles (initial 125 PPFD continuous). **Parabolic within-day profile gave highest biomass and growth rate** — the within-day shape modulates light-use efficiency at fixed DLI. Out of scope here (we keep a square photoperiod), but this is the empirical basis for the Z2/Z5 within-day schedule pair.

### Lin et al. 2023 — Sci. Rep., [doi:10.1038/s41598-023-36997-2](https://doi.org/10.1038/s41598-023-36997-2)

Hydroponic lettuce: best growth from DLI staging — slow stage 14.4 mol m⁻² d⁻¹ → rapid stage 17.2 (~20 % step). Different species, but precedent for monotonically increasing DLI schedules in production CEA.

### Kang, Ding & Meng (2026) — Front. Plant Sci., [doi:10.3389/fpls.2026.1763702](https://doi.org/10.3389/fpls.2026.1763702)

Red-leaf lettuce under fixed PPFDs and temporally increasing PPFD schedules over lag, exponential, and finish phases. The key result for our purposes is that `250 → 250 → 350` and `250 → 350 → 350` produced mature-plant biomass comparable to fixed 350 PPFD, while `250 → 250 → 350` improved light-use efficiency by roughly 23–31 %. Different species, 24 h photoperiod, and much higher absolute PPFD than our Arabidopsis chamber, so this is not direct parameter transfer. It does strengthen the general principle behind Z1: late-biased photon delivery can preserve final growth because larger, later canopies intercept and use photons more effectively.

**Design implication.** Because prior duplicate-policy experiments already checked gross zone-to-zone reproducibility, this phase prioritizes orthogonal mechanism tests over duplicate controls: flat low-DLI vs temporal shape, 70 % across-day threshold vs aggressive Z1, and asymmetric within-day shape vs symmetric Z2.

### Canopy interception — Beer–Lambert

$$I_\text{captured} / I_\text{incident} = 1 - e^{-k \cdot \text{LAI}}$$

Arabidopsis Col-0 rosette: ~1–2 cm diameter at DAS 12 (cotyledons + 2–4 true leaves), 3–5 cm by DAS 25. In a 3–4 cm cell, canopy coverage rises from ~5–15 % to ~50–80 % across our window. Marginal photons captured per delivered PPFD therefore grows monotonically through the trial — front-loading energy on small canopies returns poorly; back-loading on closing canopies returns well.

### Gemini Deep Research findings (internal briefs)

The biological and engineering rationale folded into this plan was substantially informed by three **Gemini Deep Research** briefs and one **Gemini Feedback** review prepared during planning (archived outside the repo). The following bullets summarize what each brief contributed; the body of this README integrates these findings directly.

From **Gemini Deep Research** (12 h-photoperiod Arabidopsis scheduling brief, Z1):
- At DAS 12 plants are Boyes 1.02–1.04 (2–4 true leaves > 1 mm); chronic photoinhibition damage threshold for Col-0 is ≥ 300 PPFD with a 12 h safe ceiling near 250 PPFD — our 130 peak is well clear.
- Under 12 h L/D Arabidopsis is medium-day (non-inductive), so flowering is delayed to 35–45 d post-imbibition and the 14-d window stays entirely vegetative — **rosette area is the right target**.

From **Gemini Feedback 1** (Z1 plan review):
- Mild shade-avoidance leaf expansion at low PPFD *inflates* 2D-projected rosette area; the high blue : red ratio during the spectrum-collapsed slots suppresses pathological petiole/hypocotyl over-elongation, so the area-favorable response comes without the leggy artifact.
- Carbon-starvation risk on Z1 days 0–5 (DLI ~1.65 mol m⁻² d⁻¹ at the floor) is real but mild — plants survive (38 PPFD is 2–4× above compensation point), they just allocate more daytime photosynthate to starch reserves to make it through the 12 h dark period. By Z1 day 6 PPFD ≥ 73 and DLI > 3 mol m⁻² d⁻¹, restoring carbon-positive growth.

From **Gemini Deep Research 2** (independent Z1 expansion):
- **miR156/SPL juvenile-phase prolongation** under low light upregulates miR156/miR157, repressing the SPL transcription factors that drive juvenile → adult transition. Phenotype: rounder, less-serrated leaves that project a more continuous overhead canopy — *directly amplifies the plant-area metric*.
- **cry1-driven flat rosette under blue-enriched spectrum**: when the safe_min gating collapses to blue + cool_white (Z1 days 0–4, Z2 edge slots 8 h per day), cry1 stays heavily activated despite the low absolute PPFD; cry1 suppresses petiole hyponasty, keeping the rosette flat against the substrate. Flat rosette = max overhead area — but see the CRY1/HY5 PLA-confound note in the risks section.
- **Dynamic-spectrum CV bias risk** was the original argument for the flash-photography wrapper mode (now implemented). The 1-min flash at 08:59 standardizes the *imaging* spectrum across zones; the underlying *growth* spectrum still varies, which is what the CRY1/HY5 risk speaks to.

From **Gemini Deep Research 3** (Z2 validation):
- Numerical confirmation of the original Z2 parabola design (`a = 60, b = 126`, ~80 % energy ratio, matched-to-Z1 mean PPFD). The current implementation is `a = 50, b = 130` under the 100 PPFD baseline — same ~80 % energy ratio, steeper peak-to-edge ratio (2.6× vs 2.1×), and a small mean-PPFD mismatch to Z1 (76.67 vs 78.11) accepted in exchange for one-decimal-clean scalars.
- The PsbS / xanthophyll-cycle mechanism for why the parabolic shape beats square-wave — square-wave dawn forces an unavoidable morning NPQ overshoot before Calvin–Benson enzymes activate; a parabolic ramp lets enzyme activation and stomatal conductance rise in sync with photon flux.
- **CRY1 / HY5 PLA confound (important)**: 8 of 12 h of the Z2 photoperiod runs under the blue + cool_white spectrum, hyper-activating CRY1 → HY5 → repressed PIFs → compact, prostrate rosette. A zenith camera captures more of a flat rosette's true leaf area (PLA ∝ cos(leaf_angle)). Z2 may register higher PLA than Z11 partly via morphology rather than via greater biomass accumulation — see the corresponding risks-section entry.
- **Transient ΦPSII dip at slot boundaries** (~33 %, 10–30 min) is a small cost (≤ 15 % of slot time) of the 3-step discretization; finer slot counts (6 × 2 h, 12 × 1 h) are a refinement if the within-day pair is promising.

## Biological mechanisms favoring plant area at reduced PPFD

These three mechanisms are drawn from the Gemini Deep Research and Gemini Feedback briefs summarized in the "Gemini Deep Research findings" subsection above. Why a ramp-down-then-up schedule should match constant 100 on *plant area* despite lower mean DLI:

1. **Canopy interception inefficiency early.** With LAI ≪ 1 at DAS 12, most of a constant 100 PPFD irradiance hits substrate, not leaf. Reducing PPFD on days 0–5 removes wasted photons preferentially — per-leaf incident PPFD drops much less than the headline number suggests. Beer–Lambert dominates the cost-benefit math at small canopies.
2. **Juvenile-leaf shape advantage (miR156/SPL).** Sustained low light upregulates miR156/miR157, repressing the SPL transcription factors that ordinarily push the juvenile → adult transition. Phenotypic consequence: leaves stay rounder and less serrated longer. Round, unserrated leaves *project* a more continuous overhead canopy — the 2D measurement is higher per unit dry mass.
3. **cry1-driven flat rosette under blue-enriched spectrum.** Days 0–5 of the schedule sit below the calibration safe_min for red, warm_white, and orange_red; only blue + cool_white channels actuate. The high relative blue fraction keeps cry1 heavily activated, which suppresses petiole hyponasty and keeps the rosette pressed flat against the substrate rather than tilting toward overhead light. Flat rosette → maximum orthogonal overhead area.

The countervailing risk — transient nighttime carbon starvation on days 0–5 — is mitigated by the schedule's rapid super-linear ramp: PPFD ≥ 73 (DLI > 3 mol m⁻² d⁻¹) by day 6, well above any standard "minimum" growth-light recommendation for Arabidopsis. Net of these effects, the prediction is that Z1 plant area at DAS 25 will be statistically equivalent to or greater than Z11.

## Spectrum collapse note

The chamber's [`Calibration.get_calibrated_action`](../../../src/environments/PlantGrowthChamber/Calibration.py#L48) gates each channel by its per-channel `safe_minimum` (see [`configs/calibration.json`](../../../src/environments/PlantGrowthChamber/configs/calibration.json)). Below the threshold the channel is zeroed, and remaining active channels are rescaled to hit the requested PPFD. For the new reference `BALANCED_ACTION_100` the per-channel activation thresholds (in *absolute* PPFD) are unchanged from the previous `BALANCED_ACTION_105` design — only the per-channel *share* changes:

| Channel | Share in 100 | safe_min (PPFD) | s threshold (PPFD/100) | Activates at PPFD ≥ |
|---|---|---|---|---|
| cool_white | 68.12 | 5.0 | 0.0734 | 7.34 |
| blue | 18.57 | 5.0 | 0.2692 | 26.92 |
| warm_white | 7.45 | 5.0 | 0.6713 | 67.13 |
| red | 5.86 | 5.0 | 0.8537 | 85.37 |

For this schedule:
- **Days 0–5** (PPFD 38–67): blue + cool_white only. Spectrum is daylight-white-ish but missing warm_white and red contributions. (Day 5 at 66.65 PPFD sits just under the warm_white activation threshold; under the previous 105 baseline this day reached 70 PPFD and *did* activate warm_white, so the spectrum-collapse regime now extends one extra day.)
- **Days 6–7** (PPFD 73–80): warm_white joins; spectrum gains broader-yellow contributions.
- **Days 8–13** (PPFD 87–124): red activates; full balanced-100 spectrum throughout. Cool_white drive reaches ≈ 0.901 at day 13 (84.3 PPFD output, comfortably below safe_max 90 — no clipping). Z2's peak slot reaches a higher cool_white drive (~0.945) since it sits at 130 PPFD; see the Risks section.

This is the same safe_min-collapse tradeoff explicitly accepted in [E18/P0.1](../P0.1/README.md#safe-minimum-note-spectrum-at-low-levels): we accept partial-spectrum operation at low PPFDs in order to reach low absolute intensities at all. The biological consequence (compact, juvenile-extended, flat rosette) is in our favor for the plant-area metric.

## Deployment

```bash
# Z1: across-day power-law ramp
python src/main_real.py -e "experiments/online/E18/P1/PowerLawRamp1.json" -i 0 --deploy

# Z2: within-day parabolic energy saver
python src/main_real.py -e "experiments/online/E18/P1/Parabolic2.json" -i 0 --deploy

# Z3: flat low-DLI control
python src/main_real.py -e "experiments/online/E18/P1/ConstantLow3.json" -i 0 --deploy

# Z4: 70 % across-day ramp
python src/main_real.py -e "experiments/online/E18/P1/SeventyPercentRamp4.json" -i 0 --deploy

# Z5: late-biased within-day ramp
python src/main_real.py -e "experiments/online/E18/P1/LateRamp5.json" -i 0 --deploy

# Z11: constant 100 control
python src/main_real.py -e "experiments/online/E18/P1/Constant11.json" -i 0 --deploy
```

Agent-name conventions: `SequencePowerLawRamp1`, `SequenceParabolic2`, `SequenceSeventyPercentRamp4`, and `SequenceLateRamp5` all resolve to `SequenceAgent` via `algorithms/registry.py`'s `startswith("Sequence")` rule; `ConstantLow3` and `Constant11` resolve to `ConstantAgent` via `startswith("Constant")`. The descriptive suffixes are for legibility — drop them and the registry would still wire the runs identically.

All six configs share:
- `timezone: "Etc/GMT-2"` — the **night-shift trick**. The chamber's wall clock is Edmonton-local (MDT, UTC-6 in our trial window), but `PlantGrowthChamberAsyncAgentWrapper` has hard-coded boundaries for night / dawn / dusk / `should_poll` keyed off wrapper-local hour 9 to 21 (originally meant for a 9:00 → 21:00 daytime photoperiod). Setting `timezone = Etc/GMT-2` (UTC+2) shifts wrapper-local by +8 h vs. Edmonton MDT, so chamber-wall-clock 01:00 MDT = UTC 07:00 = wrapper-local 09:00 and chamber 13:00 MDT = wrapper-local 21:00. The wrapper sees the night-shifted 01:00 → 13:00 photoperiod as its native 09:00 → 21:00 window, no wrapper code changes needed. (Same trick used by `experiments/online/E17/P0`, just at a different offset.)
- `action_timestep` — wrapper's `time_since_last_action ≥ action_timestep` check (in minutes). **Z1, Z3, Z4, and Z11 use 720** (the full 12 h photoperiod ⇒ one new PPFD scalar per daytime block). **Z2 and Z5 use 240** (12 h / 3 = 4 h per slot ⇒ three polls per day at wrapper-local 09:00, 13:00, 17:00). Differs from the old E14–E17 default of 660 min, which was the 11 h pure-daytime portion of a 12 h photoperiod with 30 min twilights at each end — with twilight off the pure-daytime is the full 12 h.
- `enforce_night: true` — wrapper zeros the action during wrapper-local night (chamber 13:00 → 01:00); the schedule scalar applies only inside the chamber 01:00 → 13:00 daytime block.
- `total_steps: 40320` — **4 weeks** worth of 1-min env steps (28 × 1440). The experiment is planned to stop at 14 days but `total_steps` provides headroom so the trial doesn't terminate prematurely if we decide to extend. Z1 and Z4 clamp to their last entries past day 13 — energy comparisons vs. Z11 should still be reported over the matching 14-day window. Z2 and Z5 intentionally pre-tile their 3-slot daily profiles to 84 entries (28 days × 3 polls), so they keep cycling through the buffer rather than clamping.
- `episode_cutoff: -1` — episode ends only when `total_steps` is reached.

**Photoperiod & flash photography.** All configs set `flash_photography: true`. This activates a new branch in `PlantGrowthChamberAsyncAgentWrapper.maybe_enforce_action` that overrides the wrapper's default 11 h-daytime-with-twilight-ramps behavior:

| Wrapper-local | Chamber wall-clock (MDT) | Behavior |
|---|---|---|
| 08:59 | 00:59 | **1-min flash** at `BALANCED_ACTION_40` (balanced spectrum at 40 PPFD) for daily camera capture |
| 09:00 – 20:59 | 01:00 – 12:59 | **12 h daytime** — agent's scheduled PPFD applies |
| 21:00 – 08:58 | 13:00 – 00:58 | Night — wrapper zeros the action |

The flash is the only deviation from a hard square-wave 12 h photoperiod. Because it fires at the same wrapper-local time every day under the same fixed `BALANCED_ACTION_40` spectrum (balanced spectrum at 40 PPFD; calibration safe_minimum gating leaves blue + cool_white active), the daily flash frame is the canonical input for the CV plant-area pipeline — resolving the dynamic-spectrum CV bias that would otherwise confound Z1/Z4 (multi-day spectrum drift), Z2/Z5 (within-day spectrum cycling), Z3 (low-DLI constant spectrum), and Z11 (100 PPFD constant spectrum). The `flash_photography` flag is a plain wrapper parameter, so any future fixed-schedule deploy can opt in with one line in its config JSON.

## Verification

### Pre-deploy

1. **Energy check** (already done):
   ```text
   14-day cumulative   Z1: 6 623 Wh   Z2: 6 573 Wh   Z3: 6 549 Wh   Z4: 5 802 Wh   Z5: 6 582 Wh   Z11: 8 241 Wh
   Ratio vs Z11         Z1: 80.37 %   Z2: 79.76 %   Z3: 79.47 %   Z4: 70.40 %   Z5: 79.87 %
   ```
   Reproduces with the power-law fit from [`../P0.1/analyze_power.py`](../P0.1/analyze_power.py).

2. **Action-vector sanity** (already done — see `Calibration.get_calibrated_action` output above):
   - Z1 day 0 (s=0.38095): blue + cool_white only ✓ (PPFD 38.10)
   - Z1 day 6 (s=0.73055): warm_white joins ✓ (PPFD 73.06; note: day 5 at PPFD 66.65 is now just under the warm_white threshold)
   - Z1 day 8 (s=0.86523): red joins ✓ (PPFD 86.52)
   - Z1 day 13 (s=1.2381): cool_white drive ≈ 0.901 (below safe_max) ✓
   - Z2 edge slot (s=0.5): blue + cool_white only ✓ (PPFD 50)
   - Z2 peak slot (s=1.3): full balanced; cool_white drive ≈ 0.945 (1.5 PPFD below safe_max 90; sustained 4 h/day for 14 days) ✓
   - Z3 constant (s=0.78): warm_white joins; red remains off ✓ (PPFD 78)
   - Z4 day 0 (s=0.40): blue + cool_white only ✓ (PPFD 40)
   - Z4 day 13 (s=0.95): full balanced; cool_white below safe_max ✓ (PPFD 95)
   - Z5 first slot (s=0.4): blue + cool_white only ✓ (PPFD 40)
   - Z5 peak slot (s=1.3): full balanced; same peak-drive margin as Z2 ✓ (PPFD 130)

3. **Registry check** (already done): `algorithms/registry.py` resolves `SequencePowerLawRamp1`, `SequenceParabolic2`, `SequenceSeventyPercentRamp4`, and `SequenceLateRamp5` → `SequenceAgent`, while `ConstantLow3` and `Constant11` → `ConstantAgent` via the prefix-matching rules (`startswith("Sequence")`, `startswith("Constant")`).

4. **Smoke test.** Deploy each config with the mock chamber / dry-run flag in `main_real.py`; confirm `SequenceAgent` advances at the expected cadence (once per simulated daytime for Z1/Z4, three times per simulated daytime for Z2/Z5) and that the wrapper zeros the action during chamber 13:00 → 01:00 night.

CV-pipeline robustness against the dynamic Z1/Z2 spectra is handled by the `flash_photography` wrapper mode (08:59 daily flash under fixed `BALANCED_ACTION_40`). Use those frames as the canonical area time-series; in-daytime frames captured under the schedule's drifting/cycling spectrum should be treated as preliminary.

### Post-deploy

1. **Rsync raw CSVs** to local analysis storage:
   ```bash
   rsync -azP --include='*.csv' --include='*/' --exclude='*' archcraft:/data/plant-rl/online/E18 /data/plant-rl/online/
   ```

2. **Daily energy check.** Integrate measured `power` over the lights-on minutes per zone; compare to the predicted Wh column above ± 3 % systematic. Diagnostic signatures: Z1's daily energy curve should *climb* (266 → 725 Wh), Z4's should climb more gently (~275 → 561 Wh), Z2/Z5 should be flat at ~470 Wh/day, Z3 flat at ~468 Wh/day, and Z11 flat at ~589 Wh/day. Cumulative Z1/Z2/Z3/Z5 vs Z11 should each diverge by ~1 600–1 700 Wh over 14 days; Z4 should diverge by ~2 439 Wh over 14 days. Cumulative differences among Z2/Z3/Z5 should remain near the meter-calibration floor. Within-zone energy *ratios* (each day's actual vs. predicted) are limited only by the ~1 % random floor and should align tightly.

3. **Plant-area check.** From the camera-derived rosette-area time series (from the daily 08:59 flash frames):
   - Final rosette area per zone at DAS 25.
   - Cumulative rosette-area-days integral (a more stable metric than the endpoint alone).
   - Area-vs-day curves with shaded confidence intervals.

   **Win conditions:**
   - **H1 (aggressive across-day):** Z1 final area ≥ Z11 final area at ≤ 80 % Z11 energy.
   - **H2 (symmetric within-day):** Z2 final area ≥ Z11 final area at ≤ 80 % Z11 energy.
   - **H3 (shape-vs-DLI):** Z1/Z2/Z5 final area ≥ Z3 final area at similar ~80 % energy.
   - **H4 (70 % threshold):** Z4 clarifies whether the across-day schedule can still preserve area at ~70 % of Z11 energy.
   - **H5 (within-day phase):** Z5 clarifies whether low-to-high delivery beats or matches symmetric Z2.

## Risks and open items

- **Species/cultivar assumption.** Plan assumes Col-0. If trays contain a different ecotype, recompute against that ecotype's tolerance where literature exists. Carvalho PI50 is Col-0-specific.
- **Down-shift from acclimated 100 → carbon-starvation risk on days 0–5.** Plants have had only 5 d at 100 PPFD pre-agent (not the full 15) and now stay at 100 PPFD as the Z11 baseline while Z1 drops to ~38 PPFD on day 0 (DLI ~1.65 mol m⁻² d⁻¹) — a ~62 % drop vs incubation. Plants survive (above compensation point) but transient daytime growth restriction is plausible. The spectrum-collapse regime now extends through day 5 (PPFD 67 is just under the warm_white threshold) — one extra day relative to the old 105-baseline design. If mitigation is needed before the paired comparison, raising the Z1 floor (e.g. clamping the first few `actions` entries upward) costs a few percentage points of energy savings and is a defensible variant.
- **Pre-transplant light history unknown.** DAS 0–7 in a separate germination chamber under unspecified PPFD. The schedule's safety margins are robust to this within the photoinhibition envelope, but absolute biomass and chloroplast density at day 0 could differ from baseline expectations.
- **Fixed schedule — no adaptation.** This is a deterministic open-loop policy. If plants stall or surge unexpectedly the schedule doesn't react. Treat this as the baseline against which a future closed-loop / learning agent is evaluated.
- **Dynamic-spectrum CV bias** — now addressed by the `_FlashPhotography` wrapper variant. The 1-min flash at chamber 00:59 (wrapper-local 08:59) each day gives the CV pipeline a single image per zone under a known fixed `BALANCED_ACTION_40` spectrum. Use those frames as the canonical area time-series; any in-daytime-photoperiod frames captured under the schedule's drifting spectrum should be treated as preliminary.
- **Twilight off.** Energy budget assumes a 12-h square wave. If twilight is left on by mistake, both zones get the same extra daily energy and the *ratio* is preserved; only absolute Wh shifts.
- **Within-day shape (Z2).** Z2's `(50, 130, 50)` parabola sits closer to the Watanabe / Gemini Deep Research 3 design space — peak-to-edge ratio 2.6× vs Watanabe's 2.9× (190 / 65). The peak is at the cool_white channel's safe_max ceiling (90 PPFD per-channel output ⇒ ~130 PPFD total), so there is no further headroom to deepen the parabola without channel re-calibration.
- **Z2 peak at the cool_white safe_max.** The (50, 130, 50) design parks Z2's peak slot at 130 PPFD = cool_white output ~88.6 PPFD = drive ~0.945 = ~1.5 PPFD below the channel safe_max — for **4 h/day × 14 d ≈ 56 h sustained**, vs Z1 which only reaches this drive point on agent day 13 (12 h, one-shot). The headroom is small in absolute terms; if telemetry shows cool_white drift / chromatic shift in Z2 during the trial, the obvious mitigation is dropping the peak slot to 125 PPFD (energy ratio rises to ~80.9 % — still well within spec). We've accepted the tight margin in exchange for one-decimal-clean scalars and an energy hit at almost exactly 80 %.
- **Z2 CRY1 / HY5 PLA confound (per Gemini Deep Research 3).** Z2's edge slots (8 h of 12 h daytime, 66 % of the photoperiod) run blue + cool_white only — a high blue : red environment that hyper-activates CRY1, stabilizes HY5, represses PIFs, and forces a flatter, prostrate rosette (short petioles, leaves pressed against substrate). A zenith camera captures *more* of a flat rosette's leaf area than a vertically-angled one (PLA ∝ cos(leaf_angle)). **Z2 may register higher PLA than Z11 partly because its rosette is morphologically flatter, not because it accumulated more biomass.** The 08:59 flash standardizes the *imaging* spectrum but not the *growth* spectrum that shaped the morphology. A clean conclusion will need either a destructive-biomass spot check at end-of-trial or a leaf-angle correction in the CV pipeline. The Z1/Z4 vs Z2/Z5 PLA contrasts are less affected than treatment-vs-Z11 because the low-PPFD arms all spend substantial time under blue-shifted spectra, but Z2/Z5 still spend 8 h/day there while Z1/Z4 do so mainly in the early low-PPFD days.
- **Z2 transient ΦPSII dip at slot boundaries.** Each 50 → 130 step at the 4-h boundary causes a >33 % transient ΦPSII dip lasting 10–30 min while Rubisco and stomatal conductance catch up — slightly larger than the previous (60 → 126) design's dip given the steeper step. With 4-h slots, the dip is still ≤ 15 % of slot time and shouldn't dominate daily assimilation, but it is a real cost of the 3-step discretization. Finer slot counts (6 × 2 h or 12 × 1 h, same mean PPFD and energy) are the next refinement if the within-day pair is promising.
- **Z2/Z5 symmetric-vs-asymmetric phase split.** Z2 keeps equal edge slots (50, 130, 50), while Z5 shifts photons later (40, 60, 130). Read the pair together: if Z2 wins, symmetric ramping around the photoperiod midpoint matters; if Z5 wins or matches, avoiding high-PPFD dawn may be the dominant within-day lever.
- **Daily-DLI mismatch within the trial.** Z1 varies ~1.65 → ~5.35 mol m⁻² d⁻¹, Z4 varies ~1.73 → ~4.10, while Z2/Z5/Z3 are constant near ~3.31–3.37. Report cumulative-area-day integrals and end-of-trial area, not single-day snapshots; include 14-day mean PPFD and schedule family as covariates when interpreting cross-arm contrasts.
- **Z3 flat low-DLI interpretability.** Z3 is not a production recommendation; it is the shape-null comparator. If Z3 performs as well as the temporally redistributed arms, the result says ~80 % DLI is sufficient under these conditions, not that the schedule shape is valuable.
- **Z4 70 % threshold interpretability.** Z4 is not matched to a flat 70 % DLI control, so if it fails we cannot separate too-low cumulative DLI from ramp-shape cost. If it succeeds, the practical result is strong: the chamber can likely cut another ~10 % energy beyond the 80 % target without losing final projected area.
- **Z5 asymmetric phase risk.** Z5 shifts photons late in the photoperiod. That may reduce dawn NPQ overshoot, but it may also leave too little early carbon assimilation and starch priming. Compare Z5 to Z2 on both final area and cumulative area-days, not endpoint alone.

## See also

- [`../P0.1/README.md`](../P0.1/README.md) — power characterization sweep; source of `P(PPFD) = 9.71 + 0.164·PPFD^1.19` and the safe-min spectrum-collapse precedent.
- [`../P0.1/analyze_power.py`](../P0.1/analyze_power.py) — power-law fit reproduction; reuse for the post-deploy energy verification.

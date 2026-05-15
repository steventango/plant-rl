# Experiment E18 / Phase P1 — Across-day and within-day energy-saving PPFD schedules

## Overview

Three-arm 14-day comparison testing whether *temporal redistribution* of a constrained daily light integral can deliver **≥ the camera-derived plant area** of a constant 105 PPFD white-balanced control policy while **consuming ≤ 80 % of its lights-on energy**. Two treatments isolate two different temporal levers at matched 14-day mean PPFD and matched cumulative energy:

- **Z1 — across-day lever**: smooth power law in plant age, `PPFD(t) = 0.7397 · DAS_sowing(t)^1.6059`, ramps from 40 µmol m⁻² s⁻¹ on agent day 0 to 130 on day 13. Within-day intensity constant.
- **Z2 — within-day lever**: symmetric daily parabola, three 4-h slots of `(60, 126, 60)` PPFD. Across-day intensity constant.
- **Z3 — control**: constant 105 PPFD, no redistribution.

The biological bet for Z1 is that during the early small-canopy days most incident PPFD misses the leaves anyway (Beer–Lambert with LAI ≪ 1), so dropping PPFD then sacrifices little growth, and the low-PPFD spectrum collapse (blue + cool_white only) actually *favors* the 2D rosette-area metric via cry1-activated flat rosettes and miR156/SPL juvenile-leaf shape. The biological bet for Z2 is the Watanabe 2023 finding that a parabolic within-day profile suppresses the morning NPQ overshoot and increases daily integral of net CO₂ assimilation at the *same* DLI as a square wave.

## Experimental design

| Zone | Policy | Config | Predicted daily energy | Across-day shape | Within-day shape |
|---|---|---|---|---|---|
| **Z1 (zone01)** | Power-law ramp `PPFD = 0.7397·DAS^1.6059` | `PowerLawRamp1.json` | 494 Wh/day avg | super-linear in DAS | constant |
| **Z2 (zone02)** | Daily symmetric parabola `(60, 126, 60) PPFD × 4 h` | `Parabolic2.json` | 495 Wh/day | constant | low → high → low |
| **Z3 (zone03)** | Constant 105 PPFD | `Constant3.json` | 617 Wh/day | constant | constant |

All three zones share identical wrapper settings (`enforce_night = true`, `flash_photography = true`, `timezone = "Etc/GMT-9"`, `total_steps = 40320`) so chamber-side timing artifacts cancel in cross-zone comparisons. Z1 and Z3 use `action_timestep = 720` (one new PPFD scalar per 12-h photoperiod); Z2 uses `action_timestep = 240` (three slots per photoperiod).

**Hypothesis.** At end of trial (DAS 25):
- **H1 (across-day):** Z1 final rosette area ≥ Z3 final rosette area, while Z1's 14-day energy ≤ 80 % of Z3's.
- **H2 (within-day):** Z2 final rosette area ≥ Z3 final rosette area, while Z2's 14-day energy ≤ 80 % of Z3's.
- **H3 (lever comparison):** Z1 and Z2 share the same 14-day cumulative energy (6 921 / 6 931 Wh, both ≈ 80 % of Z3) and the same 14-day mean PPFD (≈ 82 µmol m⁻² s⁻¹), so the Z1↔Z2 contrast isolates the effect of *where the redistribution happens* (across-day vs within-day).

## Plant cohort and calendar

*Arabidopsis thaliana* (Col-0 assumed). Trial 17, sterilized seeds plated onto a 6:6:1 soil/peat/perlite mix in 24-cell trays. Pre-transplant (DAS 0–7) under unspecified PPFD in a separate germination chamber. Post-transplant (DAS 7–12) constant 105 PPFD incubation on the same 18:00→06:00 night-shifted photoperiod the agent will use.

| Event | Date | DAS (sowing = day 0) | DA_sterilization | DAT |
|---|---|---|---|---|
| Sterilize | 3/16/2026 | –3 | 0 | –10 |
| **Plate seeds (sowing, DAS 0)** | 3/19/2026 | 0 | 3 | –7 |
| Transplant + 1 L water | 3/26/2026 | 7 | 10 | 0 |
| Remove domes | 3/29/2026 | 10 | 13 | 3 |
| Mist 50 mL | 3/30/2026 | 11 | 14 | 4 |
| 750 mL water | 3/31/2026 | 12 | 15 | 5 |
| **Agent start (18:00 local, twilight off)** | 3/31/2026 18:00 | **12** | 15 | 5 |
| Agent day 0 photoperiod | 3/31 18:00 → 4/1 06:00 | 12 | 15 | 5 |
| Agent day 13 final photoperiod | 4/13 18:00 → 4/14 06:00 | 25 | 28 | 18 |
| Trial harvest | 4/14 06:00 | 26 | 29 | 19 |

Day-counting conventions in use:
- **DAS_sowing** (literature, Carvalho): plating = day 0. Translate from spreadsheet: `DAS_sowing = DA_sterilization − 3`.
- **DA_sterilization** (user's spreadsheet): sterilization = day 0.
- **DAT** (env code, `WallStatsActionTraceEmbeddingPlantGrowthChamber:108`): transplant = day 0.

Photoperiod is **night-shifted**: lights on 18:00, off 06:00 local. Twilight ramps are disabled before deploy, so the photoperiod is a hard 12 h square wave. The same schedule applied during the 5-d incubation period.

## Schedule

Smooth power law in plant age, anchored at PPFD(DAS 12) = 40 and PPFD(DAS 25) = 130:

$$\text{PPFD}(t) = 0.7397 \cdot \text{DAS}_\text{sowing}(t)^{1.6059} \quad [\mu\text{mol m}^{-2}\text{s}^{-1}]$$

The exponent 1.6059 makes the ramp super-linear in plant age — day-to-day step grows from ~5.5 PPFD early to ~8.3 PPFD late, mirroring the super-linear leaf-area expansion that drives canopy interception. One closed-form, monotonic, no step discontinuities.

| Agent day | DAS | PPFD | s = PPFD / 105 | Active LED channels |
|---|---|---|---|---|
| 0 | 12 | 40.00 | 0.38095 | blue + cool_white |
| 1 | 13 | 45.49 | 0.43321 | blue + cool_white |
| 2 | 14 | 51.24 | 0.48795 | blue + cool_white |
| 3 | 15 | 57.24 | 0.54512 | blue + cool_white |
| 4 | 16 | 63.49 | 0.60465 | blue + cool_white |
| 5 | 17 | 69.98 | 0.66648 | blue + cool_white + warm_white |
| 6 | 18 | 76.71 | 0.73055 | blue + cool_white + warm_white |
| 7 | 19 | 83.67 | 0.79681 | blue + cool_white + warm_white |
| 8 | 20 | 90.85 | 0.86523 | full balanced (red activates at PPFD ≥ 85.4) |
| 9 | 21 | 98.25 | 0.93574 | full balanced |
| 10 | 22 | 105.87 | 1.00833 | full balanced — crosses incubation level |
| 11 | 23 | 113.71 | 1.08293 | full balanced |
| 12 | 24 | 121.75 | 1.15954 | full balanced |
| 13 | 25 | 130.00 | 1.23810 | full balanced (cool_white drive = 0.945 / safe_max 90 PPFD) |

Mean PPFD over 14 d: **81.8 µmol m⁻² s⁻¹**. Mean DLI: **3.54 mol m⁻² d⁻¹** (vs. 4.54 for constant 105).

## Z2 schedule — daily symmetric parabola

Z2 explores the orthogonal lever: within-day redistribution. The same 12 h photoperiod, the same 80 % cumulative-energy ceiling, and (deliberately) the *same 14-day mean PPFD* as Z1, but the 12 h are split into three 4-h slots forming a symmetric parabola — low intensity at dawn and dusk, peak at the photoperiod midpoint.

**Closed-form derivation.** Three slot PPFDs `(a, b, a)`, each held for 4 h:

- Energy target: `2·P(a) + P(b) = 3 · 41.12 W = 123.5 W` (= 80 % of Z3's 51.4 W mean).
- Mean-PPFD matched to Z1: `(2a + b) / 3 = 82.0 → 2a + b = 246`.

Numerical search over the super-linear `P(PPFD) = 9.71 + 0.164·PPFD^1.19` curve yields **a = 60, b = 126** as the cleanest integer pair satisfying both constraints (energy +0.3 W vs target; mean exact). Peak-to-edge ratio 2.1×.

| Slot | Chamber MDT | Wrapper-local | PPFD | s = PPFD/105 | P (W) | Daily E (Wh, 4 h) | Active channels |
|---|---|---|---|---|---|---|---|
| **Flash** (1 min) | 17:59 | 08:59 | 105 (BALANCED_ACTION_105) | — | 51.4 | 0.9 | full balanced |
| 1 (4 h) | 18:00 – 22:00 | 09:00 – 13:00 | 60 | 0.57143 | 31.14 | 125 | blue + cool_white |
| 2 (4 h) | 22:00 – 02:00 | 13:00 – 17:00 | 126 | 1.20000 | 61.44 | 246 | full balanced (cw drive 0.910) |
| 3 (4 h) | 02:00 – 06:00 | 17:00 – 21:00 | 60 | 0.57143 | 31.14 | 125 | blue + cool_white |
| Night (12 h) | 06:00 – 17:59 | 21:00 – 08:58 | 0 | — | 0 (7.21 baseline) | 0 | none |

**Daily lights-on energy: 495 Wh → 80.3 % of Z3's 617 Wh.** 14-day cumulative: **6 931 Wh** vs. Z3's 8 636 Wh. Daily mean PPFD 82.0 µmol m⁻² s⁻¹ → DLI 3.54 mol m⁻² d⁻¹ — identical to Z1's 14-day mean.

**Why symmetric edges (60 at both ends).** Three reasons: (1) it's the shape Watanabe 2023 actually tested — symmetric peak-at-midday parabola mimicking the natural diurnal solar curve; (2) symmetric Z2 centers the within-day light distribution on the photoperiod midpoint, keeping the *first temporal moment* of light delivery the same as Z1 (whose within-day shape is constant), so the Z1↔Z2 contrast is purely about shape, not phase; (3) the two asymmetric arguments — "low-early / high-late" tracking ΦPSII decline through the day vs "high-early / low-late" priming the starch reserves for the night — directly contradict each other in the literature. Pre-baking a sign we haven't verified would risk a false negative. Asymmetric variants are the natural P2 follow-up if symmetric Z2 wins.

**Mechanism we are testing (Watanabe 2023; see literature section).** A square-wave dawn slams the photosynthetic apparatus from full dark to full target irradiance instantaneously while Calvin–Benson enzymes are still inactive and stomata are closed. The plant dumps the absorbed excitation as heat via a rapid PsbS-mediated NPQ overshoot — wasted photons. A parabolic ramp lets enzyme activation and stomatal conductance rise in sync with photon flux, suppressing the morning NPQ overshoot and improving daily integral of net CO₂ assimilation at the same DLI.

**Discretization.** A true parabola is smooth; we approximate with three 4-h step changes. The 60 → 126 transition causes a transient ΦPSII dip (~33 %, 10–30 min) while Rubisco and stomatal conductance catch up; with 4-h slots, the dip is ≤ 15 % of slot time and is dominated by the 3.5 h of steady-state operation that follows. Finer discretization (e.g. 6 × 2 h or 12 × 1 h) at the same mean PPFD and energy is a follow-up refinement.

**SequenceAgent details.** `action_timestep: 240` polls the agent at wrapper-local 09:00, 13:00, 17:00 (chamber 18:00, 22:00, 02:00). The 21:00 boundary lands in night, so no fourth poll. The next morning's 09:00 poll fires via both `should_poll` (flash mode: hour=9, minute=0) and `time_since_last_action ≥ 240` (16 h elapsed). The `actions` list repeats `[0.57143, 1.20000, 0.57143]` 28 times (84 entries) so the schedule keeps cycling cleanly through the 4-week `total_steps` buffer; if the trial extends past 14 days, Z2 continues delivering its daily parabola instead of clamping to a fixed value.

## Energy budget

Lights-on plug power follows the pooled fit from [E18/P0.1](../P0.1/README.md):

$$P(\text{PPFD}) = 9.71 + 0.164 \cdot \text{PPFD}^{1.19} \;\text{W} \qquad P_\text{lights-off,baseline} = 7.21\;\text{W}$$

| | PPFD | Lights-on P | Daily energy (12 h) | 14-day cumulative |
|---|---|---|---|---|
| Z3 — constant 105 baseline | 105 | 51.4 W | **617 Wh/d** | **8 636 Wh** |
| Z1 — across-day power-law ramp | mean 81.8 | mean 41.2 W | mean **494 Wh/d** | **6 921 Wh** |
| Z2 — within-day parabola (60, 126, 60) | mean 82.0 | mean 41.2 W | **495 Wh/d** | **6 931 Wh** |
| 80 % target | — | 41.1 W | 494 Wh/d | 6 909 Wh |

**Z1 / Z3 = 80.15 %; Z2 / Z3 = 80.26 % → both treatment arms ≈ 19.8 % savings.** Z1 and Z2 differ in cumulative energy by only 10 Wh (~0.1 %) — well below the meter-calibration floor.

Per-day predicted Wh (Z1 vs Z3):

| Day | Z1 P (W) | Z1 Wh | Z3 Wh |
|---|---|---|---|
| 0 | 22.91 | 275 | 617 |
| 1 | 25.09 | 301 | 617 |
| 2 | 27.42 | 329 | 617 |
| 3 | 29.92 | 359 | 617 |
| 4 | 32.57 | 391 | 617 |
| 5 | 35.37 | 424 | 617 |
| 6 | 38.33 | 460 | 617 |
| 7 | 41.44 | 497 | 617 |
| 8 | 44.70 | 536 | 617 |
| 9 | 48.12 | 577 | 617 |
| 10 | 51.68 | 620 | 617 |
| 11 | 55.40 | 665 | 617 |
| 12 | 59.27 | 711 | 617 |
| 13 | 63.28 | 759 | 617 |

Z1 crosses the Z3 daily-energy line on day 10 (PPFD ≈ 106) — after that point Z1 is *spending more* than Z3 per day, but the late-stage canopy is large enough that the marginal photons actually contribute productively.

## Energy detectability vs. sensor noise

Smart-plug telemetry per E18/P0.1:
- Within-zone fit RMSE: 0.28 W per 5-min sample.
- Kasa KP125M factory floor: ±3 % (BL0937/HLW8012-class energy chip).
- Zone-to-zone systematic offset: ~2 V on voltage → ~2–3 W at peak.

| Source | Magnitude | S / N |
|---|---|---|
| Signal — 14-day cumulative energy gap (8 636 − 6 921) | **1 715 Wh** | — |
| Systematic per-zone meter cal (±3 % on 8 636) | ±259 Wh | **6.6 ×** |
| Random within-day power noise (`√14 · 0.3`) | ±1.1 Wh | **1 500 ×** |

The 80 % spec is comfortably above sensor noise — ~7× the systematic factory floor. Comparing Z1 to its own predicted curve (rather than to Z3) cancels the per-zone systematic; daily within-zone energy *ratios* are limited only by the random floor (~1 % of daily Wh).

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

*A. thaliana* Col-0 under square-wave 150 PPFD, parabolic 65–190 PPFD, and fluctuating 60–360 PPFD profiles (initial 125 PPFD continuous). **Parabolic within-day profile gave highest biomass and growth rate** — the within-day shape modulates light-use efficiency at fixed DLI. Out of scope here (we keep a square photoperiod) but the empirical basis for the Z2 within-day parabolic policy in the follow-up plan.

### Lin et al. 2023 — Sci. Rep., [doi:10.1038/s41598-023-36997-2](https://doi.org/10.1038/s41598-023-36997-2)

Hydroponic lettuce: best growth from DLI staging — slow stage 14.4 mol m⁻² d⁻¹ → rapid stage 17.2 (~20 % step). Different species, but precedent for monotonically increasing DLI schedules in production CEA.

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
- Carbon-starvation risk on Z1 days 0–4 (DLI 1.7 mol m⁻² d⁻¹) is real but mild — plants survive (40 PPFD is 2–4× above compensation point), they just allocate more daytime photosynthate to starch reserves to make it through the 12 h dark period. By Z1 day 5 PPFD ≥ 70 and DLI > 3 mol m⁻² d⁻¹, restoring carbon-positive growth.

From **Gemini Deep Research 2** (independent Z1 expansion):
- **miR156/SPL juvenile-phase prolongation** under low light upregulates miR156/miR157, repressing the SPL transcription factors that drive juvenile → adult transition. Phenotype: rounder, less-serrated leaves that project a more continuous overhead canopy — *directly amplifies the plant-area metric*.
- **cry1-driven flat rosette under blue-enriched spectrum**: when the safe_min gating collapses to blue + cool_white (Z1 days 0–4, Z2 edge slots 8 h per day), cry1 stays heavily activated despite the low absolute PPFD; cry1 suppresses petiole hyponasty, keeping the rosette flat against the substrate. Flat rosette = max overhead area — but see the CRY1/HY5 PLA-confound note in the risks section.
- **Dynamic-spectrum CV bias risk** was the original argument for the flash-photography wrapper mode (now implemented). The 1-min flash at 08:59 standardizes the *imaging* spectrum across zones; the underlying *growth* spectrum still varies, which is what the CRY1/HY5 risk speaks to.

From **Gemini Deep Research 3** (Z2 validation):
- Numerical confirmation of the Z2 parabola design (`a = 60, b = 126`, ~80 % energy ratio, matched-to-Z1 mean PPFD).
- The PsbS / xanthophyll-cycle mechanism for why the parabolic shape beats square-wave — square-wave dawn forces an unavoidable morning NPQ overshoot before Calvin–Benson enzymes activate; a parabolic ramp lets enzyme activation and stomatal conductance rise in sync with photon flux.
- **CRY1 / HY5 PLA confound (important)**: 8 of 12 h of the Z2 photoperiod runs under the blue + cool_white spectrum, hyper-activating CRY1 → HY5 → repressed PIFs → compact, prostrate rosette. A zenith camera captures more of a flat rosette's true leaf area (PLA ∝ cos(leaf_angle)). Z2 may register higher PLA than Z3 partly via morphology rather than via greater biomass accumulation — see the corresponding risks-section entry.
- **Transient ΦPSII dip at slot boundaries** (~33 %, 10–30 min) is a small cost (≤ 15 % of slot time) of the 3-step discretization; finer slot counts (6 × 2 h, 12 × 1 h) are the natural follow-up.

## Biological mechanisms favoring plant area at reduced PPFD

These three mechanisms are drawn from the Gemini Deep Research and Gemini Feedback briefs summarized in the "Gemini Deep Research findings" subsection above. Why a ramp-down-then-up schedule should match constant 105 on *plant area* despite lower mean DLI:

1. **Canopy interception inefficiency early.** With LAI ≪ 1 at DAS 12, most of a constant 105 PPFD irradiance hits substrate, not leaf. Reducing PPFD on days 0–4 removes wasted photons preferentially — per-leaf incident PPFD drops much less than the headline number suggests. Beer–Lambert dominates the cost-benefit math at small canopies.
2. **Juvenile-leaf shape advantage (miR156/SPL).** Sustained low light upregulates miR156/miR157, repressing the SPL transcription factors that ordinarily push the juvenile → adult transition. Phenotypic consequence: leaves stay rounder and less serrated longer. Round, unserrated leaves *project* a more continuous overhead canopy — the 2D measurement is higher per unit dry mass.
3. **cry1-driven flat rosette under blue-enriched spectrum.** Days 0–4 of the schedule sit below the calibration safe_min for red, warm_white, and orange_red; only blue + cool_white channels actuate. The high relative blue fraction keeps cry1 heavily activated, which suppresses petiole hyponasty and keeps the rosette pressed flat against the substrate rather than tilting toward overhead light. Flat rosette → maximum orthogonal overhead area.

The countervailing risk — transient nighttime carbon starvation on days 0–4 — is mitigated by the schedule's rapid super-linear ramp: PPFD ≥ 70 (DLI > 3 mol m⁻² d⁻¹) by day 5, well above any standard "minimum" growth-light recommendation for Arabidopsis. Net of these effects, the prediction is that Z1 plant area at DAS 25 will be statistically equivalent to or greater than Z3.

## Spectrum collapse note

The chamber's [`Calibration.get_calibrated_action`](../../../src/environments/PlantGrowthChamber/Calibration.py#L48) gates each channel by its per-channel `safe_minimum` (see [`configs/calibration.json`](../../../src/environments/PlantGrowthChamber/configs/calibration.json)). Below the threshold the channel is zeroed, and remaining active channels are rescaled to hit the requested PPFD. For `BALANCED_ACTION_105` the activation thresholds are:

| Channel | Share in 105 | safe_min (PPFD) | s threshold | Activates at PPFD ≥ |
|---|---|---|---|---|
| cool_white | 71.53 | 5.0 | 0.0699 | 7.34 |
| blue | 19.50 | 5.0 | 0.2564 | 26.92 |
| warm_white | 7.82 | 5.0 | 0.6394 | 67.13 |
| red | 6.15 | 5.0 | 0.8130 | 85.37 |

For this schedule:
- **Days 0–4** (PPFD 40–63): blue + cool_white only. Spectrum is daylight-white-ish but missing warm_white and red contributions.
- **Days 5–7** (PPFD 70–84): warm_white joins; spectrum gains broader-yellow contributions.
- **Days 8–13** (PPFD 91–130): red activates; full balanced-105 spectrum throughout. Cool_white drive reaches 0.945 at day 13 (88.6 PPFD output, below safe_max 90 — no clipping).

This is the same safe_min-collapse tradeoff explicitly accepted in [E18/P0.1](../P0.1/README.md#safe-minimum-note-spectrum-at-low-levels): we accept partial-spectrum operation at low PPFDs in order to reach low absolute intensities at all. The biological consequence (compact, juvenile-extended, flat rosette) is in our favor for the plant-area metric.

## Deployment

```bash
# Z1: across-day power-law ramp
python src/main_real.py -e "experiments/online/E18/P1/PowerLawRamp1.json" -i 0 --deploy

# Z2: within-day parabolic energy saver
python src/main_real.py -e "experiments/online/E18/P1/Parabolic2.json" -i 0 --deploy

# Z3: constant 105 control
python src/main_real.py -e "experiments/online/E18/P1/Constant3.json" -i 0 --deploy
```

Agent-name conventions: `SequencePowerLawRamp1` and `SequenceParabolic2` both resolve to `SequenceAgent` via `algorithms/registry.py`'s `startswith("Sequence")` rule; `Constant3` resolves to `ConstantAgent` via `startswith("Constant")`. The descriptive suffixes are for legibility — drop them and the registry would still wire the runs identically.

All three configs share:
- `timezone: "Etc/GMT-9"` — the **night-shift trick**. The chamber's wall clock is Edmonton-local (MDT, UTC-6 in our trial window), but `PlantGrowthChamberAsyncAgentWrapper` has hard-coded boundaries for night / dawn / dusk / `should_poll` keyed off wrapper-local hour 9 to 21 (originally meant for a 9:00 → 21:00 daytime photoperiod). Setting `timezone = Etc/GMT-9` (UTC+9) shifts wrapper-local by +15 h vs. Edmonton MDT, so chamber-wall-clock 18:00 MDT = UTC 00:00 = wrapper-local 09:00 and chamber 06:00 MDT = wrapper-local 21:00. The wrapper sees the night-shifted 18:00 → 06:00 photoperiod as its native 09:00 → 21:00 window, no wrapper code changes needed. (Same trick used by `experiments/online/E17/P0`.)
- `action_timestep` — wrapper's `time_since_last_action ≥ action_timestep` check (in minutes). **Z1 and Z3 use 720** (the full 12 h photoperiod ⇒ one new PPFD scalar per daytime block). **Z2 uses 240** (12 h / 3 = 4 h per slot ⇒ three polls per day at wrapper-local 09:00, 13:00, 17:00). Differs from the old E14–E17 default of 660 min, which was the 11 h pure-daytime portion of a 12 h photoperiod with 30 min twilights at each end — with twilight off the pure-daytime is the full 12 h.
- `enforce_night: true` — wrapper zeros the action during wrapper-local night (chamber 06:00 → 18:00); the schedule scalar applies only inside the chamber 18:00 → 06:00 daytime block.
- `total_steps: 40320` — **4 weeks** worth of 1-min env steps (28 × 1440). The experiment is planned to stop at 14 days but `total_steps` provides headroom so the trial doesn't terminate prematurely if we decide to extend. Z1's `SequenceAgent` clamps to its last entry (PPFD 130) past day 13 — energy comparison vs. Z3 should still be reported over the matching 14-day window. Z2's `actions` list is intentionally pre-tiled to 84 entries (28 days × 3 polls), so it keeps cycling the parabola through the buffer rather than clamping.
- `episode_cutoff: -1` — episode ends only when `total_steps` is reached.

**Photoperiod & flash photography.** Both configs set `flash_photography: true`. This activates a new branch in `PlantGrowthChamberAsyncAgentWrapper.maybe_enforce_action` that overrides the wrapper's default 11 h-daytime-with-twilight-ramps behavior:

| Wrapper-local | Chamber wall-clock (MDT) | Behavior |
|---|---|---|
| 08:59 | 17:59 | **1-min flash** at `BALANCED_ACTION_105` (105 PPFD) for daily camera capture |
| 09:00 – 20:59 | 18:00 – 05:59 | **12 h daytime** — agent's scheduled PPFD applies |
| 21:00 – 08:58 | 06:00 – 17:58 | Night — wrapper zeros the action |

The flash is the only deviation from a hard square-wave 12 h photoperiod. Because it fires at the same wrapper-local time every day under the same fixed `BALANCED_ACTION_105` spectrum, the daily flash frame is the canonical input for the CV plant-area pipeline — resolving the dynamic-spectrum CV bias that would otherwise confound Z1 (multi-day spectrum drift), Z2 (within-day spectrum cycling), and Z3 (constant spectrum). The `flash_photography` flag is a plain wrapper parameter, so any future fixed-schedule deploy can opt in with one line in its config JSON.

## Verification

### Pre-deploy

1. **Energy check** (already done):
   ```text
   14-day cumulative   Z1: 6 921 Wh   Z2: 6 931 Wh   Z3: 8 636 Wh
   Ratio vs Z3         Z1: 80.15 %   Z2: 80.26 %
   ```
   Reproduces with the power-law fit from [`../P0.1/analyze_power.py`](../P0.1/analyze_power.py).

2. **Action-vector sanity** (already done — see `Calibration.get_calibrated_action` output above):
   - Z1 day 0 (s=0.38095): blue + cool_white only ✓
   - Z1 day 5 (s=0.66648): warm_white joins ✓
   - Z1 day 8 (s=0.86523): red joins ✓
   - Z1 day 13 (s=1.2381): cool_white drive 0.945 (below safe_max 1.0) ✓
   - Z2 edge slot (s=0.57143): blue + cool_white only ✓
   - Z2 peak slot (s=1.20000): full balanced; cool_white drive 0.910 (below safe_max) ✓

3. **Registry check** (already done): `algorithms/registry.py` resolves `SequencePowerLawRamp1` → `SequenceAgent`, `SequenceParabolic2` → `SequenceAgent`, `Constant3` → `ConstantAgent` via the prefix-matching rules (`startswith("Sequence")`, `startswith("Constant")`).

4. **Smoke test.** Deploy each config with the mock chamber / dry-run flag in `main_real.py`; confirm `SequenceAgent` advances at the expected cadence (once per simulated daytime for Z1, three times per simulated daytime for Z2) and that the wrapper zeros the action during chamber 06:00 → 18:00 night.

CV-pipeline robustness against the dynamic Z1/Z2 spectra is handled by the `flash_photography` wrapper mode (08:59 daily flash under fixed `BALANCED_ACTION_105`). Use those frames as the canonical area time-series; in-daytime frames captured under the schedule's drifting/cycling spectrum should be treated as preliminary.

### Post-deploy

1. **Rsync raw CSVs** to local analysis storage:
   ```bash
   rsync -azP --include='*.csv' --include='*/' --exclude='*' archcraft:/data/plant-rl/online/E18 /data/plant-rl/online/
   ```

2. **Daily energy check.** Integrate measured `power` over the lights-on minutes per zone; compare to the predicted Wh column above ± 3 % systematic. Diagnostic signatures: Z1's daily energy curve should *climb* (275 → 759 Wh), Z2's should be *flat* at ~495 Wh/day, Z3's flat at ~617 Wh/day. Cumulative Z1 vs Z3 and Z2 vs Z3 should each diverge by ~1 715 Wh over 14 days; cumulative Z1 vs Z2 should differ by only ~10 Wh (well below meter noise). Within-zone energy *ratios* (each day's actual vs. predicted) are limited only by the ~1 % random floor and should align tightly.

3. **Plant-area check.** From the camera-derived rosette-area time series (from the daily 08:59 flash frames):
   - Final rosette area per zone at DAS 25.
   - Cumulative rosette-area-days integral (a more stable metric than the endpoint alone).
   - Area-vs-day curves with shaded confidence intervals.

   **Win conditions:**
   - **H1 (across-day):** Z1 final area ≥ Z3 final area at ≤ 80 % Z3 energy.
   - **H2 (within-day):** Z2 final area ≥ Z3 final area at ≤ 80 % Z3 energy.
   - **H3 (lever):** Z1 vs Z2 ordering at matched 14-day energy + DLI tells us which redistribution lever — across-day or within-day — wins.

## Risks and open items

- **Species/cultivar assumption.** Plan assumes Col-0. If trays contain a different ecotype, recompute against that ecotype's tolerance where literature exists. Carvalho PI50 is Col-0-specific.
- **Down-shift from acclimated 105 → carbon-starvation risk on days 0–4.** Plants have had only 5 d at 105 PPFD before agent start (not the full 15). Day 0 is a ~60 % drop to 40 PPFD (DLI 1.73 mol m⁻² d⁻¹). Plants survive (above compensation point) but transient daytime growth restriction is plausible. If de-risking is needed before the paired comparison, raising the floor to ~50 PPFD costs ~3 percentage points of energy savings (ratio rises from 80 % to ~83 %) and is a defensible variant.
- **Pre-transplant light history unknown.** DAS 0–7 in a separate germination chamber under unspecified PPFD. The schedule's safety margins are robust to this within the photoinhibition envelope, but absolute biomass and chloroplast density at day 0 could differ from baseline expectations.
- **Fixed schedule — no adaptation.** This is a deterministic open-loop policy. If plants stall or surge unexpectedly the schedule doesn't react. Treat this as the baseline against which a future closed-loop / learning agent is evaluated.
- **Dynamic-spectrum CV bias** — now addressed by the `_FlashPhotography` wrapper variant. The 1-min full-intensity flash at chamber 18:29 (wrapper-local 09:29) each day gives the CV pipeline a single image per zone under a known fixed `BALANCED_ACTION_105` spectrum. Use those frames as the canonical area time-series; any in-daytime-photoperiod frames captured under the schedule's drifting spectrum should be treated as preliminary.
- **Twilight off.** Energy budget assumes a 12-h square wave. If twilight is left on by mistake, both zones get the same extra daily energy and the *ratio* is preserved; only absolute Wh shifts.
- **Within-day shape (Z2).** Z2's `(60, 126, 60)` parabola is at the moderate end of the Watanabe / Gemini Deep Research 3 design space — peak-to-edge ratio 2.1× vs Watanabe's 2.9× (190 / 65). The shallower ratio is forced by the cool_white safe_max ceiling (90 PPFD ⇒ total 130). We are at the conservative end of "parabolic" relative to the literature precedent.
- **Z2 CRY1 / HY5 PLA confound (per Gemini Deep Research 3).** Z2's edge slots (8 h of 12 h daytime, 66 % of the photoperiod) run blue + cool_white only — a high blue : red environment that hyper-activates CRY1, stabilizes HY5, represses PIFs, and forces a flatter, prostrate rosette (short petioles, leaves pressed against substrate). A zenith camera captures *more* of a flat rosette's leaf area than a vertically-angled one (PLA ∝ cos(leaf_angle)). **Z2 may register higher PLA than Z3 partly because its rosette is morphologically flatter, not because it accumulated more biomass.** The 08:59 flash standardizes the *imaging* spectrum but not the *growth* spectrum that shaped the morphology. A clean conclusion will need either a destructive-biomass spot check at end-of-trial or a leaf-angle correction in the CV pipeline. The Z1 vs Z2 PLA contrast is less affected because Z1 spends only days 0–4 under heavily blue-shifted spectrum while Z2 spends 8 h every day there.
- **Z2 transient ΦPSII dip at slot boundaries.** Each 60 → 126 step at the 4-h boundary causes a ~33 % transient ΦPSII dip lasting 10–30 min while Rubisco and stomatal conductance catch up. With 4-h slots, the dip is ≤ 15 % of slot time and shouldn't dominate daily assimilation, but it is a real cost of the 3-step discretization. Finer slot counts (6 × 2 h or 12 × 1 h, same mean PPFD and energy) are the natural P2 refinement.
- **Z2 symmetric-vs-asymmetric edges (follow-up split).** Edge slots are equal (60, 126, 60) for this round — see the "Why symmetric edges" subsection in the Z2 schedule. If symmetric Z2 beats Z3 on plant area, the natural P2 follow-up is to split the win between two asymmetric variants — `(40, 126, 80)` "low-early / high-late" tracking ΦPSII decline through the day, and `(80, 126, 40)` "high-early / low-late" tracking starch-priming. Both can be designed at the same 80 % energy + matched-mean-PPFD constraint by the same numerical search.
- **Z1 vs Z2 daily-DLI mismatch within the trial.** Z1's daily DLI varies 1.73 → 5.61 mol m⁻² d⁻¹; Z2's is constant at 3.54. The 14-day means match but day-by-day comparisons aren't apples-to-apples. Report cumulative-area-day integrals and end-of-trial area, not single-day snapshots.

## See also

- [`../P0.1/README.md`](../P0.1/README.md) — power characterization sweep; source of `P(PPFD) = 9.71 + 0.164·PPFD^1.19` and the safe-min spectrum-collapse precedent.
- [`../P0.1/analyze_power.py`](../P0.1/analyze_power.py) — power-law fit reproduction; reuse for the post-deploy energy verification.

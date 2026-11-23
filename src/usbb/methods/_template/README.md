# <Method name> (`<method_id>`)

Verdict <A-D>. Tier <v0/T1/T2>. Issue: to open.

<!-- Copy this file to src/usbb/methods/<method_id>/README.md and replace every <...>.
One to three lines per section, about 35 lines in all.
Give the source of every fact (working note, 16 Nov 2026; proposal §3.3 or §4.1; or a URL) and write "to verify" for anything not checked.
Fit mode, environment group and held-out status live in card.yaml; do not repeat them here. -->

## What it is
<What the method does and what it outputs, in one or two lines, with the paper or model card URL.>

## Weights
<Repository, full revision, file sha256s, licence and access (gated or not); "None" for methods without weights.>

## Inputs
<Modalities and units it accepts, value scale or patch geometry (µm per pixel, patch size), and the vocabulary our genes or markers must map to.>

## Pitfalls on our data
<What can go wrong on our datasets: pixel size, gene or marker coverage, same or adjacent sections, pretraining exposure.>

## Why the code looks like this
<Reasons for choices not visible in the code. A guide leaves this as "To write with the implementation.">

## Done when
<Three to five checkboxes for this method, each a check someone else can repeat (a test, a value, a file); the PR template covers the general ones.>

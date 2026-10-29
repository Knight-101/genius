# Genius v5.1 — separate Genius and Grit scores

This version extends `Genius_working_prototype.ipynb`. The original notebook is unchanged.

## Jupyter notebook

Open `Genius_v5_request_based.ipynb` in Jupyter or Google Colab and run all cells.
It contains the source, configuration and synthetic samples, so it can run on its own.
This repository includes exactly one notebook; the Python package below is the same code.
A clean first run builds the synthetic database and trains the models.

## Run the Python files

```sh
python -m pip install -r requirements.txt
python run.py --share
```

The public Gradio link is temporary and needs the Python process and computer to remain online.
Without `--share`, the app runs locally at `http://127.0.0.1:7861`.
A clean first run builds the 5,000-firm synthetic database. Later runs load the local cache.

## Try the new flow

1. Select **Kaveri Foods**, read the books, and get its credit standing.
2. Open **Borrow**. Enter, for example, ₹500,000 and 60 days, then select **Calculate my quote**.
3. See the requested amount, amount supported, annual rate, actual interest, full fees,
   cash received after upfront fees, total cost and repayment schedule.
4. Try an amount above the supported limit. The response identifies a **smaller alternative**;
   it never labels the original request approved.
5. Try **Medium term**, choose months, offer collateral and enter an estimated value.
   The collateral value is unverified and the illustration remains conditional.
6. Try **Sharma** or **Swift** to see cases that need review or fail policy checks.

Changing the request, company or saved lender policy clears the old quote.

## Scoring changes

- Financial indicators appear once. A monotone logistic model learns nonnegative risk
  coefficients from the available synthetic 12-month default labels. Fixed measure-rating
  thresholds remain assumptions; this release does not claim to learn every threshold.
- The five data-integrity measures gate score publication instead of adding credit points.
- Genius and Grit are kept separate. Genius learns financial risk coefficients only; the
  financial-plus-recovery model remains an exploratory validation comparison, never the
  deployed score. There is no invented blend weight between the two scores.
- Grit retains its existing demonstrated-recovery / capacity calculation. With no assessed
  setbacks it is a capacity estimate, explicitly labelled as such. Its weights are assumptions.
- A separate **assumed demo policy** permits up to 50 bps of margin concession for unique,
  recent, completed recovery evidence. It never changes Genius, calibrated PD or adds a size
  bonus. Financial capacity indicators (already in Genius) are excluded from this concession.
- Recovery strength is the mean of gap regained, sustained recovery, operating-funded recovery
  and management response. Concession = 50 bps × max(0,(strength−50)/50) × n/(n+2) ×
  2^(−age_months/24), capped at available margin. Require quality ≥85, an assessed setback
  within 24 months and no unresolved setback. Missing evidence means zero concession, not a penalty.
- `resilience_demo.py` constructs two operating histories using identical 45% two-month demand
  shocks but different recovery speed, cost flexibility, collections response and additional
  borrowing. Grit is calculated by the existing detector, not assigned. Raw histories are in
  `samples/synthetic_resilience_histories.csv`; these are operating signals, not balanced ledger exports.
- The Borrow demo compares identical current financial inputs and a ₹5 lakh / 60-day request:
  Genius 65 in both, Grit about 74 vs 54; rates about 14.82% vs 14.95% under the reference policy.
  Both support ₹5 lakh. This isolates an assumed policy effect, not incremental default prediction.
- The existing 5,000-firm cash-flow simulator already includes latent resilience, recovery speed,
  cost response and cash buffers. Its historical outcomes have not been rewritten to make Grit win.
- Train feature months are 11–47, with 12-month labels known by month 59. Selection feature
  months are 60–83, labels known by 95. Final test feature months are 96–107. The final test
  is reported after selection, never used to choose the model. These are zero-based months.
- The 0–100 display is a fixed monotone transformation of model log-odds anchored to the
  training distribution, followed by the original financial caps. Probabilities are calibrated
  separately. Data-quality and integrity gates still override publication.
- Details and Validation show learned coefficients and out-of-time results. Coefficient
  percentages are relative magnitudes, not causal importance or changes in default probability.

## Lending changes

- Borrow starts with user inputs and an explicit calculation action.
- Risk-based pricing, affordability, fees and the schedule share one complete lender-policy
  context: sector, size, tier, tenure, security, market regime and effective date.
- Medium-term capacity uses operating cash flow plus interest already deducted, less capex
  and drawings, then a configurable stress haircut. Existing interest/principal commitments,
  debt coverage, liabilities/net worth and an estimated collateral advance limit constrain it.
  All capex is deducted conservatively; maintenance/expansion separation needs better data.
- Short-term repayment checks use stressed collections, current cash above a fixed-cost reserve,
  operating commitments and existing debt payments. This is a cash-flow approximation, not
  a complete transaction-level forecast or a validated causal loan-performance model.
- Amount and term use the same quote rate as the repayment calculation. A larger request is
  constrained by post-loan affordability; no unvalidated loan-amount PD multiplier is invented.
- Annual facility fees are charged in full on the requested facility, rather than shown as a
  prorated share of a draw. Term-loan anniversary fees appear in the schedule and affordability.
  Shock-line reservation fees are included. Taxes and external charges need lender disclosure.
- Effective annual cost uses the actual illustrated cash flows. It is explicitly **not** a
  regulatory KFS APR. Interest and fees are separate, and upfront fees reduce cash received.
- Quotes are conditional illustrations. There are no live Zoho, GST, bank, bureau, verification,
  collateral-valuation, lender-approval, disbursement or contractual repricing integrations.

## Replacing the synthetic data

Production requires permissioned financial snapshots linked to actual loan outcomes, with
outcome observation dates and records from the time of each decision. Prepare the same
financial ratings and past-only recovery features, then refit and recalibrate using matured,
chronological training/validation/test partitions. This release supplies the training and
selection implementation, not access to real borrowers or a prevalidated lender model.
Independent firms, sectors, calibration, selection bias, default losses and new-loan affordability
must be evaluated before any lending use. Repeated firm-months are not independent borrowers.

## Tests

```sh
python -m pytest -q tests
```

`tests/test_v5_lending.py` covers request validation, smaller alternatives, context overrides,
full facility fees, amortisation, renewal fees, collateral, cash-flow constraints, missing recovery
evidence, data-quality gates, financial caps and removal of duplicate Grit rewards.

## UI styling

The original Inter typography, colours and Soft teal theme are retained, with focused improvements to summary-card sizing and responsive layout. The app defaults to light mode to match those light surfaces; native Gradio hover styles are used without custom colour overrides.

Shared notebook previews serve the installed Gradio frontend assets through the existing tunnel, avoiding an observed external-CDN component stylesheet preload failure. The original Soft theme and native hover colours are retained.

## Separate-score release checks

The financial-only cache can be migrated without refitting coefficients or rewriting outcomes.
Other obsolete caches rebuild. The synthetic default validation tables validate the financial
model only; they do not validate the new assumed recovery concession. Calibrate the policy
with real, dated loan outcomes before production use.

The latest verification passed all 46 tests (15 original, 25 lending regressions, 6 separate-resilience checks).

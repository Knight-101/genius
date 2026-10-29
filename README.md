# Genius

### Financial health, resilience and borrowing support for small businesses

Genius is a financial decision-support prototype proposed as an extension to **Zoho Books**, focused on Indian micro and small trading and distribution businesses. It turns accounting records into a view of financial standing, cash-flow pressure, business resilience and borrowing capacity.

Distributors often pay suppliers before collecting from customers. Genius helps an owner understand that working-capital gap, identify what needs attention and explore financing that fits the business's repayment capacity. For a partner lender, it provides financial indicators, recovery evidence and a review workflow.

**Academic proof of concept · NUS BMF5342 · Synthetic demonstration data**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Knight-101/genius/blob/main/Genius_v5_request_based.ipynb)

## What Genius does

| Feature | Purpose |
|---|---|
| Accounting-data ingestion | Reads supported Tally and Zoho-style CSV/Excel exports and ledger-by-account workbooks; checks accounting integrity and data completeness. |
| Genius Score | Summarises financial credit standing using financial indicators and coefficients learned from synthetic default outcomes. Data-quality gates and financial red flags constrain publication. |
| Grit Score | Describes resilience through demonstrated recovery and financial capacity. It stays separate from Genius; without assessed setbacks, it is a capacity estimate. |
| Peer comparisons | Compares a business with synthetic peers in its sector and MSME size band, widening the group when needed. |
| Cash-flow and working-capital tools | Provides a 13-week cash projection, overdue-payment insights, supplier-payment analysis and early-payment calculations. |
| Borrowing illustrations | Assesses a requested amount, product and duration using stressed cash flows, existing debt, lender policy and collateral where required. Shows interest, fees, net cash received and a repayment schedule. |
| Second Opinion | Uses a separately trained ML model to assess future financial stress. Material disagreements can trigger lender review and tighter limits. |
| Ask Genius and what-if analysis | Answers supported questions from the business's records and calculations, and re-scores selected financial improvements. Chat answers do not call an external LLM. |
| Lender controls | Provides portfolio views, review queues and editable, versioned pricing and limit assumptions. |

The proposed Zoho integration would place these insights alongside the owner's existing bookkeeping workflow. **This prototype reads uploaded exports; it does not connect to live Zoho accounts or lender systems.**

## Try the prototype

### Jupyter or Google Colab

Open [Genius_v5_request_based.ipynb](Genius_v5_request_based.ipynb), or use the Colab button above, and run all cells.

The single notebook includes the Python source, configuration and synthetic sample books. The first run builds a 5,000-firm synthetic database and trains the models, which may take several minutes. The final cell launches the Gradio interface and prints a temporary public link. That link requires the notebook runtime to remain running.

### Python

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:7861`. To create a temporary public preview, run `python run.py --share`.

### Suggested walkthrough

1. Select **Kaveri Foods**, press **Read the books**, confirm the details and get its credit standing.
2. Explore **Genius**, **Grit**, the peer comparison and the score drivers.
3. Open **Borrow**, enter **₹500,000** and **60 days**, then select **Calculate my quote**.
4. Inspect the supported amount, interest, fees, total cost and repayment schedule. Try a larger request or a secured medium-term loan.
5. Open **Demo: how resilience can affect credit pricing** for the constructed fast/slow recovery comparison.

Sharma Traders and Swift Freight provide other synthetic business conditions to explore. No real company records are supplied in the sample dataset.

## Resilience example

Two constructed operating histories experience identical two-month sales shocks of 45%, with different recovery speeds, cost flexibility, collection responses and reliance on additional borrowing. Grit is calculated from those histories.

Holding current financial underwriting inputs constant gives Genius **65** for both and Grit approximately **74 versus 54**. For a ₹5 lakh, 60-day reference request, the illustrative rates are **14.82% versus 14.95%**, with the same supported amount.

The difference comes from an **assumed demo lender policy**: recent, completed recovery evidence may earn a capped margin concession. It does not change Genius or calibrated default probability, and Grit's overlapping financial-capacity indicators do not earn an additional concession. This scenario demonstrates a policy mechanism; it is not evidence that Grit predicts real defaults.

## Method and boundaries

- Financial-model coefficients are learned from **synthetic** outcomes. Indicator-rating thresholds, Grit weights, cash-flow stress assumptions and lender-policy settings remain design assumptions.
- Model evaluation uses chronological training, validation and test periods. Results demonstrate behaviour in the simulated setting, not performance on real borrowers.
- Genius is a prototype assessment, not an official credit-bureau rating. Borrowing results are conditional illustrations, not lender approvals, loan offers or regulatory Key Facts Statements.
- The app has no live banking, GST, bureau, collateral-verification, approval or disbursement integrations. A deployed product would require verified records, appropriate consent, lender participation and validation on real outcomes.
- Effective annual cost is an illustrated cash-flow calculation; a lender must disclose applicable taxes, third-party costs and contractual terms.

## Repository

```text
genius/                         Financial analysis, scoring, resilience, lending and UI
config/                         Account keywords and import configuration
samples/                        Synthetic CSV and Excel demo records
tests/                          Accounting, scoring, recovery and lending checks
tools/                          Demo-data generation and notebook packaging
Genius_v5_request_based.ipynb    Single self-contained notebook
requirements.txt                Python dependencies
run.py                          Application launcher
```

Generated databases, runtime uploads and local caches are excluded from version control.

## Tests

```bash
python -m pytest -q tests
```

The suite covers accounting consistency, historical-data boundaries, score publication, financial caps, separate resilience evidence, request validation, affordability, fees, collateral and repayment calculations.

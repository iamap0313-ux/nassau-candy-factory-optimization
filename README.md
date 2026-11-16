# Nassau Candy: Factory Reassignment Decision Support

Unified Mentor internship project by Alpesh Patel.

## What this project does

Nassau Candy makes each product at one fixed factory. This project asks: **would making a product at a different factory ship it faster or earn more profit?**

It predicts shipping lead time for every product, factory, region and ship mode, simulates moving each product to every other factory, and ranks the moves by speed, profit and risk.

## Main findings

- Lead time is decided mostly by **ship mode** (Same Day about 0 days, Standard about 5 days), not by the factory. Ship mode alone explains about 66% of the variation.
- Moving a product to a closer factory changes speed very little. The real benefit is lower freight cost.
- The safest move to pilot is three chocolate bars from Lot's O' Nuts to Wicked Choccy's. Moving the big bars to the small factories is high risk because of capacity.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Runs the full analysis: cleaning, EDA, three models, clustering, simulation, KPIs |
| `nassau_core.py` | Shared logic: distance, features, simulation, scoring |
| `app.py` | Streamlit dashboard |
| `data/` | The raw dataset |
| `outputs/` | Charts, result tables and the trained model |
| `requirements.txt` | Python libraries needed |

## How to run

```
pip install -r requirements.txt
python pipeline.py
python -m streamlit run app.py
```

## Assumptions and limits

- The order and ship dates in the raw file were shifted by a fixed amount per year, so lead time was rebuilt by removing that shift.
- Customer location is approximated by state or province.
- Freight cost is assumed at $0.15 per unit per 1,000 km, because the data has no shipping-cost field. The dashboard lets you change it.
- No factory capacity data was available, so capacity risk is estimated from order counts.

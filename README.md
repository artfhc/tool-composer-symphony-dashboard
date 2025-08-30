
# Dashboard Starter

A simple Streamlit dashboard that automatically loads the most recent CSV from Backblaze B2 (via S3-compatible API) and provides basic analytics.

## How it works
- The daily cron job writes CSV files to B2 under a prefix like `composer/YYYY/MM/composer_YYYY-MM-DD.csv`.
- This app lists objects under that prefix, picks the most recent by LastModified or filename date, and loads it into pandas.

## Environment variables (B2 mode)
Set these in your hosting platform (Render, Sliplane, local `.env`, or shell).

- `B2_BUCKET`   (required)
- `B2_KEY_ID`   (required)
- `B2_APP_KEY`  (required)
- `B2_PREFIX`   default: `composer/`
- `B2_REGION`   default: `us-west-000`
- `B2_ENDPOINT` default: `https://s3.us-west-000.backblazeb2.com`

## Local development (no B2)
- Put a CSV on your machine and set `LOCAL_DEBUG_FILE=/path/to/file.csv`
- Run: `streamlit run dashboard_app.py`

## Deployment
1. Create a repo and add these files.
2. Set environment variables in your hosting platform.
3. Start the Streamlit app (e.g., Render: Web Service; Sliplane: container with a `start` command `streamlit run dashboard_app.py --server.port $PORT --server.address 0.0.0.0`).

## Requirements
See `requirements.txt`.

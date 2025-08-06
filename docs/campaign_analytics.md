# Campaign analytics module

This view analyses campaign performance by combining three SQLite databases
stored under ``email_marketing/data``:

- `email_events.db`
- `email_map.db`
- `campaigns.db`

## Running

```bash
streamlit run email_marketing/app.py
```

Select **Campaign Metrics** from the sidebar and provide the database paths if
they differ from the defaults. Pick a campaign in the selector to inspect
metadata and computed metrics.

## Testing

The unit test `tests/test_load_data.py` creates temporary databases and verifies
that data loading and metric computation work as expected. Run all tests with:

```bash
pytest
```
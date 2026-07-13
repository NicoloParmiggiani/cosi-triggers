# cosi-triggers

Python emulation and QA for the COSI GRB onboard trigger algorithm, aligned with
`COSI_FSW_014_CTM` (`iop_trig_ra_cfg`, `iop_trig_test_cfg`, `iop_trig_thresh_cfg`,
`iop_trig_chan_cfg`).

## Data

Available here https://drive.google.com/file/d/19y_4YvCpF48XrMaYZlpl6rAgRoNbbN8T/view?usp=drive_link. Download and extract at the top repo level.

## Pipeline

Run from this directory (requires ACS binned data under `sgrbs-background-binned-acs/`):

```bash
python3 trigger_acs.py && python3 trigger_match.py && python3 trigger_qa.py
```

Add `--complete` to each stage to process every flux band (default is fast mode:
`flux-1-10` and `flux-10-100` only).

## Configuration

- `algorithms.ini` — flight-style trigger test table (16 tests, 12 channels ch0–ch11)
- `data.ini` — simulation time bin width and energy edges

## Testing and improvements

- Updating 'algorithms.ini' and running the above command will generate new triggers based on the updated settings.

# Datasets

One directory per dataset under `data/`. Every dataset gets a row here.

| Directory | Dataset | Added | Role | Source |
| --- | --- | --- | --- | --- |
| `csic-2010/` | HTTP CSIC 2010 web application attacks | 2026-11-11 | Step 1, primary | https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks |
| (planned) `ait-lds-v2/` | AIT Log Data Set V2.0 | not downloaded | Step 2 | https://zenodo.org/records/5789064 |

## csic-2010

- Files: `csic-2010-web-application-attacks.zip` (2,419,705 bytes, SHA-256
  `67D7C414C730F51B46930CDF85A852F646137903108859816EDED6A9FE3D01C2`) and the
  extracted `csic_database.csv` (29,539,583 bytes).
- 61,065 rows: 36,000 `Normal` (label 0), 25,065 `Anomalous` (label 1).
- Methods: 43,088 GET, 17,580 POST, 397 PUT. 13,498 distinct URLs.
- Columns: an unnamed first column holding `Normal`/`Anomalous`, then `Method`,
  `User-Agent`, `Pragma`, `Cache-Control`, `Accept`, `Accept-encoding`,
  `Accept-charset`, `language`, `host`, `cookie`, `content-type`, `connection`,
  `lenght`, `content`, `classification`, `URL`.
- Generated traffic against one shop application; original dataset by CSIC
  (https://www.tic.itefi.csic.es/dataset/). The licence of the Kaggle upload
  has not been checked.

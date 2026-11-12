# Datasets

One directory per dataset under `data/`. Every dataset gets a row here.

| Directory | Dataset | Added | Role | Source |
| --- | --- | --- | --- | --- |
| `csic-2010/` | HTTP CSIC 2010 web application attacks | 2026-11-11 | Step 1, primary | https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks |
| `ait-lds-v2/` | AIT Log Data Set V2.0 | 2026-11-12 (download started, `russellmitchell` first) | Step 2 | https://zenodo.org/records/5789064 |

The code of this project is under the MIT licence (`LICENSE`). The datasets are
not part of the repository and keep their own terms.

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
  has not been checked; the owner decided on 2026-11-12 not to pursue it.

## ait-lds-v2

- Source: Zenodo record 5789064, licence CC BY-NC-SA 4.0 (non-commercial,
  share-alike). The data stays out of the repository.
- Eight testbeds, 121.7 GB zipped in total. Sizes and MD5 sums from Zenodo:

| File | Size | MD5 |
| --- | --- | --- |
| `russellmitchell.zip` | 6.64 GB | `78e9b7d169b438f03816019b52ab3ff7` |
| `santos.zip` | 9.32 GB | `fbd2663a41c83f345aa08a7fb11a6c6c` |
| `fox.zip` | 14.75 GB | `b6cd133398fbed6f3e7434dafbc45756` |
| `harrison.zip` | 15.62 GB | `4fbe3dd17a194776b2851116f8b51380` |
| `wardbeck.zip` | 15.98 GB | `ff524634e64c7779def2eb804e8c52ac` |
| `shaw.zip` | 16.39 GB | `872bbec7ffe12f14e06364e328a96088` |
| `wheeler.zip` | 18.25 GB | `e3f5605d867fcc644f93ae6c736cdb30` |
| `wilson.zip` | 24.72 GB | `952dd37cb262ef0054ad6d5371bdbe70` |

- Downloaded so far: `russellmitchell.zip` (7,132,670,599 bytes, MD5 verified
  on 2026-11-12). The other seven are in progress through
  `aria2-resume.ps1` in the dataset directory (capped at 4 MiB/s, resumable).

# HCAI Project

This repository contains the web applications developed for the Human-Centric
Artificial Intelligence course.

## Run Project 4

Project 4 is a Django study prototype comparing pairwise movie choices with
complete ten-movie rankings. Its standalone configuration avoids loading the
earlier coursework applications and their separate dependencies.

Requirements:

- Python 3.10 or newer
- `project4/data/movie_metadata.csv` (included in this branch)

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-project4.txt
python manage.py migrate --settings=project4.standalone_settings
python manage.py runserver --settings=project4.standalone_settings
```

Open <http://127.0.0.1:8000/project4/>. The landing page starts the study and
provides the Tasks 1--3 report as a PDF download.

Run all Project 4 checks with:

```powershell
python manage.py check --settings=project4.standalone_settings
python manage.py test project4 --settings=project4.standalone_settings
```

The full shared site still uses `pbl.settings`; its earlier projects may require
additional packages that are outside the Project 4 dependency set.

# HCAI Project

Django web application containing four Human-Centric Artificial Intelligence coursework projects. It uses SQLite and includes the datasets required by Projects 3 and 4; no external database, API credentials, Node.js runtime, or dataset download is required. The dataset for Project-1 is already available in the repository.

## Prerequisites

- Python 3.13

## Setup and run

Run the following commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
python manage.py runserver
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The development server listens on <http://127.0.0.1:8000/>. Available application entry points are:

| Application | URL |
| --- | --- |
| Project index | <http://127.0.0.1:8000/home/> |
| Project 1 | <http://127.0.0.1:8000/project1/> |
| Project 2 | <http://127.0.0.1:8000/project2/> |
| Project 3 | <http://127.0.0.1:8000/project3/> |
| Project 4 | <http://127.0.0.1:8000/project4/> |

# Project 4 report

`project4_report.html` is the editable, print-ready source for the Tasks 1–3
report. The generated file served by Django is:

`project4/static/project4/report/project4_report.pdf`

To rebuild it on Windows with Google Chrome installed, run the following from
the repository root:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless=new `
  --disable-gpu `
  --no-pdf-header-footer `
  --print-to-pdf="$PWD\project4\static\project4\report\project4_report.pdf" `
  "file:///$($PWD.Path.Replace('\', '/'))/project4/report/project4_report.html"
```

The HTML uses only embedded print styles, so generating the PDF does not need
network access or additional application dependencies.

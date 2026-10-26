# Refresh the table of contents / page fields of a .docx in Microsoft Word, save it, and export a PDF next to it.
# Windows + Microsoft Word only. Usage (from the repository root):
#   powershell -File scripts\docx_to_pdf.ps1 reports\module1_report.docx
param([Parameter(Mandatory = $true)][string]$Docx)
$ErrorActionPreference = "Stop"
$path = (Resolve-Path $Docx).Path
$pdf = [System.IO.Path]::ChangeExtension($path, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($path)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }   # second pass: page numbers settle after the TOC grows
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)   # 17 = wdExportFormatPDF
    $doc.Close()
    Write-Host "written: $pdf"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

"""Build every Module 1 deliverable from executed results: block diagram, poster, slides, report.

Usage (from the repository root):  python scripts/make_deliverables.py
Then, on Windows with Microsoft Word, refresh the table of contents and export the PDF:
    powershell -File scripts\\docx_to_pdf.ps1 reports\\module1_report.docx
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_architecture  # noqa: E402
import make_poster  # noqa: E402
import make_report  # noqa: E402
import make_slides  # noqa: E402

if __name__ == "__main__":
    make_architecture.main()
    make_poster.main()
    make_slides.main()
    make_report.main()

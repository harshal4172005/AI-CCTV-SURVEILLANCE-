# src/report_generator.py

from fpdf import FPDF
import os

class PDFReport:
    def __init__(self, title="PPE Violation Report"):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.title = title

    def header(self):
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, self.title, ln=True, align="C")
        self.pdf.ln(10)

    def add_violation(self, violation):
        self.pdf.set_font("Arial", '', 12)
        self.pdf.cell(0, 10, f"Time: {violation['timestamp']}", ln=True)
        self.pdf.cell(0, 10, f"Violation: {violation['violation_type']}", ln=True)
        self.pdf.ln(3)

        if os.path.exists(violation['image_path']):
            self.pdf.image(violation['image_path'], w=100)
        else:
            self.pdf.cell(0, 10, "Image not found", ln=True)
        
        self.pdf.ln(10)

    def generate(self, violations, save_path="ppe_violation_report.pdf"):
        self.pdf.add_page()
        self.header()

        for v in violations:
            self.add_violation(v)

        self.pdf.output(save_path)
        return save_path

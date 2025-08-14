import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

class PDFReport:
    """
    Generates a PDF report from violation data with a summary.
    """
    def __init__(self):
        self.output_path = os.path.join("app", "reports")
        os.makedirs(self.output_path, exist_ok=True)

    def generate(self, violations):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "PPE Violation Report", 0, 1, 'C')
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        
        # Add violation summary
        violation_counts = {}
        for v in violations:
            violation_type = v['violation_type']
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Total Violations: {len(violations)}", 0, 1)
        
        for v_type, count in violation_counts.items():
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"- {v_type}: {count}", 0, 1)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Violation Details", 0, 1)
        pdf.ln(5)

        for i, violation in enumerate(violations):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Violation {i+1}: {violation['violation_type']}", 0, 1)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 5, f"Timestamp: {violation['timestamp']}", 0, 1)
            
            if os.path.exists(violation['image_path']):
                pdf.image(violation['image_path'], w=60)
            else:
                pdf.cell(0, 10, "Image not found.", 0, 1)
            pdf.ln(5)

        filename = f"violation_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        filepath = os.path.join(self.output_path, filename)
        pdf.output(filepath)
        return filepath


class CSVReport:
    """
    Generates a CSV report from violation data.
    """
    def __init__(self):
        self.output_path = os.path.join("app", "reports")
        os.makedirs(self.output_path, exist_ok=True)

    def generate(self, violations):
        if not violations:
            return None

        # Prepare data for pandas DataFrame
        data = {
            "timestamp": [v["timestamp"] for v in violations],
            "violation_type": [v["violation_type"] for v in violations],
            "image_path": [v["image_path"] for v in violations]
        }
        
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        filename = f"violation_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(self.output_path, filename)
        df.to_csv(filepath, index=False)
        return filepath
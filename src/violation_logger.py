# src/violation_logger.py

from datetime import datetime

class ViolationLogger:
    def __init__(self):
        self.violations = []

    def add_violation(self, frame_image_path, violation_type):
        violation = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'violation_type': violation_type,
            'image_path': frame_image_path
        }
        self.violations.append(violation)

    def get_violations(self):
        return self.violations

    def clear(self):
        self.violations = []

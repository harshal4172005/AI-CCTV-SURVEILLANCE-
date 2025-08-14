import json
import os
from datetime import datetime

class ViolationLogger:
    def __init__(self):
        self.log_file = "app/data/violations.json"
        self.violations = []
        self._load_data()

    def _load_data(self):
        """Loads violation data from the JSON file."""
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                try:
                    self.violations = json.load(f)
                except json.JSONDecodeError:
                    self.violations = []

    def _save_data(self):
        """Saves current violation data to the JSON file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(self.violations, f, indent=4)

    def add_violation(self, image_path, violation_type):
        """Adds a new violation entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violation_entry = {
            "timestamp": timestamp,
            "violation_type": violation_type,
            "image_path": image_path
        }
        self.violations.append(violation_entry)
        self._save_data() # Save the new data

    def get_violations(self):
        """Returns the list of all logged violations."""
        return self.violations

    def clear(self):
        """Clears all violation data."""
        self.violations = []
        self._save_data() # Clear the data from the file
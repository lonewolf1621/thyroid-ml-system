import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_check(reference_path, current_path):

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    report.save_html("monitoring/drift_report.html")

    print("Drift report generated")
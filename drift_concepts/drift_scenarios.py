"""
Description: This module reads the drift scenarios defined by the .csv files in the directory ./drift_concepts/scenarios
 These .csv files define group specific distributed asynchronous drift patterns. This approach of defining drift
 scenarios is adopted by the implementation of FairFedDrfit [3]:
 [3]T. Salazar, J. Gama, H. Araújo, and P. H. Abreu, “Unveiling group-specific distributed concept drift: A fairness
 imperative in federated learning,” IEEE Access, vol. 12, pp. 1–18, 2024.

Author: Nisal Hemadasa
Date: 07-01-2026
Version: 1.0
"""

def get_drift_ids(scenario):
    f = open("./drift_concepts/scenarios/{}.csv".format(scenario))
    drift_ids = f.read().split("\n")
    drift_ids = [d.split(",") for d in drift_ids]
    drift_ids = [[int(a) for a in b] for b in drift_ids]
    f.close()

    return drift_ids
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


def read_drift_pattern_from_csv(num_scenario: int) -> list[list[int]]:
    """Reads the drift scenario from a .csv file and returns the drift ids as a list of lists.
    :param num_scenario: The scenario number to read.
    :return: A list of lists containing the drift ids for each client at each timestep.
            - outer list : timesteps (len(drift_pattern) -> number of timesteps)
            - inner list : clients (len(drift_pattern[0]) -> number of clients)
    """
    f = open("./drift_concepts/scenarios/{}.csv".format(num_scenario))
    drift_pattern = f.read().split("\n")
    drift_pattern = [d.split(",") for d in drift_pattern]
    drift_pattern = [[int(a) for a in b] for b in drift_pattern]
    f.close()

    return drift_pattern

def transpose_drift_pattern(drift_pattern: list[list[int]]):
    """Transposes the drift_pattern data structure from timestep-client format to client-timestep format and return
    relevant statistics.
    :param drift_pattern: A list of lists containing the drift ids for each client at each timestep.
            - outer list : timesteps (len(drift_pattern) -> number of timesteps)
            - inner list : clients (len(drift_pattern[0]) -> number of clients)
    :return: A tuple containing:
            - transposed drift_pattern
                - outer list : clients (len(drift_ids_col) -> number of clients)
                - inner list : timesteps (len(drift_ids_col[0]) -> number of timesteps)
            - number of clients
            - number of unique drift types
            - number of timesteps
    """
    # get the total number of clients
    num_clients = len(drift_pattern[0])
    # set() to store unique drift types
    drifts = set()
    # initialize data structure to store transposed drift pattern
    drift_pattern_transposed = [[] for _ in range(num_clients)]

    for timestep in range(len(drift_pattern)):
        for client in range(len(drift_pattern[timestep])):
            drift_id = drift_pattern[timestep][client]
            drift_pattern_transposed[client].append(drift_id)
            drifts.add(drift_id)

    num_timestamps = len(drift_pattern)
    num_drift_types = len(drifts)

    return drift_pattern_transposed, num_clients, num_drift_types, num_timestamps
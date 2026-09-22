"""Round-indexed local data history for the PaLA adaptation of FairFedDrift."""
from data.utils import DatasetSnapshot


class ClientDataHistory:
    """One client's arrivals; cluster assignments are managed by the strategy separately."""

    def __init__(self, window: int | None = 100):
        """
        Initialize bounded history without retaining any live dataset references.
        :param window: Positive retention length in communication rounds, or None.
        """
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("History window must be None or a positive integer of rounds.")
        self.window = window
        self.current_round = -1
        self._last_arrival_round = -1
        self._snapshots = {}

    def advance(self, round_idx: int) -> list[int]:
        """
        Advance the communication clock and release expired snapshots.
        :param round_idx: Nondecreasing round index; warm-up is 0, first training round is 1.
        :return: Expired arrival rounds for later assignment-history cleanup.
        """
        if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
            raise ValueError("History round must be a non-negative integer.")
        if round_idx < self.current_round:
            raise ValueError("History cannot move backwards in communication rounds.")
        self.current_round = round_idx
        cutoff = 0 if self.window is None else max(0, round_idx - self.window + 1)
        expired = [arrival for arrival in self._snapshots if arrival < cutoff]
        for arrival in expired:
            del self._snapshots[arrival]
        return expired

    def add(self, arrival_round: int, dataset) -> bool:
        """
        Capture one arrival without refreshing or duplicating an existing arrival.
        :param arrival_round: Current round of a new batch, or retained arrival round for reuse.
        :param dataset: Selected local dataset to snapshot only when this is a new arrival.
        :return: True for a new snapshot, False for repeated use of an arrival.
        """
        if isinstance(arrival_round, bool) or not isinstance(arrival_round, int) or arrival_round < 0:
            raise ValueError("History arrival round must be a non-negative integer.")
        if arrival_round in self._snapshots or arrival_round == self._last_arrival_round:
            return False
        if arrival_round < self.current_round or arrival_round < self._last_arrival_round:
            raise ValueError("Cannot insert an old arrival into current history.")
        # Validate/materialize first so a failed capture leaves the clock/history unchanged.
        snapshot = DatasetSnapshot(dataset)
        self.advance(arrival_round)
        self._snapshots[arrival_round] = snapshot
        self._last_arrival_round = arrival_round
        return True

    def records(self) -> list[tuple[int, DatasetSnapshot]]:
        """
        Return retained arrivals in order, without exposing the history mapping.
        :return: Arrival-round and snapshot pairs; consumers must release expired references.
        """
        return list(self._snapshots.items())

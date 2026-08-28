import json
import os
import time


class QuantLogger:
    def __init__(self, task, dataset, out_dir=None, run_id=None):
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "result", "logs")
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{dataset}_{run_id or int(time.time())}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._round = None

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def start_round(self, stage, r, **fields):
        self._round = {"stage": stage, "round": r, "events": {}, **fields}

    def event(self, name, inc=1):
        if self._round is not None:
            self._round["events"][name] = self._round["events"].get(name, 0) + inc

    def end_round(self, **fields):
        if self._round is None:
            return
        self._round.update(fields)
        self._fh.write(json.dumps(self._round, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        self._round = None

    def log(self, stage, r, **fields):
        self._fh.write(json.dumps({"stage": stage, "round": r, **fields}, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

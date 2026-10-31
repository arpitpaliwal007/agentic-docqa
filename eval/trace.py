"""Collect opt-in events, including events emitted inside parallel research calls."""
import threading


class Trace:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def __call__(self, event):
        with self.lock:
            self.events.append(dict(event))

    def reset(self):
        with self.lock:
            self.events.clear()

    def evidence(self):
        with self.lock:
            parents = [e["content"] for e in self.events if e["kind"] == "parent" and not e.get("error")]
            values = list(parents)
            for event in self.events:
                if event["kind"] == "preview":
                    values.append(event["content"])
                elif event["kind"] == "search" and not event.get("error"):
                    for chunk in event["chunks"]:
                        if not any(chunk["content"] in parent for parent in parents):
                            values.append(f"File: {chunk['source']}\nParent ID: {chunk['parent_id']}\n{chunk['content']}")
        return "\n\n".join(dict.fromkeys(values))

    def stats(self):
        with self.lock:
            grades = [e for e in self.events if e["kind"] == "grade"]
            return {"grade_events": len(grades),
                    "insufficient_events": sum(e["verdict"] == "insufficient" for e in grades),
                    "llm_insufficient_events": sum(e["verdict"] == "insufficient" and e["origin"] == "llm" for e in grades),
                    "grader_errors": sum(e["origin"] == "judge_parse_or_call_failure" for e in grades),
                    "retry_events": sum(bool(e["retry_scheduled"]) for e in grades),
                    "fallback_events": sum(e["kind"] == "fallback" for e in self.events),
                    "routes": [e["route"] for e in self.events if e["kind"] == "route"]}

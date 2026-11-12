"""The awake/sleep loop over a stream of problems, with a text playbook and hand-set rules.

Awake: solve a batch of problems with the current playbook in the prompt.
Sleep: turn this batch's failures into candidate lessons, then test each one with paired
replays: re-solve recent remembered problems of its family with and without it, using the
same sampling seed. A candidate is promoted if its gain is at least `tau` points. A lesson
whose removal no longer costs anything on recent problems of its family is dropped, which
is how stale lessons leave after a rule changes.

The model is reached only through two callables, so the logic can be tested with stubs:
    solver(entries, playbook_texts, seed) -> [(answer, score), ...]
    writer(failures)                      -> [(family, lesson_text), ...]
where failures are (entry, answer) pairs.
"""
from dataclasses import dataclass, field


@dataclass
class Lesson:
    family: str
    text: str
    gain: float          # last measured gain, in points
    added_at: int        # problem index at promotion


@dataclass
class LifetimeResult:
    scores: list = field(default_factory=list)       # awake score per problem, in order
    events: list = field(default_factory=list)       # promotions and removals, for the log
    replays: int = 0                                  # extra problems solved while testing


def family_of(entry):
    return entry["metadata"]["family"]


def texts(lessons):
    return [lesson.text for lesson in lessons]


def replay_gain(solver, probes, with_texts, without_texts, seed):
    """Paired gain in points: same problems, same seed, with and without."""
    with_scores = [s for _, s in solver(probes, with_texts, seed)]
    without_scores = [s for _, s in solver(probes, without_texts, seed)]
    return 100 * (sum(with_scores) - sum(without_scores)) / len(probes)


def run_lifetime(stream, solver, writer, tau, sleep_every=25, replay_n=12, capacity=10,
                 seed=0):
    playbook, memory, result = [], [], LifetimeResult()
    for start in range(0, len(stream), sleep_every):
        batch = stream[start:start + sleep_every]
        outcomes = solver(batch, texts(playbook), seed + start)
        result.scores += [s for _, s in outcomes]
        memory += [(entry, answer, s) for entry, (answer, s) in zip(batch, outcomes)]
        now = start + len(batch)

        def probes_for(family):
            recent = [e for e, _, _ in memory if family_of(e) == family]
            return recent[-replay_n:]

        # Remove lessons that stopped helping on recent problems of their family.
        active = {family_of(e) for e in batch}
        for lesson in [l for l in playbook if l.family in active]:
            probes = probes_for(lesson.family)
            others = [l for l in playbook if l is not lesson]
            gain = replay_gain(solver, probes, texts(playbook), texts(others), seed + now)
            result.replays += 2 * len(probes)
            if gain <= 0:
                playbook.remove(lesson)
                result.events.append(("remove", now, lesson.family, lesson.text, gain))
            else:
                lesson.gain = gain

        # Test candidate lessons from this batch's failures; promote the best ones.
        failures = [(entry, answer) for entry, answer, s in memory[start:now] if s < 1]
        tested = []
        for family, text in writer(failures) if failures else []:
            probes = probes_for(family)
            gain = replay_gain(solver, probes, texts(playbook) + [text], texts(playbook),
                               seed + now)
            result.replays += 2 * len(probes)
            tested.append((gain, family, text))
        for gain, family, text in sorted(tested, key=lambda t: -t[0]):
            if gain < tau:
                continue
            if len(playbook) >= capacity:
                weakest = min(playbook, key=lambda l: l.gain)
                if weakest.gain >= gain:
                    continue
                playbook.remove(weakest)
                result.events.append(("evict", now, weakest.family, weakest.text, weakest.gain))
            playbook.append(Lesson(family, text, gain, now))
            result.events.append(("promote", now, family, text, gain))
    return result

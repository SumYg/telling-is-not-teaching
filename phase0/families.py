"""Candidate Reasoning Gym families for early test 1.

Chosen for reasoning that a reusable strategy could help with, and for a mix of task types.
Families that need external data downloads (such as codeio) are left out.
"""

CANDIDATES = [
    # arithmetic search
    "countdown", "puzzle24", "cryptarithm",
    # logic
    "knights_knaves", "zebra_puzzles", "self_reference", "quantum_lock",
    # constraint puzzles
    "futoshiki", "mini_sudoku", "kakurasu", "survo",
    # pattern and function induction
    "arc_1d", "list_functions",
    # graphs and grids
    "word_ladder", "shortest_path", "graph_color", "course_schedule", "rotten_oranges",
    "game_of_life",
    # planning
    "jugs",
]

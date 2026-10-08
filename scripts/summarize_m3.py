"""Summarizes an M3 run (the JSON lines crucible.rl.mbpp_grpo writes)."""
import json
import statistics
import sys

recs = [json.loads(line) for line in open(sys.argv[1])]
for r in recs:
    if r.get("phase") == "tasks":
        print("tasks:", {k: (len(v) if isinstance(v, list) else v) for k, v in r.items() if k != "phase"})
    elif r.get("phase") == "eval":
        print(f"eval {r['when']}: pass@1 {100 * r['pass_at_1']:.1f}% of {r['n']} (tests passed {100 * r['mean_test_fraction']:.1f}%), "
              f"usual grader {100 * r.get('usual_pass', float('nan')):.1f}%, usual-only passes {r.get('usual_only')}, "
              f"truncated {100 * r['truncated']:.1f}%, mean {r['mean_tokens']:.0f} tokens")
steps = [r for r in recs if "step" in r]
if steps:
    k = min(10, len(steps))
    mean = lambda key, xs: statistics.mean(x[key] for x in xs)
    print(f"steps: {len(steps)}; reward first {k} {mean('reward', steps[:k]):.3f} -> last {k} {mean('reward', steps[-k:]):.3f}; "
          f"all tests pass {mean('pass_all', steps[:k]):.3f} -> {mean('pass_all', steps[-k:]):.3f}")
    print(f"answers the usual grader passes and crucible does not: {sum(s.get('usual_only', 0) for s in steps)} "
          f"of {sum(s['answers'] for s in steps)} (crucible-only: {sum(s.get('crucible_only', 0) for s in steps)})")
    print(f"per step: median {statistics.median(s['time_step'] for s in steps):.1f} s, "
          f"grading median {statistics.median(s['reward_seconds'] for s in steps):.2f} s; "
          f"total {steps[-1]['elapsed']:.0f} s")
for r in recs:
    if r.get("phase") == "paired":
        print(f"paired, same {r['n']} tasks (seed {r['seed']}): both pass {r['both_pass']}, fixed {r['fixed']} "
              f"({r['fixed_had_not_parsed']} had not parsed, {r['fixed_had_failed_tests']} had failed tests), "
              f"broke {r['broke']} ({r['broke_now_unparsed']} now do not parse), neither {r['neither']}; "
              f"McNemar exact p = {r['mcnemar_p']:.3g}")
        print(f"tasks {r['ids_sha1']} before {r['before_bits']}")
        print(f"tasks {r['ids_sha1']} after  {r['after_bits']}")

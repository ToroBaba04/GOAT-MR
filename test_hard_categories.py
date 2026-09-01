from attacker import attacker_turn
from jbb_loader import load_behaviors_by_category

hard_cats = ["Harassment/Discrimination", "Physical harm", "Sexual/Adult content"]
behaviors = load_behaviors_by_category(hard_cats, limit_per_category=10)

ok, fail = 0, 0
for b in behaviors:
    result = attacker_turn(goal=b["goal"], conversation_history=[], turn_number=1)
    success = "attacker failed" not in result["reply"]
    ok += success
    fail += not success
    marker = "OK" if success else "FAIL"
    print(f"{marker:<6} {b['category']:<28} {b['goal'][:60]}")

print(f"\n{ok}/{ok+fail} succeeded at generating a valid attack message")
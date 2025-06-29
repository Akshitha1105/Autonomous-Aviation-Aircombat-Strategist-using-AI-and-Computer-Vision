def generate_strategy(user_plane, enemy_plane, knowledge_base):
    user_info = knowledge_base[user_plane]
    enemy_info = knowledge_base[enemy_plane]

    def get_rank(value):
        rank = {"very low": 1, "low": 2, "medium": 3, "high": 4, "very high": 5}
        return rank.get(value.lower(), 0)

    strategy = {
        "advantages": [],
        "disadvantages": [],
        "win_probability": 0.5,
        "counter_strategy": "",
        "escape_plan": ""
    }

    # Role advantage
    user_role = user_info["role"].lower()
    enemy_role = enemy_info["role"].lower()

    if user_role in ["fighter", "multirole", "attack helicopter"] and enemy_role in ["transport", "bomber"]:
        strategy["advantages"].append("You are a combat aircraft, enemy is less agile and combat-capable.")
        strategy["win_probability"] += 0.3
    elif user_role in ["transport", "amphibious search & rescue"] and enemy_role in ["fighter", "multirole", "attack helicopter"]:
        strategy["disadvantages"].append("Enemy is combat-focused. You are vulnerable.")
        strategy["win_probability"] -= 0.4

    # Speed
    if get_rank(user_info["speed"]) > get_rank(enemy_info["speed"]):
        strategy["advantages"].append("You are faster.")
        strategy["win_probability"] += 0.1
    elif get_rank(user_info["speed"]) < get_rank(enemy_info["speed"]):
        strategy["disadvantages"].append("Enemy is faster.")
        strategy["win_probability"] -= 0.1
        

    # Maneuverability
    if get_rank(user_info["maneuverability"]) > get_rank(enemy_info["maneuverability"]):
        strategy["advantages"].append("Higher agility allows better evasion and positioning.")
        strategy["win_probability"] += 0.2
    elif get_rank(user_info["maneuverability"]) < get_rank(enemy_info["maneuverability"]):
        strategy["disadvantages"].append("Enemy has better agility.")
        strategy["win_probability"] -= 0.2

    # Armor
    if get_rank(user_info["armor"]) > get_rank(enemy_info["armor"]):
        strategy["advantages"].append("Better armored against attacks.")
        strategy["win_probability"] += 0.1
    elif get_rank(user_info["armor"]) < get_rank(enemy_info["armor"]):
        strategy["disadvantages"].append("Enemy has heavier armor.")
        strategy["win_probability"] -= 0.1

    # Counter-strategy and escape plans
    if strategy["win_probability"] >= 0.6:
        strategy["counter_strategy"] = "Engage using hit-and-run or flanking maneuvers."
    elif strategy["win_probability"] < 0.4:
        strategy["escape_plan"] = "Use altitude or terrain to evade. Request air support or return to base."
        strategy["counter_strategy"] = "Avoid direct engagement. Distract and retreat."
    else:
        strategy["counter_strategy"] = "Engage with caution. Monitor enemy behavior and exploit weaknesses."

    # Clamp win probability
    strategy["win_probability"] = round(max(0.0, min(strategy["win_probability"], 1.0)), 2)

    return strategy

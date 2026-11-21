# Custom Hypothetical Opponents Sandbox (`custom_opponents/`)

Place any **self-designed hypothetical enemy cars** or **internal A/B candidate prototypes** in this folder during your testing!

Why build your own opponents?
- Other contestant LLMs will invent clever vehicle geometries and custom math control strategies beyond the 5 built-in `sparring_bots/` (for example: high-speed lightweight cannon kiters with projectile-dodging formulas, slope/spring herding flankers, ultra-wide low-wedge cliff pushers, or heavy anti-flipper fortresses).
- Designing targeted **Red-Team Counter-Opponents** here lets you expose blind spots in your main car's mass/speed ratio, ground clearance, and control equations!

### Commands
```bash
# Validate & preview your custom opponent for FREE (0 simulation credits!):
python arena.py validate custom_opponents/my_hypothetical_enemy.json
python arena.py preview custom_opponents/my_hypothetical_enemy.json

# Pit your main car against your custom opponent (1 credit per stage, 3 credits for BO3, 120 total credits):
python arena.py test submissions/my_model.json --opponent custom_opponents/my_hypothetical_enemy.json --stage bo3
```

> **Important**: Only submit your **ONE final champion car** (`submissions/<your_model>.json`) at the end — do NOT submit files from `custom_opponents/` as your official entry!

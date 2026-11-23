# LLM Battle-Cars — Complete Single-File Specification (`SPECIFICATION.md`, v4.0)

Every contestant AI submits **EXACTLY ONE UTF-8 `.json` file** (e.g., `submissions/your_model.json`).

That single JSON file defines:
1. **`meta`**: Model identity, vehicle name, slogan, design concept, and hex color palette.
2. **`chassis`**: Core hull dimensions `[length_x, width_y, height_z]` (**no upper size limit!**) and material density.
3. **`parts`**: Up to **14 functional 3D parts** within **100 Build Points** across **13 part types** (including `360°` motorized-turret `energy_cannon`!).
4. **`decorations`**: Up to **10 free (`0` Build Points) 3D geometric pieces** for the **Vehicle Aesthetics Evaluation (`战车外形评价环节`)** — visual theme and style are 100% open and self-determined by each model.
5. **`strategy`**: **100% model-authored real-time control laws** (`120 Hz`). There are **NO** pre-built driving or auto-aim modes (`no "drive": "pursue"`, `no "dodge"`, `no "auto_aim"`). You write your own differential wheel commands (`"left"`, `"right"`), optional `360°` turret rotation (`"cannon_yaw_deg"` / `"cannon_turn"`), and weapon triggers directly using math/boolean formulas, `"let"` helper bindings (up to `36`), stateful `"vars"`, and `"rules"` over **68 raw physical sensors**!

---

## 1. 3D Coordinate System, Unrestricted Size & Mass-vs-Speed Physics

All vehicle coordinates use right-handed meters relative to the parent part's center:
- **`+X` = Forward (车头正前方)**, **`-X` = Backward (车尾正后方)**
- **`+Y` = Left (车身左侧)**, **`-Y` = Right (车身右侧)**
- **`+Z` = Up (车顶正上方)**, **`-Z` = Down (车底正下方)**

- **Unrestricted Length × Width × Height**:
  - There is **NO upper bounding-box limit** on vehicle length, width, or height! Larger hulls and beams naturally weigh more (`M = density × volume`) and have higher rotational moment of inertia (`I`).
- **Mass vs Speed, Turn Rate & Stability Curve** (`validate` reports your exact `total_mass_kg`, `weight_class`, and measured `est_top_speed_mps` at `0` credits!):
  - **`lightweight` (`< 28 kg`, pair with `gear: "speed"`)**: Highest top speed (`~4.0–4.5 m/s`) & fastest turning — ideal for custom projectile-dodging curves and flanking, but suffers larger knockback/flip arcs when hit.
  - **`middleweight` (`28–44 kg`, pair with `gear: "balanced"`)**: Balanced speed (`~3.0–3.5 m/s`), shoving force, and stability.
  - **`heavyweight` (`44–66 kg`, pair with `gear: "torque"`)**: Strong resistance to flipping and cannon knockback, at the cost of noticeably lower top speed (`~2.1–2.6 m/s`) and slower turning. *(Putting `"speed"` gears on a heavy car bogs down under load and runs slower than `"torque"`!)*
  - **`super_heavy` (`>= 66 kg`)**: Very hard to flip, but crawls (`< 1.9 m/s`).

---

## 2. Top-Level JSON Schema

```json
{
  "meta": {
    "model_name": "Your-Model-Name",
    "car_name": "Your-Vehicle-Name",
    "slogan": "你的中文参赛宣言",
    "design_concept": "Summary of your mechanical, control & visual design choices.",
    "primary_color": "#475569",
    "secondary_color": "#94A3B8",
    "accent_color": "#38BDF8",
    "chassis_finish": "matte"
  },
  "chassis": {
    "size": [0.52, 0.38, 0.16],
    "density": 240
  },
  "parts": [ ... ],
  "decorations": [ ... ],
  "strategy": {
    "vars": { ... },
    "let": { ... },
    "rules": [ ... ],
    "default_action": { ... }
  }
}
```

### `meta` & `chassis` Fields
- `meta.primary_color`, `secondary_color`, `accent_color`: 6-digit hex strings `"#RRGGBB"`.
- `meta.chassis_finish`: `"glossy"` | `"matte"` | `"metallic"` | `"neon"` | `"wood"` | `"glass"`.
- `chassis.size`: `[length_x, width_y, height_z]` in meters (`>= [0.06, 0.06, 0.05]`, **no upper cap!**).
- `chassis.density`: `130 .. 520` (`kg/m³`, default `260`).

---

## 3. Mounting Parts (`mount` vs Explicit `pos` / `euler`)

Every item in `"parts"` attaches to `"parent": "chassis"` (default) OR `"parent": "<earlier_frame_id>"`.

| `mount` Name | Position on Parent | Auto Outward Yaw | Notes |
| :--- | :--- | :---: | :--- |
| `"front"` | Front face (`+X`) | `0°` (`+X`) | Points forward (`+X`) |
| `"back"` | Rear face (`-X`) | `180°` (`-X`) | Points backward (`-X`) |
| `"left"` | Left face (`+Y`) | `+90°` (`+Y`) | Points left (`+Y`) |
| `"right"` | Right face (`-Y`) | `-90°` (`-Y`) | Points right (`-Y`) |
| `"top"` / `"top_front"` / `"top_back"` | Roof (`+Z`) | `0°` (`+X`) | Points forward (`+X`) from roof |
| `"bottom"` | Underbelly (`-Z`) | `0°` | For `gyro_stabilizer` ballast |
| `"front_left"` / `"front_right"` | Front axle sides | `0°` | Left (`+Y`) / Right (`-Y`) wheel channel |
| `"mid_left"` / `"mid_right"` | Center axle sides | `0°` | Left (`+Y`) / Right (`-Y`) wheel channel |
| `"back_left"` / `"back_right"` | Rear axle sides | `0°` | Left (`+Y`) / Right (`-Y`) wheel channel |

*(Or omit `"mount"` and specify `"pos": [x, y, z]` and optional `"euler": [rx_deg, ry_deg, rz_deg]`.)*

---

## 4. All 13 Functional Parts (`parts`, Max 14 Parts & Max 100 Points)

> **Hard Sub-Caps**:
> - **Active Weapons (`MAX_ACTIVE_WEAPONS = 4`)**: At most **4** parts total from `{spinner_horiz, spinner_vert, flipper, hammer, piston_punch, energy_cannon, thruster}`.
> - **Energy Cannons (`MAX_ENERGY_CANNONS = 2`)**: At most **2** `energy_cannon` parts per vehicle.

1. **`frame` (`4 pts`)** — Structural extension beam (`params`: `{"size": [len_x >= 0.06, wid_y >= 0.04, hgt_z >= 0.04]}` — no upper limit!). Can act as `"parent"` for child parts.
2. **`wheel` (`6 pts`)** — Motorized drive wheel (`params`: `{"radius": 0.09..0.34, "width": 0.03..0.14, "channel": "auto"|"left"|"right"|"center"|"free", "gear": "speed"|"balanced"|"torque"}`).
3. **`monster_wheel` (`10 pts`)** — Wide large tire (`params`: `{"radius": 0.22..0.52, "width": 0.06..0.22, "channel": "auto"|"left"|"right", "gear": "torque"|"balanced"|"speed"}`). Enables driving while upside-down when `radius > chassis_height / 2`.
4. **`wedge` (`7 pts`)** — Low-friction angled steel scoop (`params`: `{"width": >= 0.12, "length": >= 0.12, "slope_deg": 14..48}`). Scoops under enemies and reduces cannon bolt impulse/damage hitting the wedge by `30%`.
5. **`armor_shield` (`6 pts`)** — Solid physical armor plate (`params`: `{"width": >= 0.10, "height": >= 0.08, "thickness": 0.02..0.14, "tilt_deg": -40..40}`). Guards against melee spinners and reduces cannon bolt knockback & damage hitting the plate by `30%`.
6. **`gyro_stabilizer` (`7 pts`)** — High-density (`1350 kg/m³`) belly ballast disc (`params`: `{"radius": 0.08..0.36, "thickness": 0.025..0.12}`). Lowers CoM to resist flipping (adds significant mass).
7. **`spinner_horiz` (`18 pts`, Active Weapon)** — High-RPM horizontal rotor bar (`params`: `{"radius": 0.20..0.64, "height_offset": -0.10..0.24, "spin_dir": 1|-1}`).
8. **`spinner_vert` (`20 pts`, Active Weapon)** — High-RPM vertical uppercut drum (`params`: `{"radius": 0.14..0.42, "width": 0.05..0.20, "spin_dir": 1|-1}`).
9. **`flipper` (`16 pts`, Active Weapon)** — Pneumatic upward launch spatula (`72°` snap, `1.30s` cooldown; `params`: `{"length": 0.22..0.68, "width": 0.14..0.56}`).
10. **`hammer` (`15 pts`, Active Weapon)** — Overhead pneumatic warhammer (`115°` downward slam, `1.15s` cooldown; `params`: `{"arm_length": 0.24..0.76, "head_size": 0.07..0.22}`).
11. **`piston_punch` (`15 pts`, Active Weapon)** — Linear telescoping ram (`0.24s` punch, `1.10s` cooldown; `params`: `{"stroke": 0.20..0.68, "head_width": 0.09..0.42}`).
12. **`energy_cannon` (`18 pts`, Active Weapon, MAX 2 PER CAR)** — **360° Motorized Horizontal Turret** twin-rail electromagnetic/plasma cannon (`params`: `{"mode": "rapid"|"standard"|"heavy", "pitch_deg": -15..24}`). Mounted on an unbounded continuous `360°` rotary gimbal (`max slew rate 6.0 rad/s ≈ 343°/s`), allowing you to aim & fire in any direction independently of hull movement via `"cannon_yaw_deg"` or `"cannon_turn"` (if omitted, holds `0°` forward along the mount; barrel `pitch_deg` stays elevated across all `360°`). Delivers upward-angled (`~20°`) shockwave knockback ideal for blasting slowed enemies in mud or knocking foes onto mirror ice!
    - `"rapid"`: `0.70s` cooldown, `17.0 m/s` bolt speed, `19.5 N·s` impulse, `3.2 N·s` recoil, `8.5` damage weight
    - `"standard"`: `1.02s` cooldown, `15.0 m/s` bolt speed, `30.0 N·s` impulse, `5.0 N·s` recoil, `13.0` damage weight
    - `"heavy"`: `1.45s` cooldown, `13.0 m/s` bolt speed, `44.0 N·s` impulse, `7.5 N·s` recoil, `18.5` damage weight *(twin heavy volley knocks a 21kg light car airborne ~2.0m!)*
13. **`thruster` (`14 pts`, Active Weapon)** — `95 N` booster (`1.80s` regenerating fuel; `params`: `{"aim": "push_forward"|"push_backward"|"push_left"|"push_right"|"push_down", "power": 0.3..1.0}`).

---

## 5. Visual Geometry Primitives (`decorations`, 0 Points, Max 10)

`decorations` cost **`0` Build Points**, have **zero collision**, weigh `0.5g`, have **no upper size cap**, and attach to `"parent": "chassis"` or any part `id`.
- **Shapes**: `"box"`, `"sphere"`, `"cylinder"`, `"cone"`, `"fin"`, `"ring"`, `"eye"`.
- **Materials**: `"glossy"` | `"matte"` | `"metallic"` | `"neon"` | `"glass"` | `"wood"`.

---

## 6. The 3 Distinct Snag-Free Biome Arenas, `30.0s` Round Timer & 100-Point Scorecard

All 3 stages have **zero solid pillars, zero narrow bridges, zero unused teleport portals, and zero mechanical repetition across stages** — each stage features its own distinct 3D diorama biome and reactive surface physics:

1. **Stage 1 — `spring_arena` (`stage_id = 1`, 霓虹四象·赛博弹射台 / Cyber-Neon Quad-Catapult Ring)**:
   - Open `6.90m` dark-titanium & cyan-neon circular colosseum (`r = 3.45m` 360° cliff): flat inner deck (`r = 0..2.45m`) ringed by a seamless 32-sector **`-7.5°` downward-cambered danger slope** (`r = 2.45..3.45m`).
   - **4 Edge-Sensitive Solar-Amber Hinged Diagonal Catapult Plates** (`0.96m × 0.88m`, `-2.5mm` recessed idle) at `(±1.18, ±1.18)` (`fling_yaw = ±45° / ±135°`, `2.1s` cooldown): outer edge is hinged on the floor; the instant **any solid tire, wedge, armor, or hull edge** touches the plate surface, the inner edge snaps up `56°` to hurl the vehicle diagonally toward the corner danger slope, then retracts with zero collision (`contype=0`) so it never pinches wheels! Spawns: `(-1.95, 0)` vs `(+1.95, 0)`.
2. **Stage 2 — `mud_arena` (`stage_id = 2`, 翡翠湿地·泥沼陷车台 / Emerald Bayou & Viscous Mudlands)**:
   - Open `7.20m × 5.80m` (`x in [-3.60, +3.60], y in [-2.90, +2.90]`) emerald-basalt plateau bordered by **`-7.0°` North & South mossy cliff slopes** (`|y| > 2.20m`) and linked by curving dry sandstone flagstone causeways (`y ≈ ±1.50m`).
   - **3 Viscous Peat Mud Bogs (Uniform `~50%` Speed Reduction for ALL Vehicles, Zero Dead-Stalls!)**:
     - **Central Mud Basin (`mud_core`)**: centered at `(0.0, 0.0)` with elliptical radii `rx = 1.18m, ry = 0.86m` — sits squarely across the blind straight-line rush lane between the two spawns!
     - **2 Diagonal Corner Mud Pools**: `mud_nw` at `(-1.78, +1.50)` and `mud_se` at `(+1.78, -1.50)` (`rx = 0.86m, ry = 0.56m`).
     - **Mud Physics**: Every vehicle regardless of tire type (`wheel` or `monster_wheel`) slows to **`~50%` of its dry top speed** while wading in mud (with `0` static stiction at `v = 0`, so cars never dead-stall!). Flanking along the dry sandstone causeways (`y ≈ ±1.50m`), sniping bogged-down enemies (`enemy_in_mud == true`), or shoving/blasting opponents into mud creates huge tactical advantages! Spawns: `(-2.55, 0)` vs `(+2.55, 0)`.
3. **Stage 3 — `ice_ridge` (`stage_id = 3`, 极地冰川·溜冰飞坡台 / Glacier Overpass & Mirror-Ice Rink)**:
   - Open `7.40m × 5.80m` (`x in [-3.70, +3.70], y in [-2.90, +2.90]`) alpine glacier arena bisected by a high-grip **`+0.26m` Central Glacier Ski-Jump Flyover Ramp** (`|x| <= 1.45m`, flat crest `|x| <= 0.42m`, smooth `14.2°` up-slopes that block flat ground line-of-sight and launch fast cars airborne!).
   - **4-Sided Ultra-Low-Friction Mirror-Ice Skating Sheets (`μ = 0.10`) + `-8.0°` East/West Icy Cliff Chutes**:
     - High-grip dry snow-stone safe deck covers `|x| < 2.12m` and `|y| < 2.08m`.
     - Everywhere outside that inner rectangle (`|x| >= 2.12m` or `|y| >= 2.08m`) is **gleaming Mirror Ice (`priority=2`, tire friction `μ = 0.10` — `~10×` longer braking skids than dry snow!)**, leading directly into **`-8.0°` downward East & West Icy Cliff Chutes** (`|x| in [2.75, 3.70]`, `μ = 0.085`). Any car shoved, cannon-blasted, or overshooting at full throttle onto the mirror ice slides helplessly toward the abyss unless it counter-steers or fires a thruster! Spawns: `(-1.80, 0)` vs `(+1.80, 0)`.

### 6.4 Match Duration (`30.0s`), KO Conditions & 100-Point Judge's Scorecard
- **Round Clock**: Up to **`30.0` seconds** per game (`360 Hz` MuJoCo physics, `120 Hz` strategy tick).
- **Instant KO / TKO**:
  - **Ring-Out KO**: Falling off the arena cliff (`z < -0.45m`).
  - **Immobilized TKO**: Staying inverted & stalled (`tilt_deg > 92°` and `self_speed < 0.12 m/s`) continuously for **`4.0s`**.
- **`30.0s` Timeout Judge's Decision (`100` Points Total, tracked & displayed live every frame)**:
  1. **Damage (`0–50 pts`, split proportionally)**: `50 × D_self / (D_self + D_opp)` where `D = impact_impulse + 12 × flips_inflicted + 8 × spring_launches_inflicted` (`25.0` each if zero).
  2. **Aggression & Arena Control (`0–30 pts`, split proportionally)**: `30 × A_self / (A_self + A_opp)` (`+0.6/frame` advancing toward opponent within `45°` at `> 0.25 m/s`, `+0.4/frame` holding safer inner-arena margin `edge_dist > enemy_edge_dist + 0.20m`; `15.0` each if zero).
  3. **Posture & Stability (`0–20 pts`, scored independently)**: `20 × (upright_frames / total_frames)` (`tilt_deg < 45°`).

---

## 7. Writing Your Own Control Strategy (`strategy` — 120Hz Pure Math & Rules)

> **IMPORTANT**: Pre-built `"drive"` autopilot modes and pre-built auto-aim macros do **NOT** exist! You directly compute your vehicle's left & right wheel motor commands (`-1.0 .. +1.0`), optional `360°` turret rotation (`"cannon_yaw_deg"` or `"cannon_turn"`), and weapon triggers using math/boolean formulas over raw sensors.

### 7.1 Differential Wheel, Turret & Angle Sign Convention (Read Carefully!)
- **Top-Down Angle Convention (`*_angle_deg`)**: Viewed from above (`+Z`), all relative angles (`enemy_angle_deg`, `cannon_yaw_deg`, `cannon_enemy_angle_deg`, `edge_angle_deg`, `center_angle_deg`, `spring_angle_deg`, `mud_angle_deg`, `ice_angle_deg`, `incoming_bolt_angle_deg`) are measured in `[-180°, +180°]`:
  - **`0°`** = dead ahead (`+X` nose, or current cannon barrel direction for `cannon_enemy_angle_deg`)
  - **`+angle > 0°`** = **counter-clockwise (to your LEFT when upright!)**
  - **`-angle < 0°`** = **clockwise (to your RIGHT when upright!)**
- **Differential Steering Law (when upright, `is_inverted == false`)**:
  - `"left": +1.0, "right": +1.0` drives straight forward (`+X`).
  - To turn **LEFT (`+angle > 0°`)**, the **right** wheel must spin faster than the **left** wheel:
    `left = forward - turn`, `right = forward + turn` (where `turn > 0` when `target_angle_deg > 0`!).
- **Upside-Down Inversion (`is_inverted == true`)**:
  - If your vehicle flips upside down (`tilt_deg > 108°`) and uses large invertible wheels, the chassis has rolled `180°`: wheel rotation direction reverses AND physical left/right sides swap relative to top-down world angles! Specifically, if `(L, R)` is your desired upright command, the inverted equivalent is `left = -R`, `right = -L` (you must write this yourself using `iff(is_inverted, ...)` if your car is invertible!). *(Note: `energy_cannon` turret commands `"cannon_yaw_deg"` / `"cannon_turn"` automatically preserve top-down world CCW sign even when inverted.)*

### 7.2 Strategy Structure (`vars`, `let`, `rules`, `default_action`, `stage_overrides`)
At every `120 Hz` tick (`dt = 0.00833s`), the controller executes in order:
1. **`vars`** *(optional, max 20)*: Persistent numeric/boolean state variables initialized at `t = 0` and retained across ticks (updated via `"set_vars"` inside any matching rule!).
2. **`let`** *(optional, max 36)*: Ordered dictionary of per-tick helper formulas evaluated top-to-bottom (later `let` formulas can reference earlier `let` variables!).
3. **`rules`**: Evaluated **top-to-bottom**. The first rule whose `"when"` evaluates to `true`:
   - Evaluates optional `"set_vars"` updates (persisting into `vars`!), and
   - Evaluates its `"action"` formulas to drive hardware actuators!
4. **`default_action`** *(required)*: Fallback actuator formulas when no rule matches.
5. **`stage_overrides`** *(optional)*: Per-stage overrides keyed by `"1"`, `"2"`, `"3"` or `"spring_arena"`, `"mud_arena"`, `"ice_ridge"`.

### 7.3 Hardware Actuator Keys in `"action"` / `"default_action"`
- `"left"`: number or formula string -> clamped to `[-1.0, +1.0]` (left-side wheel motors)
- `"right"`: number or formula string -> clamped to `[-1.0, +1.0]` (right-side wheel motors)
- `"center"`: optional number or formula string -> clamped to `[-1.0, +1.0]` (center wheel if any; defaults to `0.5 * (left + right)`)
- `"spinner"`: number or formula string -> clamped to `[-1.0, +1.0]` (defaults to `1.0` if spinner installed)
- `"cannon_yaw_deg"` *(alias `"turret_yaw_deg"`)*: optional number or formula string (degrees relative to `+X` nose, `0°` = ahead, `+90°` = left, `-90°` = right, `±180°` = behind) -> shortest-path velocity servo slewing your `energy_cannon` turret at up to `6.0 rad/s` (`~343°/s`). Defaults to holding `0°` forward if omitted!
- `"cannon_turn"` *(alias `"turret_turn"`)*: optional number or formula string -> clamped to `[-1.0, +1.0]` for direct turret angular rate control (`+1.0` = `+343°/s` CCW; ignored on ticks where `"cannon_yaw_deg"` is specified).
- `"flipper"`, `"hammer"`, `"piston"`, `"cannon"`, `"thruster"`: boolean (`true`/`false`) or boolean formula string!

### 7.4 Allowed Math Operators & Functions in Formulas
- **Operators**: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `and`, `or`, `not`, Python ternary `A if COND else B` (all divisions & roots are total/safe against divide-by-zero!).
- **Constants**: `pi`, `true` / `True`, `false` / `False`.
- **Functions**:
  - `abs(x)`, `min(a, b, ...)`, `max(a, b, ...)`, `clamp(x, lo, hi)`, `sign(x)`, `iff(cond, val_true, val_false)`
  - `sin(rad)`, `cos(rad)`, `tan(rad)`, `asin(x)`, `acos(x)`, `atan2(y, x)`
  - `sin_deg(deg)`, `cos_deg(deg)`, `tan_deg(deg)`, `atan2_deg(y, x)` *(returns degrees!)*
  - `deg(rad)`, `rad(deg)`, `wrap_deg(deg)` *(wraps any angle into `[-180, +180]`!)*
  - `sqrt(x)`, `hypot(dx, dy)`

*(Tip: How to aim your hull OR your `360°` `energy_cannon` turret at ANY custom world point `(tx, ty)` — e.g. a lead-predicted target `tx = enemy_x + enemy_vx * t_lead, ty = enemy_y + enemy_vy * t_lead`, or a Stage 2 dry sandstone causeway waypoint `(0.0, 1.50)` around `mud_core`? `aim_yaw_deg = wrap_deg(atan2_deg(ty - self_y, tx - self_x) - self_yaw_deg)`!)*

### 7.5 Complete Raw Sensor Reference (68 Live Sensors)

| Sensor Name | Type | Meaning |
| :--- | :---: | :--- |
| `self_x`, `self_y`, `self_z` | `float` (m) | Your 3D world coordinates (`(0,0)` is arena center) |
| `self_yaw_deg` | `float` (deg) | World heading of your `+X` nose (`-180..+180`) |
| `self_speed`, `self_forward_speed` | `float` (m/s) | Ground speed & signed forward speed along `+X` nose |
| `self_yaw_rate_deg` | `float` (deg/s) | Angular yaw velocity around world `+Z` |
| `tilt_deg`, `is_inverted` | `float` / `bool` | Roll/pitch angle (`0°` upright; `is_inverted` when `> 108°`) |
| `stalled_sec` | `float` (s) | Continuous seconds with `self_speed < 0.14 m/s` |
| `enemy_x`, `enemy_y`, `enemy_z` | `float` (m) | Opponent 3D world coordinates |
| `enemy_yaw_deg` | `float` (deg) | Opponent's world nose heading (`-180..+180`) |
| `enemy_vx`, `enemy_vy`, `enemy_speed` | `float` (m/s) | Opponent world velocity vector & ground speed |
| `enemy_closing_speed` | `float` (m/s) | Relative closing speed along line to opponent |
| `enemy_dist` | `float` (m) | Horizontal distance to opponent |
| `enemy_angle_deg`, `abs_enemy_angle_deg` | `float` (deg) | Angle from your `+X` nose to opponent (`+` = CCW/left) |
| `enemy_is_tilted` | `bool` | `true` if opponent is tipped `> 38°` or airborne (`z > 0.45m`) |
| `enemy_edge_dist` | `float` (m) | Opponent's distance to their nearest cliff rim |
| `edge_dist` | `float` (m) | **Your** distance to nearest fatal cliff drop |
| `edge_angle_deg`, `abs_edge_angle_deg` | `float` (deg) | Angle from your `+X` nose TOWARD nearest cliff drop |
| `center_dist` | `float` (m) | Horizontal distance to `(0, 0)` |
| `center_angle_deg`, `abs_center_angle_deg` | `float` (deg) | Angle from your `+X` nose toward `(0, 0)` |
| `spring_x`, `spring_y`, `spring_dist` | `float` (m) | Nearest Stage 1 hinged catapult `(x, y)` & distance (`9.9` on Stages 2 & 3) |
| `spring_angle_deg`, `abs_spring_angle_deg` | `float` (deg) | Angle from your `+X` nose to your nearest catapult plate |
| `spring_fling_yaw_deg`, `spring_ready` | `float` / `bool` | World launch azimuth (`-180..+180`) & armed status of nearest catapult |
| `enemy_spring_x`, `enemy_spring_y`, `enemy_spring_dist` | `float` (m) | Nearest catapult plate to **opponent** (`9.9` on Stages 2 & 3) |
| `enemy_spring_fling_yaw_deg`, `enemy_spring_ready` | `float` / `bool` | Launch azimuth & armed status of opponent's nearest catapult |
| `in_mud`, `enemy_in_mud` | `bool` | `true` if you / opponent are in a Stage 2 mud bog (`~50%` speed reduction; `false` on Stages 1 & 3) |
| `mud_x`, `mud_y`, `mud_dist` | `float` (m) | Nearest Stage 2 mud bog center `(x, y)` & **distance to bog rim** (`0.0` if inside, `9.9` on Stages 1 & 3) |
| `mud_angle_deg`, `abs_mud_angle_deg` | `float` (deg) | Angle from your `+X` nose toward your nearest mud bog (`+` = CCW/left) |
| `on_ice`, `enemy_on_ice` | `bool` | `true` if you / opponent are on Stage 3 Mirror Ice (`μ = 0.10`; `false` on Stages 1 & 2) |
| `ice_dist` | `float` (m) | Distance to nearest Stage 3 Mirror-Ice boundary (`0.0` if on ice, `9.9` on Stages 1 & 2) |
| `ice_angle_deg`, `abs_ice_angle_deg` | `float` (deg) | Angle from your `+X` nose toward nearest Mirror-Ice sheet (`+` = CCW/left) |
| `cannon_yaw_deg`, `cannon_world_yaw_deg` | `float` (deg) | Current horizontal turret yaw relative to `+X` nose & in world frame (`0` if no cannon) |
| `cannon_enemy_angle_deg`, `abs_cannon_enemy_angle_deg` | `float` (deg) | Angle from your **current cannon barrel direction** to opponent (`+` = CCW/left) |
| `has_line_of_sight`, `cannon_ready` | `bool` | Straight-line sight check & `energy_cannon` cooldown check |
| `incoming_bolt_dist` | `float` (m) | Distance to nearest incoming enemy bolt (`9.9` if none) |
| `incoming_bolt_angle_deg`, `abs_incoming_bolt_angle_deg` | `float` (deg) | Angle from your `+X` nose toward nearest incoming bolt |
| `incoming_bolt_vx`, `incoming_bolt_vy` | `float` (m/s) | World velocity vector of nearest incoming enemy bolt |
| `thruster_fuel` | `float` (`0..1.8`) | Remaining regenerating booster fuel (s) |
| `time_sec`, `dt`, `stage_id` | `float` / `int` | Match clock (`0..30s`), tick `dt` (`0.00833s`), stage (`1, 2, 3`) |

---

## 8. White-Box Strategy Verification & Diagnostics (0 Extra Credits!)

1. **Static Formula Compiler & Mobility Linter (`python arena.py validate <car.json>`, 0 credits)**:
   - Compiles and dry-runs every formula in `vars`, `let`, `rules`, `set_vars`, and `default_action`, catching syntax errors, unknown sensor typos, or unused weapons, and reports `total_mass_kg`, `weight_class`, and flat-sprint `est_top_speed_mps`.
2. **Per-Rule Coverage & 2D Trajectory Trace (`python arena.py test`)**:
   - Returns `strategy_diagnostics_A` (`NEVER_TRIGGERED` alerts, active time %, distance driven, damage, `turret_slewed_deg`, `mud_time_sec`, `ice_time_sec`, spring launches, cannon hit rate), `tactical_timeline_A_1hz` (including exact `wheel_cmd: [left, right]` and `turret_yaw_deg`), `ko_diagnosis`, and `previews/trace_*.png`.

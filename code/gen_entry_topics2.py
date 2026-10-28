# -*- coding: utf-8 -*-
"""MIA2 項目3(読解): topics_themes.json を整える二段目(オーナー承認 2026-10-25)
 ① 少ないテーマ(昆虫・宇宙・天気)に、手で書いた題材を足す
 ② 全題材をQwenが「いちばん合うテーマ」へ振り分ける(違うテーマに入っていたものを移す)。低学年に難しい・変な題材は消す
 ③ 語彙の条件(2語は辞書にある名前、1語は頻度、割れる語なし)をもう一度かけて重複を除く
出力: data/reading_entry/topics_themes_v2.json / .txt  (元の topics_themes.json は残す)
使い方: python gen_entry_topics2.py   (先にvLLMサーバーを起動)
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa_local as local
import gen_entry_topics as T

OUT = T.OUT
MANUAL = {
 "insects": "ant, bee, butterfly, ladybug, ladybird, dragonfly, moth, caterpillar, cricket, grasshopper, firefly, mosquito, fly, wasp, hornet, beetle, spider, snail, worm, earthworm, centipede, millipede, cicada, termite, flea, bumblebee, honeybee, praying mantis, stick insect, cockroach, slug, scorpion, pill bug, cocoon, larva, anthill, beehive, cobweb, honey, wing, antenna, queen bee, lightning bug, june bug, daddy longlegs, tarantula, locust, aphid, maggot, tadpole",
 "space": "sun, moon, earth, mercury, venus, mars, jupiter, saturn, uranus, neptune, pluto, star, planet, comet, meteor, asteroid, galaxy, milky way, solar system, rocket, astronaut, spaceship, space station, satellite, telescope, orbit, eclipse, full moon, new moon, crescent moon, shooting star, constellation, big dipper, north star, space suit, moon rock, crater, black hole, universe, sky, spacecraft, rover, space shuttle, launch pad, moon landing, sunrise, sunset, night sky, day, night",
 "vehicles": "airport, train station, bus stop, harbor, garage, highway, train track, runway, car, bus, train, truck, boat, ship, plane, bicycle, tractor, fire truck, police car, ambulance, helicopter, subway, taxi, motorcycle, scooter, sled, canoe, sailboat, hot air balloon, rowboat, wagon, stroller, skateboard",
 "weather": "rain, snow, wind, cloud, sun, storm, thunder, lightning, rainbow, fog, mist, hail, sleet, frost, ice, dew, breeze, tornado, hurricane, blizzard, flood, drought, heat wave, thunderstorm, raindrop, snowflake, snowman, puddle, spring, summer, autumn, winter, sunshine, shower, drizzle, gust, cyclone, weather, forecast, thermometer, icicle, snowstorm, rain cloud, sandstorm, windy day, cold weather, hot weather, season, freezing rain, snowball",
}
THEME_DESC = "\n".join(f"{k}: {v}" for k, v in T.THEMES.items())
JUDGE = """Themes:
{themes}
For each numbered word below, choose the ONE theme that fits it best, or "none".
Every ordinary thing, place, person, job or event that a child in grade 1-2 (age 6-8) knows MUST get a theme: choose the closest one (for example "airport" -> vehicles, "actor" -> school_work, "gold" -> earth, "shadow" -> weather).
Use "none" ONLY if the word is not a real thing (an adjective, verb, feeling, idea, made-up thing), or a child in grade 1-2 would not know it, or it is a hard word for such a child (body-science words, technical words).
Examples: "polar bear" -> animals, "umbrella" -> home, "tadpole" -> animals, "palate" -> none, "operation" -> none.
Output one line per word in the form "number: theme" and nothing else.
{lines}"""


def assign(items):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
    txt = local.ask(JUDGE.format(themes=THEME_DESC, lines=lines), max_tokens=900, temperature=0.0)
    out = {}
    for m in re.finditer(r"(\d+)\s*[:.)-]\s*([a-z_]+)", txt):
        i, th = int(m.group(1)), m.group(2)
        if 1 <= i <= len(items) and (th in T.THEMES or th == "none"):
            out[i - 1] = th
    return [(t, out.get(i, "none")) for i, t in enumerate(items)]


def main():
    cur = json.load(open(os.path.join(OUT, "topics_themes.json")))
    start = {t: len(v) for t, v in cur.items()}
    cand = {}
    for th, v in cur.items():
        for t in v:
            cand.setdefault(t, th)
    for th, s in MANUAL.items():
        for t in (x.strip() for x in s.split(",")):
            if T.ok(t):
                cand.setdefault(t, th)
    items = sorted(cand)
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = [p for part in ex.map(assign, [items[i:i + 80] for i in range(0, len(items), 80)]) for p in part]
    new, moved, removed = {t: [] for t in T.THEMES}, 0, []
    for t, th in res:
        if th == "none":
            removed.append(t); continue
        moved += th != cand[t]
        new[th].append(t)
    json.dump(new, open(os.path.join(OUT, "topics_themes_v2.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "topics_themes_v2.txt"), "w", encoding="utf-8") as f:
        for t, v in new.items():
            f.write(f"## {t} ({len(v)})\n" + ", ".join(v) + "\n\n")
        f.write(f"## 消した題材 ({len(removed)})\n" + ", ".join(removed) + "\n")
    print(f"候補 {len(items)} → 採用 {sum(len(v) for v in new.values())} / 別のテーマへ移動 {moved} / 消した {len(removed)}")
    for t in T.THEMES:
        print(f"{t}: {start[t]} → {len(new[t])}")


if __name__ == "__main__":
    main()

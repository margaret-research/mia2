# -*- coding: utf-8 -*-
"""追加題材(topics_add2.json)から、テーマ違い・低学年に不向き・重複(綴り違い・複数形違い)を手で除いて確定する → topics_add_final.json/.txt"""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reading_entry")
CUT = {x.strip() for x in """farm, wings, fin, aquarium, cows, garden snail, honeycomb, stinger, spider web, red light, solar panel, ticket, ticket booth, luggage, parking lot, life jacket, buoy, dock, wharf, fuel, platform,
brick, metal, fire hose, fire pit, drain, footprint, fountain, stone wall, windmill, wind turbine, compost, cold water, hot water, fire, deserts, country, atmosphere, temperature, drop, drip, rain barrel,
space, cosmos, hubble, voyager, orion, zodiac, oxygen mask, electric toothbrush, baby, sound, taste, touch, smell, sight, nap, breath, food, lunch, snack, vitamin, ice cube, banana peel, pepper mill, soda can, pulp,
grandma, grandpa, parent, haircut, hall, ballroom, board, can, deck, light, plastic, leather, silk, wool, yarn, thread, spray, pump, vacuum, magnet, robot, globe, bulb, wire, coin, dime, slime, sling, sticker, theater, picture, park,
paint, game, play, balance, race, tumbler, twister, rattle, gym shoes, page, helper, history, test, folder, office, coconut tree, grape vine, ice skate, roller skates, finger paints, mud pies, stick figures, jumping jacks,
beech tree, fir tree, aloe, weeds, treehouse, seatbelt, tyre, ax, wind chimes, aeroplane, racing car, chimp, urchin, lego""".replace("\n", " ").split(",")}
add = json.load(open(os.path.join(D, "topics_add2.json")))
old = {t for v in json.load(open(os.path.join(D, "topics_themes_v2.json"))).values() for t in v}
new = {th: [t for t in v if t not in CUT and t not in old] for th, v in add.items()}
json.dump(new, open(os.path.join(D, "topics_add_final.json"), "w"), ensure_ascii=False, indent=1)
with open(os.path.join(D, "topics_add_final.txt"), "w", encoding="utf-8") as f:
    for th, v in new.items():
        f.write(f"## {th} ({len(v)})\n" + ", ".join(v) + "\n\n")
print(sum(map(len, add.values())), "→", sum(map(len, new.values())), {t: len(v) for t, v in new.items()})

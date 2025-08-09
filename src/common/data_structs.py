import pandas as pd
from enum import Enum
from pydantic import BaseModel


class InputData(BaseModel):
    total: int = 0
    attack: int = 0
    defence: int = 0
    strength: int = 0
    hitpoints: int = 0
    ranged: int = 0
    prayer: int = 0
    magic: int = 0
    cooking: int = 0
    woodcutting: int = 0
    fletching: int = 0
    fishing: int = 0
    firemaking: int = 0
    crafting: int = 0
    smithing: int = 0
    mining: int = 0
    herblore: int = 0
    agility: int = 0
    thieving: int = 0
    slayer: int = 0
    farming: int = 0
    runecraft: int = 0
    hunter: int = 0
    construction: int = 0
    league: int = 0
    bounty_hunter_hunter: int = 0
    bounty_hunter_rogue: int = 0
    cs_all: int = 0
    cs_beginner: int = 0
    cs_easy: int = 0
    cs_medium: int = 0
    cs_hard: int = 0
    cs_elite: int = 0
    cs_master: int = 0
    lms_rank: int = 0
    soul_wars_zeal: int = 0
    abyssal_sire: int = 0
    alchemical_hydra: int = 0
    barrows_chests: int = 0
    bryophyta: int = 0
    callisto: int = 0
    cerberus: int = 0
    chambers_of_xeric: int = 0
    chambers_of_xeric_challenge_mode: int = 0
    chaos_elemental: int = 0
    chaos_fanatic: int = 0
    commander_zilyana: int = 0
    corporeal_beast: int = 0
    crazy_archaeologist: int = 0
    dagannoth_prime: int = 0
    dagannoth_rex: int = 0
    dagannoth_supreme: int = 0
    deranged_archaeologist: int = 0
    general_graardor: int = 0
    giant_mole: int = 0
    grotesque_guardians: int = 0
    hespori: int = 0
    kalphite_queen: int = 0
    king_black_dragon: int = 0
    kraken: int = 0
    kreearra: int = 0
    kril_tsutsaroth: int = 0
    mimic: int = 0
    nightmare: int = 0
    nex: int = 0
    phosanis_nightmare: int = 0
    obor: int = 0
    phantom_muspah: int = 0
    sarachnis: int = 0
    scorpia: int = 0
    skotizo: int = 0
    tempoross: int = 0
    the_gauntlet: int = 0
    the_corrupted_gauntlet: int = 0
    theatre_of_blood: int = 0
    theatre_of_blood_hard: int = 0
    thermonuclear_smoke_devil: int = 0
    tombs_of_amascut: int = 0
    tombs_of_amascut_expert: int = 0
    tzkal_zuk: int = 0
    tztok_jad: int = 0
    venenatis: int = 0
    vetion: int = 0
    vorkath: int = 0
    wintertodt: int = 0
    zalcano: int = 0
    zulrah: int = 0
    rifts_closed: int = 0
    artio: int = 0
    calvarion: int = 0
    duke_sucellus: int = 0
    spindel: int = 0
    the_leviathan: int = 0
    the_whisperer: int = 0
    vardorvis: int = 0


class CombatSkill(Enum):
    ATTACK = "attack"
    DEFENCE = "defence"
    STRENGTH = "strength"
    HITPOINTS = "hitpoints"
    RANGED = "ranged"
    PRAYER = "prayer"
    MAGIC = "magic"


class GatheringSkill(Enum):
    WOODCUTTING = "woodcutting"
    FISHING = "fishing"
    MINING = "mining"
    FARMING = "farming"
    HUNTER = "hunter"


class ArtisanSkill(Enum):
    COOKING = "cooking"
    FLETCHING = "fletching"
    FIREMAKING = "firemaking"
    CRAFTING = "crafting"
    SMITHING = "smithing"
    HERBLORE = "herblore"
    RUNECRAFT = "runecraft"
    CONSTRUCTION = "construction"


class SupportSkill(Enum):
    AGILITY = "agility"
    THIEVING = "thieving"
    SLAYER = "slayer"


class MemberSkill(Enum):
    FLETCHING = "fletching"
    HERBLORE = "herblore"
    AGILITY = "agility"
    THIEVING = "thieving"
    SLAYER = "slayer"
    FARMING = "farming"
    HUNTER = "hunter"
    CONSTRUCTION = "construction"


class Skill(Enum):
    ATTACK = "attack"
    DEFENCE = "defence"
    STRENGTH = "strength"
    HITPOINTS = "hitpoints"
    RANGED = "ranged"
    PRAYER = "prayer"
    MAGIC = "magic"
    COOKING = "cooking"
    WOODCUTTING = "woodcutting"
    FLETCHING = "fletching"
    FISHING = "fishing"
    FIREMAKING = "firemaking"
    CRAFTING = "crafting"
    SMITHING = "smithing"
    MINING = "mining"
    HERBLORE = "herblore"
    AGILITY = "agility"
    THIEVING = "thieving"
    SLAYER = "slayer"
    FARMING = "farming"
    RUNECRAFT = "runecraft"
    HUNTER = "hunter"
    CONSTRUCTION = "construction"

    def __repr__(self):
        return self.value


class ClueScroll(Enum):
    CS_ALL = "cs_all"
    CS_BEGINNER = "cs_beginner"
    CS_EASY = "cs_easy"
    CS_MEDIUM = "cs_medium"
    CS_HARD = "cs_hard"
    CS_ELITE = "cs_elite"
    CS_MASTER = "cs_master"

    def __repr__(self):
        return self.value


class Minigame(Enum):
    LEAGUE = "league"
    BOUNTY_HUNTER_HUNTER = "bounty_hunter_hunter"
    BOUNTY_HUNTER_ROGUE = "bounty_hunter_rogue"
    LMS_RANK = "lms_rank"
    SOUL_WARS_ZEAL = "soul_wars_zeal"
    CS_ALL = "cs_all"
    CS_BEGINNER = "cs_beginner"
    CS_EASY = "cs_easy"
    CS_MEDIUM = "cs_medium"
    CS_HARD = "cs_hard"
    CS_ELITE = "cs_elite"
    CS_MASTER = "cs_master"

    def __repr__(self):
        return self.value


class Boss(Enum):
    abyssal_sire = "abyssal_sire"
    alchemical_hydra = "alchemical_hydra"
    barrows_chests = "barrows_chests"
    bryophyta = "bryophyta"
    callisto = "callisto"
    cerberus = "cerberus"
    chambers_of_xeric = "chambers_of_xeric"
    chambers_of_xeric_challenge_mode = "chambers_of_xeric_challenge_mode"
    chaos_elemental = "chaos_elemental"
    chaos_fanatic = "chaos_fanatic"
    commander_zilyana = "commander_zilyana"
    corporeal_beast = "corporeal_beast"
    crazy_archaeologist = "crazy_archaeologist"
    dagannoth_prime = "dagannoth_prime"
    dagannoth_rex = "dagannoth_rex"
    dagannoth_supreme = "dagannoth_supreme"
    deranged_archaeologist = "deranged_archaeologist"
    general_graardor = "general_graardor"
    giant_mole = "giant_mole"
    grotesque_guardians = "grotesque_guardians"
    hespori = "hespori"
    kalphite_queen = "kalphite_queen"
    king_black_dragon = "king_black_dragon"
    kraken = "kraken"
    kreearra = "kreearra"
    kril_tsutsaroth = "kril_tsutsaroth"
    mimic = "mimic"
    nex = "nex"
    nightmare = "nightmare"
    phosanis_nightmare = "phosanis_nightmare"
    obor = "obor"
    sarachnis = "sarachnis"
    scorpia = "scorpia"
    skotizo = "skotizo"
    tempoross = "tempoross"
    the_gauntlet = "the_gauntlet"
    the_corrupted_gauntlet = "the_corrupted_gauntlet"
    theatre_of_blood = "theatre_of_blood"
    theatre_of_blood_hard = "theatre_of_blood_hard"
    thermonuclear_smoke_devil = "thermonuclear_smoke_devil"
    tombs_of_amascut = "tombs_of_amascut"
    tombs_of_amascut_expert = "tombs_of_amascut_expert"
    tzkal_zuk = "tzkal_zuk"
    tztok_jad = "tztok_jad"
    venenatis = "venenatis"
    vetion = "vetion"
    vorkath = "vorkath"
    wintertodt = "wintertodt"
    zalcano = "zalcano"
    zulrah = "zulrah"

    def __repr__(self):
        return self.value


level_exp = [
    {"level": 1, "exp": 0, "exp_diff": None},
    {"level": 2, "exp": 83, "exp_diff": 83},
    {"level": 3, "exp": 174, "exp_diff": 91},
    {"level": 4, "exp": 276, "exp_diff": 102},
    {"level": 5, "exp": 388, "exp_diff": 112},
    {"level": 6, "exp": 512, "exp_diff": 124},
    {"level": 7, "exp": 650, "exp_diff": 138},
    {"level": 8, "exp": 801, "exp_diff": 151},
    {"level": 9, "exp": 969, "exp_diff": 168},
    {"level": 10, "exp": 1154, "exp_diff": 185},
    {"level": 11, "exp": 1358, "exp_diff": 204},
    {"level": 12, "exp": 1584, "exp_diff": 226},
    {"level": 13, "exp": 1833, "exp_diff": 249},
    {"level": 14, "exp": 2107, "exp_diff": 274},
    {"level": 15, "exp": 2411, "exp_diff": 304},
    {"level": 16, "exp": 2746, "exp_diff": 335},
    {"level": 17, "exp": 3115, "exp_diff": 369},
    {"level": 18, "exp": 3523, "exp_diff": 408},
    {"level": 19, "exp": 3973, "exp_diff": 450},
    {"level": 20, "exp": 4470, "exp_diff": 497},
    {"level": 21, "exp": 5018, "exp_diff": 548},
    {"level": 22, "exp": 5624, "exp_diff": 606},
    {"level": 23, "exp": 6291, "exp_diff": 667},
    {"level": 24, "exp": 7028, "exp_diff": 737},
    {"level": 25, "exp": 7842, "exp_diff": 814},
    {"level": 26, "exp": 8740, "exp_diff": 898},
    {"level": 27, "exp": 9730, "exp_diff": 990},
    {"level": 28, "exp": 10824, "exp_diff": 1094},
    {"level": 29, "exp": 12031, "exp_diff": 1207},
    {"level": 30, "exp": 13363, "exp_diff": 1332},
    {"level": 31, "exp": 14833, "exp_diff": 1470},
    {"level": 32, "exp": 16456, "exp_diff": 1623},
    {"level": 33, "exp": 18247, "exp_diff": 1791},
    {"level": 34, "exp": 20224, "exp_diff": 1977},
    {"level": 35, "exp": 22406, "exp_diff": 2182},
    {"level": 36, "exp": 24815, "exp_diff": 2409},
    {"level": 37, "exp": 27473, "exp_diff": 2658},
    {"level": 38, "exp": 30408, "exp_diff": 2935},
    {"level": 39, "exp": 33648, "exp_diff": 3240},
    {"level": 40, "exp": 37224, "exp_diff": 3576},
    {"level": 41, "exp": 41171, "exp_diff": 3947},
    {"level": 42, "exp": 45529, "exp_diff": 4358},
    {"level": 43, "exp": 50339, "exp_diff": 4810},
    {"level": 44, "exp": 55649, "exp_diff": 5310},
    {"level": 45, "exp": 61512, "exp_diff": 5863},
    {"level": 46, "exp": 67983, "exp_diff": 6471},
    {"level": 47, "exp": 75127, "exp_diff": 7144},
    {"level": 48, "exp": 83014, "exp_diff": 7887},
    {"level": 49, "exp": 91721, "exp_diff": 8707},
    {"level": 50, "exp": 101333, "exp_diff": 9612},
    {"level": 51, "exp": 111945, "exp_diff": 10612},
    {"level": 52, "exp": 123660, "exp_diff": 11715},
    {"level": 53, "exp": 136594, "exp_diff": 12934},
    {"level": 54, "exp": 150872, "exp_diff": 14278},
    {"level": 55, "exp": 166636, "exp_diff": 15764},
    {"level": 56, "exp": 184040, "exp_diff": 17404},
    {"level": 57, "exp": 203254, "exp_diff": 19214},
    {"level": 58, "exp": 224466, "exp_diff": 21212},
    {"level": 59, "exp": 247886, "exp_diff": 23420},
    {"level": 60, "exp": 273742, "exp_diff": 25856},
    {"level": 61, "exp": 302288, "exp_diff": 28546},
    {"level": 62, "exp": 333804, "exp_diff": 31516},
    {"level": 63, "exp": 368599, "exp_diff": 34795},
    {"level": 64, "exp": 407015, "exp_diff": 38416},
    {"level": 65, "exp": 449428, "exp_diff": 42413},
    {"level": 66, "exp": 496254, "exp_diff": 46826},
    {"level": 67, "exp": 547953, "exp_diff": 51699},
    {"level": 68, "exp": 605032, "exp_diff": 57079},
    {"level": 69, "exp": 668051, "exp_diff": 63019},
    {"level": 70, "exp": 737627, "exp_diff": 69576},
    {"level": 71, "exp": 814445, "exp_diff": 76818},
    {"level": 72, "exp": 899257, "exp_diff": 84812},
    {"level": 73, "exp": 992895, "exp_diff": 93638},
    {"level": 74, "exp": 1096278, "exp_diff": 103383},
    {"level": 75, "exp": 1210421, "exp_diff": 114143},
    {"level": 76, "exp": 1336443, "exp_diff": 126022},
    {"level": 77, "exp": 1475581, "exp_diff": 139138},
    {"level": 78, "exp": 1629200, "exp_diff": 153619},
    {"level": 79, "exp": 1798808, "exp_diff": 169608},
    {"level": 80, "exp": 1986068, "exp_diff": 187260},
    {"level": 81, "exp": 2192818, "exp_diff": 206750},
    {"level": 82, "exp": 2421087, "exp_diff": 228269},
    {"level": 83, "exp": 2673114, "exp_diff": 252027},
    {"level": 84, "exp": 2951373, "exp_diff": 278259},
    {"level": 85, "exp": 3258594, "exp_diff": 307221},
    {"level": 86, "exp": 3597792, "exp_diff": 339198},
    {"level": 87, "exp": 3972294, "exp_diff": 374502},
    {"level": 88, "exp": 4385776, "exp_diff": 413482},
    {"level": 89, "exp": 4842295, "exp_diff": 456519},
    {"level": 90, "exp": 5346332, "exp_diff": 504037},
    {"level": 91, "exp": 5902831, "exp_diff": 556499},
    {"level": 92, "exp": 6517253, "exp_diff": 614422},
    {"level": 93, "exp": 7195629, "exp_diff": 678376},
    {"level": 94, "exp": 7944614, "exp_diff": 748985},
    {"level": 95, "exp": 8771558, "exp_diff": 826944},
    {"level": 96, "exp": 9684577, "exp_diff": 913019},
    {"level": 97, "exp": 10692629, "exp_diff": 1008052},
    {"level": 98, "exp": 11805606, "exp_diff": 1112977},
    {"level": 99, "exp": 13034431, "exp_diff": 1228825},
    {"level": 100, "exp": 14391160, "exp_diff": 1356729},
    {"level": 101, "exp": 15889109, "exp_diff": 1497949},
    {"level": 102, "exp": 17542976, "exp_diff": 1653867},
    {"level": 103, "exp": 19368992, "exp_diff": 1826016},
    {"level": 104, "exp": 21385073, "exp_diff": 2016081},
    {"level": 105, "exp": 23611006, "exp_diff": 2225933},
    {"level": 106, "exp": 26068632, "exp_diff": 2457626},
    {"level": 107, "exp": 28782069, "exp_diff": 2713437},
    {"level": 108, "exp": 31777943, "exp_diff": 2995874},
    {"level": 109, "exp": 35085654, "exp_diff": 3307711},
    {"level": 110, "exp": 38737661, "exp_diff": 3652007},
    {"level": 111, "exp": 42769801, "exp_diff": 4032140},
    {"level": 112, "exp": 47221641, "exp_diff": 4451840},
    {"level": 113, "exp": 52136869, "exp_diff": 4915228},
    {"level": 114, "exp": 57563718, "exp_diff": 5426849},
    {"level": 115, "exp": 63555443, "exp_diff": 5991725},
    {"level": 116, "exp": 70170840, "exp_diff": 6615397},
    {"level": 117, "exp": 77474828, "exp_diff": 7303988},
    {"level": 118, "exp": 85539082, "exp_diff": 8064254},
    {"level": 119, "exp": 94442737, "exp_diff": 8903655},
    {"level": 120, "exp": 104273167, "exp_diff": 9830430},
    {"level": 121, "exp": 115126838, "exp_diff": 10853671},
    {"level": 122, "exp": 127110260, "exp_diff": 11983422},
    {"level": 123, "exp": 140341028, "exp_diff": 13230768},
    {"level": 124, "exp": 154947843, "exp_diff": 14606715},
    {"level": 125, "exp": 171048195, "exp_diff": 16000352},
]


class XPTable:
    def __init__(self) -> None:
        # level_exp.csv attained from https://oldschool.runescape.wiki/w/Experience
        self.df = pd.DataFrame(data=level_exp)

    def exp_to_level(self, exp: int) -> int:
        if exp == 0:
            return 1

        mask = self.df["exp"] <= exp
        value = self.df[mask].iloc[-1]["level"]
        return int(value)

    def level_to_exp(self, level: int) -> int:
        if level < 1:
            return 0

        mask = self.df["level"] <= level
        value = self.df[mask].iloc[-1]["exp"]
        return int(value)


XP_TABLE = XPTable()
assert XP_TABLE.exp_to_level(0) == 1, "exp: 0 should return level 1"
assert XP_TABLE.exp_to_level(13_034_431) == 99, "exp: 13,034,431 should return level 99"
assert XP_TABLE.level_to_exp(1) == 0, "lvl: 1 should have 0 exp"
assert XP_TABLE.level_to_exp(95) == 8_771_558, "lvl: 95 should have 8,771,558 exp"

SKILLS = [c.value for c in Skill]
MINIGAMES = [c.value for c in Minigame]
BOSSES = [c.value for c in Boss]

print(SKILLS)
print(MINIGAMES)
print(BOSSES)

HISCORE_COLUMNS = ["total"] + SKILLS + MINIGAMES + BOSSES

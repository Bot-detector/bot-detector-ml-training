import pandas as pd
from enum import Enum


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

class XPTable:
    def __init__(self) -> None:
        # level_exp.csv attained from https://oldschool.runescape.wiki/w/Experience
        filename = "../data/level_exp.csv"
        self.df = pd.read_csv(filename, sep="\t") # columns ['Level', 'Exp.', 'Exp. Diff', '% to 99']

    def exp_to_level(self, exp:int) -> int:
        if exp == 0:
            return 1

        mask = self.df["Exp."] <= exp
        value = self.df[mask].iloc[-1]["Level"]
        return int(value)

    def level_to_exp(self, level:int) -> int:
        if level < 1:
            return 0

        mask = self.df["Level"] <= level
        value = self.df[mask].iloc[-1]["Exp."]
        return int(value)

XP_TABLE = XPTable()
assert XP_TABLE.exp_to_level(0) == 1, "Experience of 0 should return level 1"
assert XP_TABLE.exp_to_level(13_034_431) == 99, "Experience of 13,034,431 should return level 99"
assert XP_TABLE.level_to_exp(1) == 0, "Level 1 should have 0 experience"
assert XP_TABLE.level_to_exp(95) == 8_771_558, "Level 95 should have 8,771,558 experience"

SKILLS = [c.value for c in Skill]
MINIGAMES = [c.value for c in Minigame]
BOSSES = [c.value for c in Boss]

HISCORE_COLUMNS = ["total"] + SKILLS + MINIGAMES + BOSSES
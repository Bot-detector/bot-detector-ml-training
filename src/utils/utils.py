import pandas as pd
import numpy as np
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

SKILLS = [c.value for c in Skill]
MINIGAMES = [c.value for c in Minigame]
BOSSES = [c.value for c in Boss]

HISCORE_COLUMNS = ["total"] + SKILLS + MINIGAMES + BOSSES

def get_ratio(df: pd.DataFrame, COLUMNS: list, total_column:str=None, column_suffix:str="ratio") -> pd.DataFrame:
    """
    Calculate the ratio of each column in the given DataFrame to the sum of all columns in COLUMNS.
    
    Args:
        df: The DataFrame to calculate ratios for.
        COLUMNS: A list of columns to calculate ratios for.
        total_column: The name of the column to use for the total.
        column_suffix: The suffix to use for the ratio columns.
    
    Returns:
        A new DataFrame with the same index as df where each column in COLUMNS has been replaced
        with the corresponding ratio. Additionally, the function calculates the total of all columns
        in COLUMNS and adds this as a new column in the returned DataFrame with the name specified
        by total_column.
    """
    _df = pd.DataFrame(index=df.index)
    TOTAL = df[COLUMNS].sum(axis=1)
    for column in COLUMNS:
        _df[f"{column}_{column_suffix}"] = df[column] / TOTAL
    if total_column:
        _df[total_column] = TOTAL
    return _df
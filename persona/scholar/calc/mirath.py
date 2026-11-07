"""Mirath (mawarith) - shares of an estate, in exact fractions.

The owner ruling (ARCHITECTURE.md section 11): anything needing *calculation*
is computed deterministically, never by a model. So this whole module uses
:class:`fractions.Fraction`: no floats, no rounding, no LLM in the
arithmetic - a wrong fraction is a wrong fatwa said in a living scholar's name.

``distribute`` applies the classical steps in order, each a small named
function:

1. **hajb** (:func:`_hajb`). Which heirs the closers shut out, and by whom.
2. **fards** (:func:`_fards`). The prescribed shares (al-furud al-muqaddara),
   including the rule that sisters above/below take their fard only when no
   male residuary makes them 'asaba with him.
3. **'umariyya** (:func:`_umariyya`). With spouse + 2+ sisters and mother, the
   mother takes 1/3 of the remainder after the spouse, not 1/3 of the estate.
4. **'awl** (:func:`_apply_awl`). When the fards' numerators exceed the base,
   the base is raised to their sum: every fard is scaled down proportionally.
5. **residue** (:func:`_give_residue`). What the fards leave goes to the
   'asaba, males at twice a female per head (:data:`_MALES`).
6. **radd** (:func:`_radd`). When residue is left and there is no 'asaba, it
   returns to the non-spouse fard-holders in the ratio of their fards; if the
   only holder is a spouse, the residue goes to the treasury and we say so.

``shares`` is keyed by heir *class* ("daughters" is the whole class's share);
divide by the head count for per-head numbers (:func:`per_head`). ``base`` is
the 'asl al-masala of the solved problem, so the answer reads straight against
the classical tables (husband with two full sisters 'awl on 7, ``'umariyya`` 8,
minbariyya 27).
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm

__all__ = ["HEIRS", "MirathError", "distribute"]

ZERO = Fraction(0)
ONE = Fraction(1)
HALF = Fraction(1, 2)
QUARTER = Fraction(1, 4)
EIGHTH = Fraction(1, 8)
THREE_QUARTERS = Fraction(3, 4)
THIRD = Fraction(1, 3)
SIXTH = Fraction(1, 6)
TWO_THIRDS = Fraction(2, 3)

#: survivor keys accepted by :func:`distribute`; values are head counts
HEIRS: tuple[str, ...] = (
    "husband", "wife", "sons", "daughters", "grandsons", "granddaughters",
    "father", "mother", "grandfather",
    "full_brothers", "full_sisters", "consang_brothers", "consang_sisters",
    "uterine_siblings",
)

#: heads that count twice when the residue is split 2:1
_MALES = frozenset(("sons", "grandsons", "full_brothers", "consang_brothers"))

#: any member of this = there is issue (far'), which halves the spouse
_ISSUE_KEYS = ("sons", "daughters", "grandsons", "granddaughters")

#: siblings the father or a son shuts out
_BLOCKED_BY_FATHER = ("grandfather", "full_brothers", "full_sisters",
                      "consang_brothers", "consang_sisters", "uterine_siblings")


class MirathError(ValueError):
    """Input the classical tables do not cover. Never guess a share for it."""


# --- public API ---------------------------------------------------------------


def distribute(survivors: dict[str, int]) -> dict:
    """Split one estate. Returns ``shares/blocked/base/awl_applied/radd_applied/note``.

    ``shares`` is keyed by heir class - ``"daughters"`` is the whole class's
    share, not each head's - so a caller who needs a per-head number divides by
    ``survivors[heir]``: one small operation, and no risk of double-counting a
    class. ``base`` is the classical ``'asl al-masala`` of the *solved* problem:
    the smallest integer the final shares run on, which can be read straight
    against the tables (awl on 7 for a husband with two full sisters,
    ``'umariyya`` 8, minbariyya 27).
    """
    counts = _validate(survivors)
    notes: list[str] = []
    blocked_by = _hajb(counts)
    for heir, why in sorted(blocked_by.items()):
        notes.append(f"{heir} mahjub by {why}")
    # every class stays a key of ``live``, blocked ones counted out at zero, so
    # each step below can read `live[h]` instead of guarding for a missing key
    live = {h: (0 if h in blocked_by else n) for h, n in counts.items()}

    fards = _fards(live)
    _umariyya(live, fards, notes)
    awl, shares = _apply_awl(fards, notes)

    ashes = [h for h in ("sons", "grandsons", "father", "grandfather",
                         "full_brothers", "consang_brothers") if live[h]]
    residue = ONE - sum(shares.values(), ZERO)
    if ashes and residue > 0:
        shares = _merge(shares, _give_residue(live, ashes, residue))
        notes.append("residue of {} to the 'asaba ({})".format(residue, ", ".join(ashes)))

    radd, treasury = False, False
    residue = ONE - sum(shares.values(), ZERO)
    if residue > 0:
        pool = _radd_pool(shares)
        if pool:
            shares, radd = _radd(shares, pool, residue), True
            notes.append("radd applied: remainder of {} back to {}".format(
                residue, ", ".join(pool)))
        else:
            treasury = True
            notes.append("no 'asaba and the only fard-holder is a spouse: the "
                         "remainder of {} goes to the treasury (bayt al-mal)".format(residue))

    result = {
        "shares": shares,
        "blocked": sorted(blocked_by),
        "base": _base(shares),
        "awl_applied": awl,
        "radd_applied": radd,
        "note": "; ".join(notes) if notes else "clean fards, no 'awl, no radd",
    }
    _assert_sums(result, treasury)
    return result


def per_head(result: dict, survivors: dict[str, int]) -> dict[str, Fraction]:
    """Class shares divided by head count, for the people who actually inherit."""
    return {heir: part / max(1, survivors.get(heir, 1))
            for heir, part in result["shares"].items()}


# --- the steps ----------------------------------------------------------------


def _validate(survivors: dict[str, int]) -> dict[str, int]:
    """Unknown keys, negative counts or an empty case are Refused: never guessed."""
    if not survivors:
        raise MirathError("no survivors: nothing to distribute")
    counts: dict[str, int] = {}
    for heir, n in survivors.items():
        if heir not in HEIRS:
            raise MirathError(f"unknown heir class {heir!r}")
        if int(n) != n or int(n) < 0:
            raise MirathError(f"bad head count for {heir}: {n!r}")
        if int(n):
            counts[heir] = int(n)
    for missing in HEIRS:
        counts.setdefault(missing, 0)
    if not any(counts.values()):
        raise MirathError("no survivors: nothing to distribute")
    return counts


def _has_issue(counts: dict[str, int]) -> bool:
    return any(counts[k] for k in _ISSUE_KEYS)


def _hajb(counts: dict[str, int]) -> dict[str, str]:
    """heir -> the blocker's name. The standard Hanafi blocking tree.

    A descendant son shuts out every sibling class and both grand- classes;
    the father shuts out grandfather, grandsons and all siblings. Uterine
    siblings are shut out not by brothers but by any issue or any male
    ascendant (father/grandfather), which is the easy one to get wrong.
    """
    blocked: dict[str, str] = {}
    if counts["father"]:
        for h in ("grandfather", "grandsons", "granddaughters"):
            if counts[h]:
                blocked[h] = "father"
    if counts["grandfather"] and not counts["father"]:
        pass                        # grandfather stands in for father below
    senior_m = counts["sons"] or counts["grandsons"] or counts["father"] \
        or (counts["grandfather"] and not counts["father"])
    if senior_m:
        who = "son" if counts["sons"] else ("grandson" if counts["grandsons"]
                                            else "father" if counts["father"] else "grandfather")
        for h in ("full_brothers", "full_sisters", "consang_brothers",
                  "consang_sisters", "uterine_siblings"):
            if counts[h] and h not in blocked:
                blocked[h] = who
    if _has_issue(counts) or counts["father"] or counts["grandfather"]:
        for h in ("uterine_siblings",):
            if counts[h] and h not in blocked:
                blocked[h] = "descendant or male ascendant"
    return blocked


def _fards(live: dict[str, int]) -> dict[str, Fraction]:
    """The prescribed shares, for heirs that survived :func:`_hajb`.

    Blocked heirs count zero in ``live``, so a sister here is a sister nobody
    shut out. Males present means the females of her own class are
    'asaba-with-him instead of holding a fard, which is why each sister rule
    below checks her brother first.
    """
    f: dict[str, Fraction] = {}
    issue = _has_issue(live)
    if live["husband"]:
        f["husband"] = QUARTER if issue else HALF
    if live["wife"]:
        f["wife"] = EIGHTH if issue else QUARTER
    if live["daughters"] and not live["sons"]:
        f["daughters"] = HALF if live["daughters"] == 1 else TWO_THIRDS
    if live["granddaughters"] and not live["grandsons"] and not live["sons"]:
        if live["daughters"]:
            # a single daughter leaves 1/6 of her 2/3 for the son's daughter
            if live["daughters"] == 1:
                f["granddaughters"] = SIXTH
        else:
            f["granddaughters"] = HALF if live["granddaughters"] == 1 else TWO_THIRDS
    if live["father"]:
        if issue:
            f["father"] = SIXTH          # he takes the 'asaba side too
    if live["grandfather"] and not live["father"] and issue:
        f["grandfather"] = SIXTH
    if live["mother"]:
        siblings = sum(live[k] for k in ("full_brothers", "full_sisters", "consang_brothers", "consang_sisters", "uterine_siblings"))
        f["mother"] = SIXTH if (issue or siblings >= 2) else THIRD
    if live["full_sisters"] and not live["full_brothers"]:
        f["full_sisters"] = HALF if live["full_sisters"] == 1 else TWO_THIRDS
    if live["consang_sisters"] and not live["consang_brothers"] \
            and not (live["full_brothers"] or live["full_sisters"]):
        f["consang_sisters"] = HALF if live["consang_sisters"] == 1 else TWO_THIRDS
    if live["uterine_siblings"]:
        f["uterine_siblings"] = SIXTH if live["uterine_siblings"] == 1 else THIRD
    return f


def _umariyya(live: dict[str, int], fards: dict[str, Fraction], notes: list[str]) -> None:
    """The 'umariyyat: spouse + sisters + mother, no male residuary anywhere.

    Umar reversed two of the usual answers at law. Mother's full third is cut
    to 1/3 of the REMAINDER after the spouse, otherwise she and the sisters
    take the spouse's fard too. The sisters' full fard is then only what that
    leaves: they get less than 2/3 and the base stays whole, so no 'awl.
    """
    spouse = "husband" if fards.get("husband") == HALF else ""
    sisters = next((k for k in ("full_sisters", "consang_sisters")
                    if k in fards and live[k] >= 2), "")
    if not (spouse and sisters and "mother" in fards):
        return
    if _has_issue(live) or live["father"] or live["grandfather"] \
            or any(live[k] for k in _MALES):
        return
    if sum(fards.values(), ZERO) <= ONE:
        return
    fards["mother"] = THIRD * (ONE - fards[spouse])
    fards[sisters] = ONE - fards[spouse] - fards["mother"]
    notes.append("'umariyya: mother 1/3 of the remainder after the spouse ({}), "
                 "{} reduced to {}".format(fards["mother"], sisters, fards[sisters]))


def _apply_awl(fards: dict[str, Fraction], notes: list[str]) -> tuple[bool, dict[str, Fraction]]:
    """Raise the base to the fards' sum when they overflow it: every share shrinks
    by the same factor, so their ratios to one another are untouched.
    """
    total = sum(fards.values(), ZERO)
    if total <= ONE:
        return False, dict(fards)
    notes.append("awl: fards sum to {} over the base, every share scaled by {}".format(total, ONE / total))
    factor = ONE / total
    return True, {h: p * factor for h, p in fards.items()}


def _give_residue(live: dict, ashes: list, residue: Fraction) -> dict[str, Fraction]:
    """The residue ('asaba): nearest class first, males at twice a female.

    Taken in the classical order, not split as a crowd: descendant males, then
    father / grandfather, then brothers. Two rules the tables are strict about
    and a naive per-head split gets wrong: sons shut grandsons out of the
    division, and a father with sons keeps only the sixth he was already given
    - he never takes a second per-head bite beside the son.
    """
    if residue <= ZERO or not ashes:
        return {}
    if live["sons"]:
        return _by_heads(residue, {"sons": 2 * live["sons"],
                                   "daughters": live["daughters"]})
    if live["grandsons"]:
        return _by_heads(residue, {"grandsons": 2 * live["grandsons"],
                                   "granddaughters": live["granddaughters"]})
    if live["father"]:
        return {"father": residue}
    if live["grandfather"]:
        return {"grandfather": residue}
    if live["full_brothers"]:
        return _by_heads(residue, {"full_brothers": 2 * live["full_brothers"],
                                   "full_sisters": live["full_sisters"]})
    if live["consang_brothers"]:
        return _by_heads(residue, {"consang_brothers": 2 * live["consang_brothers"],
                                   "consang_sisters": live["consang_sisters"]})
    return {}


def _by_heads(residue: Fraction, heads: dict) -> dict[str, Fraction]:
    """Split ``residue`` over class head counts already weighted 2:1."""
    total = sum(heads.values())
    if total <= 0:
        return {}
    return {h: residue * Fraction(w) / total for h, w in heads.items() if w}


def _merge(a: dict, b: dict) -> dict[str, Fraction]:
    """Union of two share maps, per-class totals added. Used where the residue
    lands on top of an existing fard, like father's sixth plus his 'asaba.
    """
    out = dict(a)
    for heir, part in b.items():
        out[heir] = out.get(heir, ZERO) + part
    return out


def _radd_pool(shares: dict) -> list:
    """Who may take a radd: current fard-holders that are not a spouse."""
    return sorted(h for h in shares if h not in ("husband", "wife"))


def _radd(shares: dict, pool: list, residue: Fraction) -> dict[str, Fraction]:
    """Send the leftover back to the fard-holders in the ratio of their fards."""
    whole = sum((shares[h] for h in pool), ZERO)
    if whole <= ZERO:
        return dict(shares)
    out = dict(shares)
    for heir in pool:
        out[heir] = shares[heir] + residue * shares[heir] / whole
    return out


def _base(shares: dict) -> int:
    """The 'asl al-masala of the *solved* problem: the denominator the shares run on.

    The classical tables read straight off it - ``'umariyya`` 8, awl on 7 for a
    husband with two full sisters, minbariyya 27 - so it is the least common
    multiple of the FINAL denominators, after 'awl, residue and radd are all in.
    """
    denominators = [p.denominator for p in shares.values() if p]
    if not denominators:
        return 1
    return lcm(*denominators)


def _assert_sums(result: dict, treasury: bool) -> None:
    """Nothing leaves this function with the estate unaccounted for. Either the
    shares sum to exactly one, or we said out loud that the treasury holds the
    rest - a silently wrong total is the one error a fatwa cannot carry.
    """
    total = sum(result["shares"].values(), ZERO)
    if total == ONE:
        return
    if treasury:
        return
    raise MirathError(f"shares sum to {total}, not 1: {result['shares']}")

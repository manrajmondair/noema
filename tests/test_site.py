"""Guard the public site against the two ways it has actually gone wrong.

The first was drift: the page went on documenting an interactive mechanism for weeks
after that mechanism was deleted, because nothing tied its prose to what shipped. The
second was worse — a figure withdrawn as contaminated stayed live because the check
grepped the markup, and the number lived in a data file the markup only fetches.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "demo" / "site"
FIGURE = ROOT / "demo" / "noema.html"

INLINE = re.compile(r"<(script|style)(?![^>]*\bsrc=)[^>]*>(.*?)</\1>", re.S)


@pytest.fixture(scope="module")
def page():
    return (SITE / "index.html").read_text()


def test_every_referenced_asset_exists(page):
    # Deleting app.mjs and leaving the tag behind costs a console error on every visit
    # and, for a stylesheet or a font, a page that renders in a fallback face.
    refs = re.findall(r'(?:src|href)="\./([^"]+)"', page)
    assert refs, "no local asset references found"
    for ref in refs:
        assert (SITE / ref).exists(), f"{ref} is referenced but not present"


def test_the_prose_points_at_sections_that_exist(page):
    # The evidence section moved from §3 to §2 when the mechanism section was cut, and
    # the sentence pointing readers at it kept the old number.
    marks = {int(n) for n in re.findall(r'class="section-mark"><span>(\d+)</span>', page)}
    assert marks, "no numbered sections found"
    for cited in {int(n) for n in re.findall(r"§(\d+)", page)}:
        assert cited in marks, f"prose cites §{cited}; sections present are {sorted(marks)}"


def test_no_withdrawn_figure_is_quoted(page):
    # FALCON held-in R2 0.87 was withdrawn when held-in-minival turned out to be a
    # byte-identical prefix of held-in-calib. It shipped anyway, from the data file.
    for name in ("index.html", "study-data.json"):
        assert "0.87" not in (SITE / name).read_text(), f"{name} quotes a withdrawn figure"


def test_the_site_carries_nothing_inline(page):
    # Keeping the site free of inline script and style is what lets the deployed policy
    # name hashes for the figure alone, and keeps those hashes meaningful.
    assert not INLINE.findall(page)


def test_one_face_carries_the_whole_page():
    css = (SITE / "styles.css").read_text()
    assert css.count("@font-face") == 1
    assert "https://" not in css


@pytest.mark.skipif(not FIGURE.exists(), reason="run demo/export.py first")
def test_the_deployed_policy_admits_the_figure():
    # The figure has to carry its script inline, since it is also handed around as a
    # single file. If staging ever stops naming those hashes the page loads blank.
    import importlib.util

    spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts" / "deploy.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)

    figure = FIGURE.read_text()
    blocks = INLINE.findall(figure)
    assert blocks, "the figure has no inline blocks; the policy assumption changed"
    found = deploy.hashes(figure)
    policy = deploy.policy(figure)
    for kind in ("script", "style"):
        for source in found[kind]:
            assert source in policy, f"{kind} hash missing from the policy"


@pytest.mark.skipif(not (ROOT / "demo" / "assets.json").exists(),
                    reason="run demo/export.py first")
def test_the_quoted_forecast_numbers_match_the_shipped_data(page):
    # Every figure in §1 is a mean over the shipped recordings. Quoting them from memory
    # is how the page came to claim skill "holds" across a horizon that loses half of it.
    calendar = json.loads((ROOT / "demo" / "assets.json").read_text())["calendar"]

    def mean(model, key, step):
        values = [day[model][key][step] for day in calendar]
        return sum(values) / len(values)

    for quoted, value in [("0.118", mean("multistep", "centred", 0)),
                          ("0.051", mean("multistep", "centred", 9)),
                          ("0.429", mean("multistep", "raw", 0)),
                          ("0.494", mean("multistep", "channel_mean", 0))]:
        assert quoted in page, f"{quoted} is no longer quoted on the page"
        assert f"{value:.3f}" == quoted, f"page says {quoted}, data says {value:.3f}"

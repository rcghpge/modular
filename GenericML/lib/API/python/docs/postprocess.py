#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import os

from bs4 import BeautifulSoup
from bs4 import Comment


def remove_h1(soup):
    """Removes the first H1 as it's redundant with the front-matter title."""
    for h in soup.find_all("h1"):
        h.extract()
    return soup


def add_data_anchors(soup):
    """Adds the data-anchor-id used by Quarto for scroll offset on anchor links
    so that they aren't covered by the site header."""
    for dt in soup.find_all("dt"):
        dl = dt.find_parent("dl")
        if dl and dt.has_attr("id") and not dl.has_attr("id"):
            dt["data-anchor-id"] = dt["id"]
            dl["id"] = dt["id"]
            del dt["id"]
    return soup


def remove_empty_dd(soup):
    """Remove empty definition elements that take up vertical space and
    cannot be selected/styled with CSS. Although Sphinx does not add
    whitespace in empty tags, it seems Quarto does, which makes them
    unselectable with the `:empty` pseudo-selector."""
    for dd in soup.find_all("dd"):
        # Check it has no raw text (either string vals or child tags)
        if not dd.text:
            dd.replace_with(Comment("No definition provided."))
    return soup


def add_quarto_include(content):
    """Add Quarto include command as raw text because bs4 deals only with
    DOM trees and Sphinx/Jinja always escape the brackets."""
    include = "{{< include /_preview-notice-engine.qmd >}}\n\n"
    pos = content.find('<div class="sphinx-docs">')
    return content[:pos] + include + content[pos:]


def process(file):
    """Processes an individual file."""
    print("Post-processing " + file)
    with open(file, "r+") as f:
        soup = BeautifulSoup(f, "html.parser")
        soup = remove_h1(soup)
        soup = add_data_anchors(soup)
        soup = remove_empty_dd(soup)
        content = f"{str(soup)}\n"
        content = add_quarto_include(content)
        f.seek(0)
        f.write(content)
        f.truncate()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dir", "-d", required=True, help="Path to directory of files."
    )
    args = parser.parse_args()

    # Accept a directory
    if os.path.isdir(args.dir):
        for dirpath, _, filenames in os.walk(args.dir):
            for file in filenames:
                if os.path.splitext(file)[1] == ".md":
                    process(os.path.join(dirpath, file))


if __name__ == "__main__":
    main()

"""Clean the TCGA pathology-report text column."""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Sequence


TEXT_COLUMN = "text"

FIELD_RULE_SPECS = [('\n'
  '        \\b(?:patient\\s+)?age\n'
  '        (?:\\s+at\\s+(?:diagnosis|presentation|surgery|collection))?\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        '
  '(?:\\d{1,3}(?:\\.\\d+)?\\s*(?:years?|yrs?|y/?o)?|unknown|not\\s+available|n/?a)\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:date\\s+of\\s+birth|birth\\s*date|d\\.?o\\.?b\\.?)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        [A-Za-z0-9,./ -]{4,40}\n'
  '        (?=$|[.;])\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:biologic(?:al)?\\s+)?(?:sex|gender)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:female|male|woman|man|girl|boy|unknown|not\\s+available|n/?a)\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b\n'
  '        (?:(?:fuhrman|nottingham|bloom(?:-richardson)?|who|figo)\\s+)?\n'
  '        (?:(?:histologic(?:al)?|tumou?r|nuclear)\\s+)?\n'
  '        grad(?:e|ing)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:\n'
  '            (?:g(?:rade)?\\s*)?[1-4]\n'
  '            |[iv]{1,4}\n'
  '            |(?:low|intermediate|high)(?:[- ]grade)?\n'
  '            |well|moderate(?:ly)?|poor(?:ly)?\n'
  '            |unknown|not\\s+available|n/?a\n'
  '        )\n'
  '        (?:\\s*(?:/|of)\\s*(?:3|4|iii|iv))?\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\bgleason\\s+(?:grade|pattern|score)?\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        \\d(?:\\s*\\+\\s*\\d)?(?:\\s*=\\s*\\d{1,2})?\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:pathologic(?:al)?|pathology|path)?\\s*\n'
  '        t(?:umou?r)?\\s*(?:category|classification|stage)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:[ycra]?p?t(?:is|x|0|[1-4])(?:[a-d])?|unknown|not\\s+available|n/?a)\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:pathologic(?:al)?|pathology|path)?\\s*\n'
  '        n(?:odal|ode)?\\s*(?:category|classification|stage)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:[ycra]?p?n(?:x|[o0]|[1-3])(?:[a-c])?|unknown|not\\s+available|n/?a)\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:pathologic(?:al)?|pathology|path)?\\s*\n'
  '        m(?:etastasis)?\\s*(?:category|classification|stage)\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:[ycra]?p?m(?:x|0|1)(?:[a-c])?|unknown|not\\s+available|n/?a)\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:ajcc\\s+)?(?:pathologic(?:al)?\\s+)?(?:tumou?r\\s+)?\n'
  '        stage(?:\\s+group)?\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:stage\\s*)?(?:0|[1-4]|i{1,3}|iv)(?:[a-d])?\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            mantis(?:\\s+(?:msi\\s+)?score)?\n'
  '            |msi(?:\\s+(?:score|status|index))?\n'
  '            |microsatellite\\s+instability(?:\\s+(?:score|status|index))?\n'
  '        )\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        (?:\n'
  '            \\d+(?:\\.\\d+)?\n'
  '            |high|low|stable|unstable|indeterminate\n'
  '            |msi[- ]?h|msi[- ]?l|mss\n'
  '            |unknown|not\\s+available|n/?a\n'
  '        )\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:tmb|tumou?r\\s+mutational\\s+burden|\n'
  '        non[- ]?synonymous\\s+(?:tmb|mutational\\s+burden))\n'
  '        \\s*(?:score|status|is|was|:|=|-)*\\s*\n'
  '        (?:\n'
  '            \\d+(?:\\.\\d+)?(?:\\s*(?:mutations?|muts?)(?:\\s*/\\s*mb)?)?\n'
  '            |high|low|intermediate|unknown|not\\s+available|n/?a\n'
  '        )\n'
  '        \\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            tumou?r\\s+(?:type|histology|classification)\n'
  '            |histologic(?:al)?\\s+(?:type|diagnosis|classification)\n'
  '            |morphologic(?:al)?\\s+(?:type|diagnosis|classification)\n'
  '            |neoplasm\\s+(?:type|classification)\n'
  '        )\n'
  '        \\s*(?:is|was|:|=|-)\\s*\n'
  '        [^.;\\r\\n]{1,220}\n'
  '        (?=$|[.;\\r\\n])\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:(?:molecular|intrinsic|genomic|tcga|pam50)\\s+)?\n'
  '        sub[- ]?type\n'
  '        \\s*(?:is|was|:|=|-)\\s*\n'
  '        [^.;\\r\\n]{1,160}\n'
  '        (?=$|[.;\\r\\n])\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            overall\\s+survival|survival(?:\\s+time)?|\n'
  '            follow[- ]?up(?:\\s+(?:time|duration))?|\n'
  '            time\\s+to\\s+(?:death|last\\s+follow[- ]?up)\n'
  '        )\n'
  '        \\s*(?:is|was|:|=|-)?\\s*\n'
  '        \\d+(?:\\.\\d+)?\\s*(?:days?|weeks?|months?|years?|yrs?)\n'
  '        \\b\n'
  '        ',
  98)]
CLAUSE_RULE_SPECS = [('\\b(?:age|aged|date\\s+of\\s+birth|birth\\s*date|d\\.?o\\.?b\\.?|year[- ]old)\\b',
  98),
 ('\n'
  '        \\b(?:\n'
  '            sex|gender|female|male|woman|man|girl|boy|\n'
  '            she|her|hers|herself|he|him|his|himself|\n'
  '            breast|mammary|uterus|uterine|endometrium|endometrial|\n'
  '            cervix|cervical|ovary|ovarian|fallopian|vagina|vaginal|\n'
  '            vulva|vulvar|prostate|prostatic|testis|testicular|\n'
  '            seminal\\s+vesicle|penis|penile\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            grade|grading|gleason|fuhrman|nottingham|broders|\n'
  '            differentiated|undifferentiated\n'
  '        )\\b\n'
  '        ',
  98),
 ('\\b(?:ajcc|uicc)\\b|\\bfigo\\b.{0,40}\\b(?:stage|staging)\\b', 98),
 ('\\b(?:msi|mss|mantis|microsatellite|mismatch[- ]repair|dmmr|pmmr)\\b', 98),
 ('\n'
  '        \\b(?:\n'
  '            tumou?r\\s+(?:type|histology|classification)|\n'
  '            histologic(?:al)?\\s+(?:type|diagnosis|classification)|\n'
  '            morphologic(?:al)?\\s+(?:type|diagnosis|classification)\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            adenocarcinoma|adeno[- ]?squamous\\s+carcinoma|carcinom[a-z]*|\n'
  '            adenokarzinom[a-z]*|karzinom[a-z]*|\n'
  '            sarcoma|glioblastoma|glioma|astrocytoma|oligodendroglioma|\n'
  '            ependymoma|medulloblastoma|melanoma|mesothelioma|lymphoma|\n'
  '            leuka?emia|myeloma|neuroblastoma|neuroendocrine\\s+tumou?r|\n'
  "            germ[- ]cell\\s+tumou?r|phyllodes\\s+tumou?r|wilms'?\\s+tumou?r|\n"
  '            oncocytoma|meningioma|cholangiocarcinoma|hepatoblastoma\n'
  '        )s?\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            clear[- ]cell|papillary|squamous(?:[- ]cell)?|urothelial|\n'
  '            transitional[- ]cell|ductal|lobular|endometrioid|serous|\n'
  '            mucinous|micropapillary|intestinal[- ]type|diffuse[- ]type|\n'
  "            lauren(?:'s)?\\s+(?:intestinal|diffuse|mixed)|\n"
  '            typus\\s+intestinalis|intestinalis\\s+(?:sec\\s+)?lauren\n'
  '        )\\b\n'
  '        ',
  98),
 ('\\b(?:cancer(?:ous)?|malignan(?:t|cy|cies)|neoplasm)s?\\b', 98),
 ('\n'
  '        \\b(?:\n'
  '            pam50|luminal\\s*[ab]|basal[- ]like|her2[- ]enriched|\n'
  '            triple[- ]negative|estrogen\\s+receptor|progesterone\\s+receptor|\n'
  '            er[- ]positive|er[- ]negative|pr[- ]positive|pr[- ]negative|\n'
  '            her2(?:/neu)?[- ]positive|her2(?:/neu)?[- ]negative|\n'
  '            idh(?:1|2)?[- ]?(?:mutant|mutation|wild[- ]?type|wt)|\n'
  '            1p\\s*/\\s*19q|co[- ]?deletion|\n'
  '            pole[- ]?(?:mutant|mutation|ultramutated)|\n'
  '            epstein[- ]barr|ebv[- ]positive|ebv[- ]negative|\n'
  '            genomically\\s+stable|chromosomal\\s+instability|\n'
  '            copy[- ]number\\s+(?:high|low)|\n'
  '            hpv[- ]positive|hpv[- ]negative|\n'
  '            p53[- ]abnormal|tp53[- ]mutant|tp53\\s+mutation\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            sub[- ]?type|pam50|idh[- ]?[12]?|1p\\s*/\\s*19q|\n'
  '            her2(?:/neu)?|estrogen\\s+receptor|progesterone\\s+receptor|\n'
  '            triple[- ]negative|epstein[- ]barr|\\bebv\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            survival|follow[- ]?up|\n'
  '            last\\s+known\\s+alive|vital\\s+status|\n'
  '            alive|died|deceased|expired|death\n'
  '        )\\b\n'
  '        ',
  98)]
ATOMIC_RULE_SPECS = [('\\b\\d{1,3}(?:\\.\\d+)?[- ](?:year|yr)s?[- ]old\\b', 98),
 ('\\b\\d{1,3}(?:\\.\\d+)?\\s*(?:y/?o|yo)\\b', 98),
 ('\\baged\\s+\\d{1,3}(?:\\.\\d+)?(?:\\s*years?)?\\b', 98),
 ('\\b(?:female|male|woman|man|girl|boy)\\b', 98),
 ('\\b(?:she|her|hers|herself|he|him|his|himself)\\b', 98),
 ('\n'
  '        \\b(?:\n'
  '            breast|mammary|uterus|uterine|endometrium|endometrial|\n'
  '            cervix|cervical|ovary|ovarian|fallopian|vagina|vaginal|\n'
  '            vulva|vulvar|prostate|prostatic|testis|testicular|\n'
  '            seminal\\s+vesicle|penis|penile\n'
  '        )\\b\n'
  '        ',
  98),
 ('(?<![A-Za-z0-9])g[1-4](?![A-Za-z0-9])', 98),
 ('\\b(?:grade\\s*)?(?:[1-4]|i{1,3}|iv)\\s*(?:/|of)\\s*(?:3|4|iii|iv)\\b', 98),
 ('\n'
  '        \\b(?:\n'
  '            low[- ]grade|intermediate[- ]grade|high[- ]grade|\n'
  '            well[- ]differentiated|moderately[- ]differentiated|\n'
  '            poorly[- ]differentiated|differentiated|undifferentiated\n'
  '        )\\b\n'
  '        ',
  98),
 ('(?<![A-Za-z0-9])(?:[ycra]?p?t)(?:is|x|0|[1-4])(?:[a-d])?(?![A-Za-z0-9])', 98),
 ('\n'
  '        (?<![A-Za-z0-9])(?:\n'
  '            (?:[ycra]?p?n)(?:x|0|[1-3])(?:[a-c])?\n'
  '            |(?:[ycra]?pn)o\n'
  '        )\n'
  '        (?:\\s*\\([^)]{1,40}\\))?(?![A-Za-z0-9])\n'
  '        ',
  98),
 ('(?<![A-Za-z0-9])(?:[ycra]?p?m)(?:x|0|1)(?:[a-c])?(?![A-Za-z0-9])', 98),
 ('\n'
  '        \\b(?:ajcc\\s+)?(?:pathologic(?:al)?\\s+)?(?:tumou?r\\s+)?\n'
  '        stage(?:\\s+group)?\\s*\n'
  '        (?:0|[1-4]|i{1,3}|iv)(?:[a-d])?\\b\n'
  '        ',
  98),
 ("\\bdukes(?:'|’)?\\s+(?:a|b[12]?|c[12]?|d)\\b", 98),
 ('\\b(?:astler|atler)[- ]coller\\s+(?:a|b[12]?|c[12]?|d)\\b', 98),
 ('\n'
  '        \\b(?:\n'
  '            msi[- ]?[hl]|mss|\n'
  '            microsatellite[- ]stable|microsatellite[- ]unstable|\n'
  '            microsatellite\\s+instability|\n'
  '            mismatch[- ]repair\\s+(?:deficient|proficient)|\n'
  '            dmmr|pmmr|mantis\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            tmb[- ]?(?:high|low|intermediate)|\n'
  '            tumou?r\\s+mutational\\s+burden|\n'
  '            \\d+(?:\\.\\d+)?\\s*(?:mutations?|muts?)\\s*/\\s*mb\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            (?:brca|ucec|lgg|stad|coadread|blca|hnsc|kirc|kirp|luad|lusc)\n'
  '            [_-][A-Za-z0-9_.+-]+|\n'
  '            luminal\\s*[ab]|basal[- ]like|her2[- ]enriched|normal[- ]like|\n'
  '            proneural|mesenchymal|classical\\s+subtype|neural\\s+subtype\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b\\d+(?:\\.\\d+)?\\s*(?:days?|weeks?|months?|years?|yrs?)\n'
  '        \\s+(?:of\\s+)?(?:overall\\s+)?survival\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:followed|follow[- ]?up)\\s+(?:for\\s+)?\n'
  '        \\d+(?:\\.\\d+)?\\s*(?:days?|weeks?|months?|years?|yrs?)\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            died|deceased|expired|alive|\n'
  '            death\\s+from|death\\s+due\\s+to|\n'
  '            last\\s+known\\s+alive|vital\\s+status\n'
  '        )\\b\n'
  '        ',
  98),
 ('\n'
  '        \\b(?:\n'
  '            (?:19|20)\\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\\d|3[01])\n'
  '            |(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\\d|3[01])[-/.](?:19|20)?\\d{2}\n'
  '            |(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|\n'
  '              jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|\n'
  '              nov(?:ember)?|dec(?:ember)?)\n'
  '              \\s+\\d{1,2}(?:st|nd|rd|th)?[,]?\\s+(?:19|20)\\d{2}\n'
  '        )\\b\n'
  '        ',
  98)]
CLEANUP_RULE_SPECS = [('\\b(?:patient\\s+)?age\\s*(?:at\\s+(?:diagnosis|presentation))?\\s*[:=]\\s*(?=$|[.;])',
  98),
 ('\\b(?:biologic(?:al)?\\s+)?(?:sex|gender)\\s*[:=]\\s*(?=$|[.;])', 98),
 ('\n'
  '        \\b(?:(?:fuhrman|nottingham|bloom(?:-richardson)?|who|figo)\\s+)?\n'
  '        (?:(?:histologic(?:al)?|tumou?r|nuclear)\\s+)?\n'
  '        grad(?:e|ing)\\s*[:=]\\s*(?=$|[.;])\n'
  '        ',
  98),
 ('\\b(?:tnm|pathologic(?:al)?\\s+stage)\\s*[:=]\\s*(?=$|[.;])', 98),
 ('\n'
  '        \\b(?:\n'
  '            tumou?r\\s+(?:type|histology|classification)|\n'
  '            histologic(?:al)?\\s+(?:type|diagnosis|classification)\n'
  '        )\\s*[:=]\\s*(?=$|[.;])\n'
  '        ',
  98),
 ('\\b(?:(?:molecular|intrinsic|genomic|tcga|pam50)\\s+)?sub[- '
  ']?type\\s*[:=]\\s*(?=$|[.;])',
  98)]


def compile_patterns(specs):
    return tuple(re.compile(pattern, flags) for pattern, flags in specs)


FIELD_RULES = compile_patterns(FIELD_RULE_SPECS)
CLAUSE_RULES = compile_patterns(CLAUSE_RULE_SPECS)
ATOMIC_RULES = compile_patterns(ATOMIC_RULE_SPECS)
CLEANUP_RULES = compile_patterns(CLEANUP_RULE_SPECS)
COMBINED_TNM_RE = re.compile(*('\n'
 '    (?<![A-Za-z0-9])\n'
 '    (?P<t>(?:[ycra]?p?t)(?:is|x|0|[1-4])(?:[a-d])?)\n'
 '    \\s*[,/ -]?\\s*\n'
 '    (?P<n>(?:[ycra]?p?n)(?:x|0|[1-3])(?:[a-c])?)\n'
 '    (?:\\s*[,/ -]?\\s*\n'
 '       (?P<m>(?:[ycra]?p?m)(?:x|0|1)(?:[a-c])?)\n'
 '    )?\n'
 '    (?![A-Za-z0-9])\n'
 '    ',
 98))
SEGMENT_SPLIT_RE = re.compile(*('(?<=[.;!?])\\s+', 32))
WHITESPACE_RE = re.compile(*('\\s+', 32))


def apply_rules(text, rules):
    for pattern in rules:
        text = pattern.sub(" ", text)
    return text


def clean_text(text):
    text = unicodedata.normalize("NFKC", text or "").replace("\x00", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = apply_rules(text, FIELD_RULES)
    text = COMBINED_TNM_RE.sub(" ", text)
    text = " ".join(
        segment
        for segment in SEGMENT_SPLIT_RE.split(text)
        if not any(pattern.search(segment) for pattern in CLAUSE_RULES)
    )
    text = apply_rules(text, ATOMIC_RULES)
    text = apply_rules(text, CLEANUP_RULES)
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r":\s*([.;])", r"\1", text)
    text = re.sub(r"([.;])(?:\s*[.;])+", r"\1", text)
    text = re.sub(r"(?:(?<=^)|(?<=[.;]))\s*[,;:]+\s*", " ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip(" \t\r\n,;:")


def set_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def clean_reports(input_csv: Path, output_csv: Path) -> None:
    set_csv_field_size_limit()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        if TEXT_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Text column {TEXT_COLUMN!r} not found. "
                f"Available columns: {reader.fieldnames}"
            )

        with output_csv.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=reader.fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                row[TEXT_COLUMN] = clean_text(row.get(TEXT_COLUMN) or "")
                writer.writerow(row)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    clean_reports(args.input_csv.resolve(), args.output_csv.resolve())


if __name__ == "__main__":
    main()

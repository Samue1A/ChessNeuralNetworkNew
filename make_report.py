from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── styles ──────────────────────────────────────────────────────────────────
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

def h2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

def h3(text):
    doc.add_heading(text, level=3)

def body(text):
    doc.add_paragraph(text)

def math_block(text):
    """Add an indented paragraph for equations (plain-text approximation)."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.6)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(10)

def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.style = doc.styles["No Spacing"]
    run = p.add_run(text)
    run.font.name  = "Courier New"
    run.font.size  = Pt(9)
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x80)

def add_table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light List Accent 1"
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        hrow.cells[i].text = h
        hrow.cells[i].paragraphs[0].runs[0].bold = True
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            t.rows[r_i + 1].cells[c_i].text = str(val)
    doc.add_paragraph()

# ── Title ────────────────────────────────────────────────────────────────────
title = doc.add_heading("Do Rising House Prices Make Local Workers Worse Off?", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph("A Two-Way Fixed Effects Panel Analysis of English Local Authorities, 2015–2018")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(12)
sub.runs[0].font.italic = True

doc.add_paragraph()

# ── 1. Introduction ──────────────────────────────────────────────────────────
h1("1. Introduction")
body(
    "UK housing affordability is one of the defining economic issues of the past two decades. "
    "Headlines routinely describe house prices 'outpacing wages,' but a sharper version of that "
    "question is: when house prices in a particular town rise, do the people who actually live and "
    "work there end up worse off in real terms?"
)
body(
    "This project tries to answer one slice of that question with public ONS data. The outcome of "
    "interest is not the house price itself, but the wage-to-rent ratio — roughly, how much of a "
    "month's rent a typical worker's pay covers. If house prices rise and the wage-to-rent ratio in "
    "the same place falls in the same year, that is at least suggestive that local workers' "
    "purchasing power over housing is being squeezed."
)
body("The research question is:")
p = doc.add_paragraph(style="No Spacing")
p.paragraph_format.left_indent = Inches(0.5)
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after  = Pt(6)
run = p.add_run(
    "Within a UK local authority, does an increase in house prices cause workers' wages "
    "(relative to rents) to fall?"
)
run.font.italic = True
body(
    "The headline finding is that within a local authority, a 10% rise in house prices in a given "
    "year is associated with a roughly 4.1% fall in the local wage-to-rent ratio in that same year. "
    "The estimate is highly statistically significant, but — as discussed at length later — the "
    "design has real limitations, and 'associated with' is doing a lot of work in that sentence."
)

# ── 2. Data ──────────────────────────────────────────────────────────────────
h1("2. Data")
body("All primary data sources are public ONS releases.")

add_table(
    ["Variable", "Source", "Unit", "Notes"],
    [
        ["Median gross weekly pay", "ASHE (via Nomis)", "£ per week per LA per year", "Workplace-based"],
        ["Median house sale price", "ONS House Prices by Local Authority", "£ per LA per year", ""],
        ["Median monthly private rent", "VOA Private Rental Market Statistics", "£ per LA per year", ""],
        ["LA-to-LA migration flows", "ONS Internal Migration matrices", "Persons per year", "Second regression spec"],
        ["Mid-year population", "ONS Mid-Year Population Estimates", "Persons per LA per year", "Migration rate denominator"],
    ]
)

body(
    "After inner-joining the three core datasets on local authority code and year, the analytical "
    "panel contains:"
)
for item in [
    "284 English local authority districts",
    "4 years: 2015–2018",
    "1,136 observations (284 × 4, balanced)",
]:
    doc.add_paragraph(item, style="List Bullet")

body(
    "The panel is restricted to 2015 onwards because the ONS median house price series has sparse "
    "LA-level coverage before then. A planned second specification using migration flows (REG 2) was "
    "dropped because the 2017–2018 ONS internal migration spreadsheets used a different XLSX layout "
    "that the parser could not reconcile. Roughly 96 LAs were lost in the merge due to LA code "
    "mismatches — a recurring nuisance with ONS data because local authority boundaries are "
    "periodically reorganised."
)

h2("2.1  Outcome variable")
body(
    "The dependent variable is the log of an approximate monthly wage-to-rent ratio:"
)
math_block("Y(i,t)  =  log( [median_weekly_pay(i,t) × 4.33]  /  median_monthly_rent(i,t) )")
body(
    "The factor 4.33 is the average number of weeks in a month (52 ÷ 12), converting weekly pay "
    "to a monthly figure. Taking logs makes the variable roughly symmetric and allows the regression "
    "coefficient to be interpreted as a percentage change. When Y falls, workers in that LA are "
    "worse off relative to housing costs — their pay buys less rent."
)

h2("2.2  Regressor")
math_block("X(i,t)  =  log( median_house_sale_price(i,t) )")
body(
    "Using logs means a 0.1-unit change in X corresponds to approximately a 10% change in house "
    "prices, making the coefficient directly interpretable as an elasticity."
)

# ── 3. Methodology ───────────────────────────────────────────────────────────
h1("3. Methodology")

h2("3.1  The model")
body("We fit a two-way fixed effects (TWFE) panel regression:")
math_block("Y(i,t)  =  β · X(i,t)  +  α(i)  +  γ(t)  +  ε(i,t)")
body(
    "where i indexes the local authority and t the year. The three key components are:"
)
for item in [
    "α(i)  —  local authority fixed effect: a separate intercept for every LA that soaks up "
    "everything permanently different about that place (geography, industrial legacy, university "
    "presence). We do not estimate what those things are; we simply prevent them from contaminating β.",
    "γ(t)  —  year fixed effect: soaks up anything that affected all LAs in a given year — "
    "Brexit uncertainty, Bank of England rate changes, the National Living Wage.",
    "β  —  the parameter of interest, identified only from within-LA, within-year deviations.",
]:
    doc.add_paragraph(item, style="List Bullet")

body(
    "In plain English: β answers the question 'in years when house prices in (say) Manchester are "
    "unusually high relative to Manchester's own typical level and relative to the national average "
    "that year, is the wage-to-rent ratio in Manchester also unusually low?'"
)

h2("3.2  Estimation: the within transformation")
body(
    "Rather than including a dummy variable for every LA and every year, we use the Frisch-Waugh-Lovell "
    "within transformation. For every variable, subtract the unit mean and the time mean, then add "
    "back the grand mean:"
)
math_block("X_tilde(i,t)  =  X(i,t)  −  X_bar(i)  −  X_bar(t)  +  X_bar")
body(
    "Running plain OLS on the transformed variables is numerically identical to the full "
    "dummy-variable regression. In Python:"
)
code_block(
    "def two_way_demean(df, var, unit='la_code', time='year'):\n"
    "    grand     = df[var].mean()\n"
    "    unit_mean = df.groupby(unit)[var].transform('mean')\n"
    "    time_mean = df.groupby(time)[var].transform('mean')\n"
    "    return df[var] - unit_mean - time_mean + grand\n\n"
    "panel['y_tilde'] = two_way_demean(panel, 'log_wage_rent')\n"
    "panel['x_tilde'] = two_way_demean(panel, 'log_house_price')\n\n"
    "beta = (panel['x_tilde'] * panel['y_tilde']).sum() \\\n"
    "     / (panel['x_tilde'] ** 2).sum()"
)
doc.add_paragraph()
body(
    "This is also why the scatter plot appears as a ball clustered around (0, 0): after removing "
    "each LA's own average and each year's national average, all remaining variation is small "
    "deviations around zero. The red regression line through that cloud is β."
)

h2("3.3  Standard errors: clustering at the LA level")
body(
    "Observations from the same LA across different years are not independent — if an LA has a "
    "structurally noisy labour market, all four of its residuals tend to be large together. "
    "We use cluster-robust standard errors (Liang-Zeger CR1), which sum variance contributions "
    "at the LA-block level rather than the individual observation level. This allows within-LA "
    "errors to be arbitrarily correlated over time."
)

# ── 4. Results ───────────────────────────────────────────────────────────────
h1("4. Results")

add_table(
    ["Quantity", "Value"],
    [
        ["Coefficient β (log house price)", "−0.4104"],
        ["Cluster-robust SE (CR1, by LA)", "0.0700"],
        ["t-statistic", "−5.862"],
        ["p-value", "6.5 × 10⁻⁹"],
        ["95% confidence interval", "[−0.548, −0.273]"],
        ["Within-R²", "0.0578"],
        ["Observations", "1,136"],
        ["Local authorities", "284"],
        ["Years", "4 (2015–2018)"],
    ]
)

h2("4.1  Interpretation")
body(
    "Because both variables are in logs, the coefficient is an elasticity. The point estimate of "
    "−0.41 says: within a given local authority, a 10% above-trend house price year is associated "
    "with a roughly 4.1% below-trend wage-to-rent ratio in that LA that year. 'Above-trend' means "
    "above what we would predict for that LA given (a) its own typical level and (b) the national "
    "average that year."
)

h2("4.2  Statistical strength")
body(
    "The t-statistic of −5.9 and p-value of approximately 6.5 × 10⁻⁹ (roughly 1-in-150-million "
    "odds of seeing this by chance if the true effect were zero) are well past any conventional "
    "significance threshold. The 95% confidence interval [−0.548, −0.273] does not come close to "
    "including zero."
)

h2("4.3  The low R²")
body(
    "The within-R² of 5.8% simply means house prices are not the dominant driver of short-run "
    "swings in the wage-to-rent ratio within a single LA. The fixed effects do most of the "
    "explanatory work — the bulk of variation in Y is between LAs and across national trends, not "
    "within a given LA over four years. This does not undermine the finding; it just means house "
    "prices are one factor among several."
)

# ── 5. Limitations ───────────────────────────────────────────────────────────
h1("5. Limitations")

limitations = [
    (
        "Only four years.",
        "TWFE needs time variation to work. Four years gives very little. Any dynamic that takes "
        "longer than four years to play out — which most housing-market processes do — is partially "
        "absorbed by the LA fixed effect rather than identified as an effect of X."
    ),
    (
        "Parallel trends is untestable.",
        "TWFE is causal under the assumption that, absent the change in X, Y would have evolved "
        "the same way across LAs. With four years and no pre-period, this assumption cannot be "
        "inspected or tested."
    ),
    (
        "Reverse causality.",
        "If a local economy weakens, investors may still bid up housing as a store of value, or "
        "squeezed locals are pushed into renting and rents rise. In that scenario Y is causing X, "
        "not the reverse. Without an instrumental variable for house prices — e.g. a national "
        "interest rate shock interacted with local land-supply elasticity — this cannot be ruled out."
    ),
    (
        "The outcome bundles two stories.",
        "A falling wage-to-rent ratio could be wages falling, rents rising, or both. The mechanism "
        "'housing costs crowd out workers' versus 'housing costs suppress wage growth' is genuinely "
        "different, and this setup cannot separate them."
    ),
    (
        "Migration specification was dropped.",
        "The planned second regression using net in-migration as an outcome had to be cut because "
        "the 2017–2018 ONS migration XLSX files use a different layout that the parser could not "
        "reconcile, leaving only 18–20 LAs for those years."
    ),
    (
        "96 LAs lost in the merge.",
        "Dropped LAs are disproportionately recently reorganised authorities. If they are "
        "systematically different — more rural, more coastal — the sample is non-random and results "
        "may not generalise to all of England."
    ),
    (
        "England only.",
        "Wales has thin coverage across the datasets; Scotland and Northern Ireland are entirely "
        "absent. Generalising beyond England requires caution."
    ),
    (
        "House prices may be a proxy.",
        "The coefficient may partly capture other features of a booming local economy — tech "
        "clusters, gentrification, rapid population inflows — rather than the causal effect of "
        "housing costs specifically."
    ),
]

for title_text, detail in limitations:
    p = doc.add_paragraph(style="List Number")
    run_bold = p.add_run(title_text + "  ")
    run_bold.bold = True
    p.add_run(detail)

# ── 6. Conclusion ────────────────────────────────────────────────────────────
h1("6. Conclusion")
body(
    "Using a balanced panel of 284 English local authorities over 2015–2018, a two-way fixed "
    "effects regression of log wage-to-rent ratio on log house price yields:"
)
math_block("β_hat  =  −0.41   (clustered SE = 0.07,   p ≈ 6.5 × 10⁻⁹)")
body(
    "Within a local authority, a year of unusually high house prices is, on average, also a year "
    "of an unusually low wage-to-rent ratio — approximately 0.41% per 1% of house prices. The sign "
    "is consistent with the intuition that rising local house prices hurt local workers in real "
    "housing terms; the magnitude is economically meaningful without being all-explaining "
    "(within-R² ≈ 6%)."
)
body(
    "The result should be read as a clean conditional correlation, not a settled causal estimate. "
    "Four years of data, an untestable parallel-trends assumption, plausible reverse causality, and "
    "an outcome variable that mixes wage and rent dynamics all temper what can be claimed."
)
body(
    "Natural next steps: extend the panel back to the early 2000s for more time variation; "
    "instrument house prices with a supply-side measure such as Saiz topographic land-supply "
    "elasticity interacted with national interest rates; and fix the migration parser to bring "
    "the second regression back in."
)
body(
    "For now: in the short run, within England, places where house prices rise are also places "
    "where wages stretch less far over rent. Whether that is causation, common cause, or both "
    "remains genuinely open."
)

# ── Save ─────────────────────────────────────────────────────────────────────
out = "output/UK_Housing_Panel_Report.docx"
import os; os.makedirs("output", exist_ok=True)
doc.save(out)
print(f"Saved: {out}")

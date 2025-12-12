import daft

from jotunn import TextSegmentation

df = daft.from_pydict(
    {
        "text": [
            """
            Peter J. BRENNAN, Secretary of Labor, United States Department of Labor, Plaintiff-Appellee, v. SIX FLAGS OVER GEORGIA, LTD., and Great Southwest Atlanta Corp., Defendants-Appellants.
            No. 72-2990.
            United States Court of Appeals, Fifth Circuit.
            Argued Feb. 7, 1973.
            Decided Feb. 22, 1973.
            Rehearing and Rehearing En Banc Denied April 10, 1973.
            Arthur J. Riggs, Dallas, Tex., Charles Kelso, Atlanta, Ga., for defendants-appellants.
            Richard F. Schubert, Sol. of Labor, Jacob I. Karro, Atty., U. S. Dept, of Labor, Washington, D. C., Beverley R. Worrell, Regional Sol., James H. Wood-son, Atty., Atlanta, Ga., Carin Ann Clauss, Donald S. Shire, U. S. Dept, of Labor, Office of Sol., Washington, D. C., for plaintiff-appellee.
            Before ALDRICH SIMPSON and CLARK, Circuit Judges.


            Hon. Bailey Aldrich, Senior Circuit Judge of the First Circuit, sitting by designation.

            PER CURIAM:

            The Secretary of Labor brought this action against Six Flags over Georgia, Ltd., and Great Southwest Atlanta Corp., hereinafter, collectively, GSA, to enjoin alleged violations of the Fair Labor Standards Act and to compel the payment of overtime to certain employees. On a stipulated record both parties moved for summary judgment. The court ruled in favor of the Secretary and GSA appeals.

            GSA operates an amusement park at Atlanta, Georgia of very substantial size. During the season it has over 1600 employees, sharply reduced during the off-season, of which some 100 are engaged in maintenance and repairs. From year to year the park remodels, and erects new structures, new buildings, and new places of entertainment, in order to maintain public interest. This new construction is mostly done during the off-season, much of the work being contracted out. During the off-season GSA pays all of its employees wages within the act, but during the season it does not, by virtue of the exemption afforded by 29 U.S.C. § 213(a)(3) to amusement and recreational establishments. During the season a small amount of new work is done by what are normally maintenance employees. It is for this that the Secretary asserts liability.

            GSA concedes, as it must, that a general contractor employed by it to do this construction work would not be conducting a recreational establishment. GSA is in no better position. It is the character of the work, not the source of the remuneration, that controls. Cf. Hodgson v. Colonnades, Inc., 5 Cir., 472 F.2d 42 (1973) (fact employees “changing the premises” are on hotel pay-roll irrelevant). The nature of the work is what gives rise to the need for an exemption ; the exemption is not a subsidy accorded to an employer because of his principal activities.

            Nor does it make any difference that the employee is doing mixed work. In any week that any particular employee does some non-exempt work he is covered fully, not pro rata. Hodgson v. Wittenburg Livestock Co., 5 Cir., 1972, 464 F.2d 1219; Mitchell v. Hunt, 5 Cir., 1959, 263 F.2d 913. GSA says that this makes an accounting problem for it. With the. legislation on the books, the problem is of its own making.

            Affirmed.
            """
        ],
    }
)

segmenter = TextSegmentation()
df = segmenter(df)
df.show()

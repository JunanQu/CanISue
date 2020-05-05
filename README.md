# Can I Sue
Team: Max Chen, Nikhil Saggi, Zachary Shine, Junan Qu, Ian Paul


Can I Sue is an application intended to help users make more informed legal decisions. Can I Sue leverages data from Harvard's Caselaw Access Project API (CAPAPI) as well as Reddit's legaladvice' subreddit (r/legaladvice). The cases retrieved from CAPAPI are meant to provide Can I Sue's users with insight into what sort of legal precedences have been set by past lawsuits, whereas r/legaladvice is meant to provide users with a means of seeking out helpful discussions that people with similar experiences have had.

In the "Search Cases" bar, the user may describe their legal situation or grievance. The more details that are provided, the more likely that Can I Sue is able to return highly relevant cases and discussions. For example, one may search:
"My neighbor has built a fence on my property." as well as: "fence built on my property"
although the former will retrieve results that are likely a bit more relevant.

There is also a drop-down calendar where the user can select the "Earliest Case Date." By default, Can I Sue will retrieve the most relevant cases from CAPAPI from any possible point in time, but if the user elects to set an "Earliest Case Date", Can I Sue will set a lower bound on the date of cases returned. This may be helpful if the user's current query is returning results whose legal precedence has become outdated due to age.

The "Jurisdiction" drop-down menu allows the user to choose what level of jurisdicition the court cases should have. By default, Can I Sue will look for cases from any court in the United States, but the user can narrow this search down so that only cases from a specific jurisdiction (e.g. New York or Arizona) are considered.

Note that search parameters other than the Search Bar are only applicable to CAPAPI results. Reddit results are only meant to be considered for discussion, and hold no legal precedence.

+++
title = 'Hidden gems, wisdom of crowds, and agent swarms'
date = 2026-10-02T20:40:00-07:00
draft = false
math = true
tags = ["AI", "agents", "crowds", "recommendation"]
+++

People call a place a hidden gem when it is good and still obscure. A best-of list, a chain, or a crowd repeating the same tip is what ends the label. An ugly room, cash only, and a non-English menu are how people search. They are not proof.

This note turns that observation into a score, then checks the score on a synthetic swarm. It is a design, not a fit to restaurant ratings, and not a user study. No language model is called. The numbers below are from one offline run: 24 seeds, mention threshold \(N_0 = 40\), top 5.

## What the threads actually agree on

Obscurity is part of the definition. On [LTHForum](https://www.lthforum.com/bb/viewtopic.php?p=199629) a place stopped being a hole in the wall once the board already knew it. A [hole-in-the-wall thread](https://www.reddit.com/r/NoStupidQuestions/comments/8avcsx/what_do_people_mean_when_they_say_hole_in_the/) describes a small, unassuming storefront, known to regulars, as opposed to a place built to please every palate. Reddit's own pages returned 403 during this pass, so a lot of comment wording is from the search index, not a full thread open. Upvote counts were not used.

The room is allowed to be bad. That is not the quality signal. A [Marginal Revolution note](https://marginalrevolution.com/marginalrevolution/2023/08/on-the-negative-correlation-between-price-and-restaurant-quality.html) (11 Aug 2023) makes the survivorship point: places that are bad at both food and atmosphere die, so the plain rooms you still see look like a rule. They are not a rule.

"Locals" means repeat neighborhood demand, not a crowd. A [Fodor's thread](https://www.fodors.com/community/europe/where-do-the-locals-eat-905274/) is full of counterexamples where a room of locals was a bad meal. One Italy comment puts the label on two axes. Tourist trap: popular and not good. Hidden gem: not popular and good. Famous and good is a normal cell, not a failure.

Independent visits beat one enthusiast. Publicity ends the status. Raw star averages are a bad quality function. People tell each other to read recent mid and low scores, because means get gamed.

## What the crowd literature actually says

[Galton, "Vox Populi"](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Collective_Intelligence/Galton_1907_Vox_Populi.pdf) (*Nature*, 7 Mar 1907) asked about 800 people to guess the weight of an ox. After dropping 13 defective cards, 787 guesses remained. The median was 1207 lb. The dressed weight was 1198 lb. The guesses were not steered by speeches.

[Condorcet's jury theorem](https://en.wikipedia.org/wiki/Condorcet%27s_jury_theorem), as stated on Wikipedia, is a majority vote. If each voter is independently right with probability \(p > 1/2\), more voters help. If \(p < 1/2\), more voters make it worse. Surowiecki's book was not opened. The independence warning used here is that Wikipedia summary, not a page of the book.

[Salganik, Dodds, and Watts](https://www.princeton.edu/~mjs3/salganik_dodds_watts06_full.pdf) (*Science* 311, 854–856, 2006) gave people the same unknown songs. In the independent condition, quality is market share with no download counts. Showing other people's choices made success more unequal and less predictable. Sorting the list by current downloads made that stronger in their experiment. It did not, in the sim below.

[Evan Miller](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html) (2009) sorts a binomial proportion by the lower end of a Wilson interval, with \(z = 1.96\) at 95%. A raw average lets one five-star outrank a large modest sample.

The [IMDb ratings FAQ](https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV) shrinks obscure titles toward the global mean before they can enter the Top 250. That is a fame chart. It is the opposite of a gem label.

[Abdollahpouri, Burke, and Mobasher](https://arxiv.org/abs/1901.07555) (arXiv:1901.07555) separate a popular head, a long tail, and a distant tail so sparse that comparison is unreliable. Too few ratings is lack of evidence, not hidden treasure.

## The functions

Nothing here was fit to ratings data. \(N_0 = 40\) and \(z = 1.96\) are declared, not estimated. Décor, cash-only, and menu language are not terms.

Rescale stars onto a binary "would go back." A rater is independent if the score was recorded before they saw anyone else's score, rank, or write-up.

**Quality** \(q \in [0,1]\) is the Wilson lower bound on the independent positive rate. Let \(\hat{p} = n_+/n\) and \(n = n_+ + n_-\). If \(n = 0\), then \(q = 0\). Otherwise

$$
q = \frac{\hat{p} + \frac{z^2}{2n} - z \sqrt{\frac{\hat{p}(1-\hat{p}) + \frac{z^2}{4n}}{n}}}{1 + \frac{z^2}{n}}
$$

Small \(n\) keeps \(q\) near 0. That penalizes "two glowing reviews." It does not treat obscurity as evidence of mediocrity. Do not also shrink \(q\) toward a global mean.

**Mainstreamness** \(m \in [0,1]\). Any one fame channel can revoke "hidden":

$$
v = \min\left(1, \frac{\log(1+N)}{\log(1+N_0)}\right)
$$

\(\ell = 1\) if it is already on a best-of list or this community's canon, else 0. \(h = 1\) if it is a chain or a formula built to please everyone, else 0. \(\kappa\) is the fraction of mentions that cite a list, a previous post, or a star average, rather than a first-person visit. Then

$$
m = 1 - (1-v)(1-\ell)(1-h)(1-\kappa)
$$

\(m = 0\) only if it is unknown, unlisted, not a chain, and the mentions are firsthand. One channel at 1 forces \(m = 1\).

**Consensus** \(c \in [0,1]\) across two sealed cohorts that cannot see each other:

$$
c = 1 - |\hat{p}_1 - \hat{p}_2|
$$

If there is only one cohort, set \(c = 1\) and let \(q\) carry the uncertainty.

**Social-copying gap.** After a cohort sees other people's choices, let \(U\) be the average absolute difference of a candidate's choice share across social worlds, and \(U_0\) the same quantity on random splits of the independent cohort. Then

$$
\delta = \max(0, U - U_0)
$$

Agreement after the tally is visible is the fake kind. Do not use it as \(c\).

**Gem score**

$$
g = q \cdot c \cdot (1 - m)
$$

on the independent \(q\) only. \(g\) is a label for the "not popular and good" cell. A famous restaurant with excellent food correctly gets a high \(q\) and a low \(g\). If the decision is what to actually pick, rank by \(q\), and use \(g\) only to surface candidates the majority has not already repeated.

## The swarm

Treat each candidate (an idea, a tool, a plan) like a song in the music lab, not like a restaurant.

Private evidence is the analog of having eaten there. Many copies of one model, same prompt, no separate sources, are one voter. Condorcet's \(p > 1/2\) fails if the crowd is one model sampled many times.

Fame \(m\) is how often this candidate, or a near duplicate, has already been said, whether an earlier round wrote it down as the answer, and whether it is the default answer the model would produce with no evidence. \(\kappa\) is the fraction of critiques that argue by citing the tally instead of the rubric.

Freeze \(q\), \(c\), and \(g\) before anyone writes the winner into the shared context. Broadcasting the gem is the social-influence treatment. The next round's \(m\) should jump.

Do not encode these analogies. Odd phrasing is not a hole in the wall. "Still being talked about" is not quality. Temperature is not local knowledge. Do not up-weight the most fluent critique. \(g\) is not a utility. Using it as the only objective fills the swarm with obscure, merely-okay ideas.

## Experiment

Four planted quadrants: high quality and low fame (gems), high quality and high fame, low quality and low fame, low quality and high fame. True quality is hidden from the ranker. Raters with private evidence draw a noisy "would go back" from that quality. Raters with no private evidence who can see a tally copy the current leader.

Same budget, 24 seeds. Rank by the raw positive rate, by Wilson \(q\), and by \(g\). Repeat after the tally is visible, and once with that list sorted by current votes.

![Precision at 5 for planted gems. Sealed votes, a visible tally, and a tally sorted by current votes. Series are the raw rate, Wilson q, and the gem score.](/images/gem_precision.png)

Independent precision at 5 for planted gems was **0.708** for \(g\), **0.550** for the raw rate, and **0.125** for \(q\). Wilson is harsh on small samples, so a real gem with few votes sinks on \(q\) even when the underlying rate is high. That low number is gem recovery, not a claim that \(q\) is a bad quality estimate. The run marked the quality hypothesis as held: \(q\) tracks true quality better than the raw rate when \(n\) is small. A correlation was not saved, so it is not quoted here.

A single five-star (\(n = 1\)) has \(q = 0.207\). A modest well-sampled gem (\(n = 20\), positive rate 0.7) has \(q = 0.481\). The five-star does not win.

![Mean independent Wilson q against mean gem score. The four series are the planted quadrants.](/images/q_versus_g.png)

\(g\) ranked an obscure-mediocre item above a famous-excellent one in **405 of 600** pairs (67.5%). That is the failure mode if \(g\) is treated as utility. Famous-excellent stays high on \(q\) and low on \(g\), which is what the label is for.

![Left: unpredictability of choice shares. Right: how often the winner changes across seeds.](/images/delta_instability.png)

A visible tally raised \(\delta\) to **0.224** and dropped raw gem precision from 0.550 to **0.342**. The popular winner also jumped between seeds. Sorting that tally did not make copying worse (\(\delta = 0.175\)). Sealed \(q\) and frozen \(g\) stayed at precision 0.708 and did not chase the broadcast.

So: the raw rate promotes famous-and-mediocre items and tiny samples. \(q\) is the quality score and a poor gem finder, because gems are sparse. \(g\) finds planted gems and, used alone, prefers obscure-okay over famous-excellent. Showing the tally hurts. Sorting it was not an extra hit in this setup. The "sorting makes it worse" hypothesis is discarded.

The simulator is not in a public repo yet. The run used numpy only, no network, and wrote a `results/summary.json` with these figures.

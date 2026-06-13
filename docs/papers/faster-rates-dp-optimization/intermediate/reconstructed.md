
## Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Raman Arora   * 1   Raef Bassily   * 2 3   Tom´as Gonz´alez   * 4   Crist´obal Guzm´an   * 4   Michael Menart   * 2   Enayat Ullah   * 1

Abstract

We study the problem of approximating stationary points of Lipschitz and smooth functions under ( ε, δ ) -differential privacy (DP) in both the finite- sum and stochastic settings.   A point   � w  is called an  α -stationary point of a function  F   :  R d   → R  if ∥∇ F (  � w ) ∥≤ α .   We give a new construction that improves   over   the   existing   rates   in   the   stochas- tic optimization setting, where the goal is to find approximate stationary points of the population risk   given   n   samples.   Our   construction   finds   a ˜ O � 1 n 1 / 3   + � √ d nε � 1 / 2 � -stationary point of the pop- ulation risk in time linear in  n .   We also provide an efficient algorithm that finds an   ˜ O �� √ d nε � 2 / 3 � - stationary point in the finite-sum setting.   This im- proves on the previous best rate of   ˜ O �� √ d nε � 1 / 2 � . Furthermore, under the additional assumption of convexity,   we   completely   characterize   the   sam- ple complexity of finding stationary points of the population risk (up to polylog factors) and show that the optimal rate on population stationarity is ˜Θ � 1 √ n  + √

d nε � .   Finally, we show that our methods can   be   used   to   provide   dimension-independent rates of  O � 1 √ n   + min �� √ rank

nε � 2 / 3 , 1 ( nε ) 2 / 5 �� on population   stationarity   for   Generalized   Linear Models   (GLM),   where   rank   is   the   rank   of   the design matrix, which improves upon the previous best known rate.

* Equal   contribution 1 Department   of   Computer   Science,   The Johns Hopkins University   2 Department of Computer Science & Engineering, The Ohio State University   3 Translational Data An- alytics Institute (TDAI), The Ohio State University   4 Institute for Mathematical and Computational Engineering, Pontificia Univer- sidad   Cat ´ olica   de   Chile.   Correspondence   to:   Michael   Menart < menart.2@osu.edu > , Enayat Ullah  < enayat@jhu.edu > .

Proceedings   of   the   40   th   International   Conference   on   Machine Learning , Honolulu, Hawaii, USA. PMLR 202, 2023.   Copyright 2023 by the author(s).

1. Introduction

Protecting users’ data in machine learning models has be- come a central concern in multiple contexts, e.g. those in- volving financial or health data.   In this respect, differential privacy (DP) is the gold standard for rigorous privacy pro- tection ( Dwork & Roth ,  2014 ).   Therefore, recent research has focused on the limits and possibilities of solving some of the most well-established machine learning problems un- der the constraint of DP. Despite intensive research, some fundamental problems remain not completely understood. One example is nonconvex optimization; namely, the task of approximating stationary points, which has been heavily studied in recent years in the non-private setting ( Fang et al. , 2018 ;   Ma   et   al. ,   2018 ;   Carmon   et   al. ,   2017 ;   Nesterov   & Polyak ,  2006 ;  Ghadimi & Lan ,  2013 ;  Arjevani et al. ,  2019 ; Foster et al. ,  2019 ).   This problem is motivated by the in- tractability of nonconvex (global) optimization, as well as by a number of settings where stationary points have been shown   to   be   global   minima   ( Ge   et   al. ,   2016 ;   Sun   et   al. , 2016 ).

1.1. Contributions

In this work, we make progress towards resolving the com- plexity of approximating stationary points in optimization under the constraint of differential privacy, for both empir- ical   and   population   risks.   A   summary   of   our   new   results is available in Table  1 .   In what follows,   d  is the problem dimension,  n  is the dataset size, and  ε, δ  are the approximate DP parameters.

Our first set of results pertains to the task of approximating stationary   points   of   the   population   risk.   Results   for   this problem are scarce.   We provide the fastest rate up to date for   this   problem   under   DP,   of   ˜ O � 1 n 1 / 3   + � √ d nε � 1 / 2 � ,   with an algorithm that moreover has oracle complexity   n  (i.e., is   single-pass).   This   algorithm   is   a   noisy   version   of   the SPIDER algorithm ( Fang et al. ,  2018 ), whose gradient esti- mators are built using a tree-aggregation data structure for prefix-sums ( Asi et al. ,  2021 ).

Next, we   focus   on   the   task   of   approximating   sta- tionary   points   in   empirical   nonconvex   optimization (a.k.a.   finite-sum   case).   In   this   context,   we   provide   al-

1



| 0                                                                                                                  | 1                                                                |
|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| Raman Arora * 1 Raef Bassily * 2 3 Tom´as Gonz´alez * 4 Crist´obal Guzm´an * 4 Michael Menart * 2 Enayat Ullah * 1 |                                                                  |
| Abstract                                                                                                           | 1. Introduction                                                  |
|                                                                                                                    | Protecting users’ data in machine learning models has be-        |
|                                                                                                                    | come a central concern in multiple contexts, e.g. those in-      |
| We study the problem of approximating stationary                                                                   |                                                                  |
|                                                                                                                    | volving financial or health data. In this respect, differential  |
| points of Lipschitz and smooth functions under                                                                     |                                                                  |
|                                                                                                                    | privacy (DP) is the gold standard for rigorous privacy pro-      |
| (ε, δ)-differential privacy (DP) in both the finite-                                                               |                                                                  |
|                                                                                                                    | tection (Dwork & Roth, 2014). Therefore, recent research         |
| sum and stochastic settings. A point (cid:98)w is called                                                           | has focused on the limits and possibilities of solving some      |
| an α-stationary point of a function F : Rd → R if                                                                  |                                                                  |
|                                                                                                                    | of the most well-established machine learning problems un-       |
| ∥∇F ( (cid:98)w)∥ ≤ α. We give a new construction that                                                             | der the constraint of DP. Despite intensive research, some       |
| improves over the existing rates in the stochas-                                                                   |                                                                  |
|                                                                                                                    | fundamental problems remain not completely understood.           |
| tic optimization setting, where the goal is to find                                                                |                                                                  |
|                                                                                                                    | One example is nonconvex optimization; namely, the task          |
| approximate stationary points of the population                                                                    |                                                                  |
|                                                                                                                    | of approximating stationary points, which has been heavily       |
| risk given n samples. Our construction finds a                                                                     |                                                                  |
|                                                                                                                    | studied in recent years in the non-private setting (Fang et al., |
| d                                                                                                                  | 2018; Ma et al., 2018; Carmon et al., 2017; Nesterov &           |
| O(cid:0)                                                                                                           |                                                                  |
| (cid:3)1/2(cid:1)-stationary point of the pop-                                                                     |                                                                  |
| nε                                                                                                                 |                                                                  |
| n1/3 + (cid:2) √                                                                                                   |                                                                  |
| ulation risk in time linear in n. We also provide                                                                  | Polyak, 2006; Ghadimi & Lan, 2013; Arjevani et al., 2019;        |
| d                                                                                                                  | Foster et al., 2019). This problem is motivated by the in-       |
| an efficient algorithm that finds an ˜O(cid:0)(cid:2) √                                                            |                                                                  |
| (cid:3)2/3(cid:1)-                                                                                                 |                                                                  |
| nε                                                                                                                 |                                                                  |
| stationary point in the finite-sum setting. This im-                                                               | tractability of nonconvex (global) optimization, as well as      |
| d                                                                                                                  | by a number of settings where stationary points have been        |
| proves on the previous best rate of ˜O(cid:0)(cid:2) √                                                             |                                                                  |
| (cid:3)1/2(cid:1).                                                                                                 |                                                                  |
| nε                                                                                                                 |                                                                  |
|                                                                                                                    | shown to be global minima (Ge et al., 2016; Sun et al.,          |
| Furthermore, under the additional assumption of                                                                    |                                                                  |
|                                                                                                                    | 2016).                                                           |
| convexity, we completely characterize the sam-                                                                     |                                                                  |
| ple complexity of finding stationary points of the                                                                 |                                                                  |
|                                                                                                                    | 1.1. Contributions                                               |
| population risk (up to polylog factors) and show                                                                   |                                                                  |
| that the optimal rate on population stationarity is                                                                |                                                                  |
| √                                                                                                                  | In this work, we make progress towards resolving the com-        |
| 1                                                                                                                  |                                                                  |
| d                                                                                                                  |                                                                  |
| Θ(cid:0)                                                                                                           |                                                                  |
| √                                                                                                                  |                                                                  |
| (cid:1). Finally, we show that our methods                                                                         |                                                                  |
| nε                                                                                                                 | plexity of approximating stationary points in optimization       |
| n +                                                                                                                |                                                                  |
| can be used to provide dimension-independent                                                                       | under the constraint of differential privacy, for both empir-    |
| rank                                                                                                               |                                                                  |
| 1                                                                                                                  |                                                                  |
| 1                                                                                                                  |                                                                  |
| (cid:3)2/3                                                                                                         |                                                                  |
| n + min (cid:0)(cid:2) √                                                                                           |                                                                  |
| √                                                                                                                  | ical and population risks. A summary of our new results          |
| ,                                                                                                                  |                                                                  |
| rates of O(cid:0)                                                                                                  |                                                                  |
| (cid:1)(cid:1) on                                                                                                  |                                                                  |
| nε                                                                                                                 |                                                                  |
| (nε)2/5                                                                                                            |                                                                  |
|                                                                                                                    | is available in Table 1.                                         |
|                                                                                                                    | In what follows, d is the problem                                |
| population stationarity for Generalized Linear                                                                     |                                                                  |
|                                                                                                                    | dimension, n is the dataset size, and ε, δ are the approximate   |
| Models (GLM), where rank is the rank of the                                                                        |                                                                  |
|                                                                                                                    | DP parameters.                                                   |
| design matrix, which improves upon the previous                                                                    |                                                                  |
| best known rate.                                                                                                   |                                                                  |
|                                                                                                                    | Our first set of results pertains to the task of approximating   |
|                                                                                                                    | stationary points of                                             |
|                                                                                                                    | the population risk. Results for                                 |
|                                                                                                                    | this                                                             |
|                                                                                                                    | problem are scarce. We provide the fastest rate up to date       |
|                                                                                                                    | d                                                                |
|                                                                                                                    | O(cid:0)                                                         |
|                                                                                                                    | for                                                              |
|                                                                                                                    | this problem under DP, of                                        |
|                                                                                                                    | (cid:3)1/2(cid:1), with                                          |
|                                                                                                                    | nε                                                               |
|                                                                                                                    | n1/3 + (cid:2) √                                                 |
| *Equal contribution                                                                                                |                                                                  |
| 1Department of Computer Science, The                                                                               |                                                                  |
|                                                                                                                    | an algorithm that moreover has oracle complexity n (i.e.,        |
| Johns Hopkins University 2Department of Computer Science &                                                         |                                                                  |
| Engineering, The Ohio State University 3Translational Data An-                                                     | is single-pass).                                                 |
|                                                                                                                    | This algorithm is a noisy version of                             |
|                                                                                                                    | the                                                              |
| alytics Institute (TDAI), The Ohio State University 4Institute for                                                 | SPIDER algorithm (Fang et al., 2018), whose gradient esti-       |
| Mathematical and Computational Engineering, Pontificia Univer-                                                     |                                                                  |
|                                                                                                                    | mators are built using a tree-aggregation data structure for     |
| sidad Cat´olica de Chile.                                                                                          |                                                                  |
| Correspondence to: Michael Menart                                                                                  |                                                                  |
|                                                                                                                    | prefix-sums (Asi et al., 2021).                                  |
| <menart.2@osu.edu>, Enayat Ullah <enayat@jhu.edu>.                                                                 |                                                                  |
|                                                                                                                    | Next,                                                            |
|                                                                                                                    | we                                                               |
|                                                                                                                    | focus                                                            |
|                                                                                                                    | on                                                               |
|                                                                                                                    | the                                                              |
|                                                                                                                    | task                                                             |
|                                                                                                                    | of                                                               |
|                                                                                                                    | approximating                                                    |
|                                                                                                                    | sta-                                                             |
| Proceedings of                                                                                                     |                                                                  |
| the 40 th International Conference on Machine                                                                      |                                                                  |
|                                                                                                                    | tionary                                                          |
|                                                                                                                    | points                                                           |
|                                                                                                                    | in                                                               |
|                                                                                                                    | empirical                                                        |
|                                                                                                                    | nonconvex                                                        |
|                                                                                                                    | optimization                                                     |
| Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright                                                         |                                                                  |
|                                                                                                                    | (a.k.a. finite-sum case).                                        |
|                                                                                                                    | In this context, we provide al-                                  |
| 2023 by the author(s).                                                                                             |                                                                  |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

gorithms   with   rate   O �� √ d nε � 2 / 3 � ,   and   oracle   complexity 1

˜ O � max �� n 5 ε 2 d � 1 / 3 , � nε √

d � 2 �� .   This   rate   is   sharper   than the best known for this problem ( Wang et al. ,  2017 ).

We continue by investigating stationary points for convex losses and give an algorithm based on the recursive regular- ization technique of ( Allen-Zhu ,  2018 ) which achieves the optimal rate of   ˜ Θ � 1 √ n   + √

d nε � on population stationarity.   To

establish optimality, we give a lower bound of  Ω � √ d nε � on empirical stationarity under DP (Theorem  4.3 ) and a non- private   lower   bound   of   Ω(   1 √ n )   on   population   stationarity (Theorem  A.2 ).   We also give a linear-time method, which achieves the optimal rate when the smoothness parameter is not so large.   We conclude the paper showing a black-box reduction that converts any DP method for finding station- ary points of smooth and Lipschitz losses into a DP method with  dimension-independent rates  for the case of general- ized linear models (GLM). Using our proposed method with Private Spiderboost as the base algorithm yields a rate of ˜ O � 1 √ n   + min �� √ rank

nε � 2 / 3 , 1 ( nε ) 2 / 5 �� on   population   sta- tionarity.   This   improves   upon   the   result   of   ( Song   et   al. , 2021 ) which proposed a method with   ˜ O �� √ rank

nε � 1 / 2 � em- pirical stationarity 2 .

1.2. Our Techniques

Our methods combine multiple techniques from optimiza- tion and differential privacy in novel ways.   The lower bound for the empirical norm of the gradient uses fingerprinting codes to a loss similar to that used for Differentially Private- Empirical   Risk   Minimization   (DP-ERM)   ( Bassily   et   al. , 2014 ), crafted to work in the unconstrained case.   This lower bound can be extended to the population gradient norm by a known re-sampling argument ( Bassily et al. ,  2019 ).   We also give a non-private lower bound of  Ω(1 / √ n )  on population stationarity with  n  samples which holds even in dimension 1, as opposed to previous results ( Foster et al. ,  2019 ).

Efficient   algorithms   for   (both   empirical   and   population) norm   of   the   gradient   are   derived   using   noisy   versions   of variance-reduced stochastic first order methods, which have proved remarkably useful in DP stochastic optimization ( Asi et al. ,  2021 ;  Bassily et al. ,  2021b ; a ).   In the case of the empir- ical risk, we use a noisy version of SpiderBoost ( Wang et al. , 2019c ).   We remark that our methods can achieve compara- ble rates when applied to similar algorithms such as Spider ( Fang et al. ,  2018 ) and Storm ( Cutkosky & Orabona ,  2019 ), but SpiderBoost allows for a larger learning rate which is

1 We consider for complexity the first-order oracle model, stan- dard for continuous optimization ( Nemirovsky & Yudin ,  1983 ). 2 This is the rate obtained after fixing a mistake in the proof of Theorem 4.1 in ( Song et al. ,  2021 ).   Specifically, in their proof, the last term in Eq.   (14) is missing a factor of  T .

considered better in practice.   For the population risk, it is worth noting that the empirical norm of the gradient does not   translate   directly   into   population   gradient   guarantees, even if the algorithm in use is uniformly stable ( Bousquet & Elisseeff ,  2002 ), since this type of guarantee does not enjoy a  stability-implies-generalization  property.   Therefore, we opt for single pass methods that combine variance-reduction with tree-aggregation; these techniques are particularly suit- able for the classical Spider algorithm ( Fang et al. ,  2018 ), which is the one we base our method on.   For the convex setting, we use recursive regularization ( Allen-Zhu ,  2018 ) which was used to achieve the optimal non-private rate by ( Foster et al. ,  2019 ).

Finally,   our   method   for   (non-convex)   GLMs   uses   the Johnson-Lindenstrauss based dimensionality reduction tech- nique similar to ( Arora et al. ,  2022 ), which focused on the convex   setting.   Moreover,   for   population   stationarity   of GLMs, we give a new uniform convergence result of gradi- ents of Lipschitz functions.   This guarantee, unlike the prior work of ( Foster et al. ,  2018 ), has only poly-logarithmic de- pendence on the radius of the constraint set, which is crucial for our analysis.

1.3. Related Work

The current work fits within the literature of differentially private   optimization,   which   has   primarily   focused   on   the convex case ( Chaudhuri et al. ,  2011 ;  Jain et al. ,  2012 ;  Kifer et al. ,  2012 ;  Bassily et al. ,  2014 ;  Talwar et al. ,  2014 ;  Jain & Thakurta ,  2014 ;  Talwar et al. ,  2015 ;  Bassily et al. ,  2019 ; Feldman et al. ,  2020 ;  Asi et al. ,  2021 ;  Bassily et al. ,  2021b ). The culmination of this line of work for the convex smooth case showed that optimal rates are achievable in linear time ( Feldman et al. ,  2020 ;  Asi et al. ,  2021 ;  Bassily et al. ,  2021b ). Our   work   shows   that   in   the   convex   case   similar   rates   are achievable for the norm of the gradient:   this result is useful, e.g., for dual formulations of linearly constrained convex programs ( Nesterov ,  2012 ), and moreover it has become a problem of independent interest ( Allen-Zhu ,  2018 ;  Foster et al. ,  2019 ). 3

Regarding   stationary   points   for   nonconvex   losses,   work in   DP   is   far   more   recent,   and   primarily   focused   on   the empirical stationarity ( Wang et al. ,  2017 ;  Zhang et al. ,  2017 ;

3 To provide a specific example, consider the dual of the reg- ularized discrete optimal transport problem, as discussed in ( Di- akonikolas & Guzm ´ an ,  2023 ), Section 5.6.   If the marginals  µ, ν   in that model are accessed through i.i.d.   samples, then this becomes an   SCO   problem.   Moreover,   it   is   argued   in   that   reference   that approximate stationary points provide approximately feasible and optimal transports through duality arguments.   Hence, the result is an SCO problem where we require  approximate stationary points.

2



| 0                                                                                       | 1                                                                  |
|:----------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                                    |
| d                                                                                       | considered better in practice. For the population risk, it is      |
| gorithms with rate O(cid:0)(cid:2) √                                                    |                                                                    |
| (cid:3)2/3(cid:1), and oracle complexity1                                               |                                                                    |
| nε                                                                                      |                                                                    |
|                                                                                         | worth noting that the empirical norm of the gradient does          |
| , (cid:0) nε√                                                                           |                                                                    |
| O(cid:0) max (cid:8)(cid:0) n5ε2                                                        |                                                                    |
| (cid:1)2(cid:9)(cid:1).                                                                 |                                                                    |
| This rate is sharper                                                                    |                                                                    |
| than                                                                                    |                                                                    |
| d                                                                                       |                                                                    |
| d                                                                                       | not                                                                |
|                                                                                         | translate directly into population gradient guarantees,            |
| the best known for this problem (Wang et al., 2017).                                    |                                                                    |
|                                                                                         | even if the algorithm in use is uniformly stable (Bousquet &       |
| We continue by investigating stationary points for convex                               | Elisseeff, 2002), since this type of guarantee does not enjoy      |
| losses and give an algorithm based on the recursive regular-                            | a stability-implies-generalization property. Therefore, we         |
| ization technique of (Allen-Zhu, 2018) which achieves the                               | opt for single pass methods that combine variance-reduction        |
| √                                                                                       |                                                                    |
| 1                                                                                       |                                                                    |
| d                                                                                       |                                                                    |
| √                                                                                       | with tree-aggregation; these techniques are particularly suit-     |
| optimal rate of ˜Θ(cid:0)                                                               |                                                                    |
| (cid:1) on population stationarity. To                                                  |                                                                    |
| nε                                                                                      |                                                                    |
| n +                                                                                     |                                                                    |
|                                                                                         | able for the classical Spider algorithm (Fang et al., 2018),       |
| d                                                                                       |                                                                    |
| establish optimality, we give a lower bound of Ω(cid:0) √                               |                                                                    |
| (cid:1) on                                                                              |                                                                    |
| nε                                                                                      | which is the one we base our method on. For the convex             |
| empirical stationarity under DP (Theorem 4.3) and a non-                                |                                                                    |
|                                                                                         | setting, we use recursive regularization (Allen-Zhu, 2018)         |
| 1                                                                                       |                                                                    |
| √                                                                                       |                                                                    |
| n ) on population stationarity                                                          | which was used to achieve the optimal non-private rate by          |
| (Theorem A.2). We also give a linear-time method, which                                 |                                                                    |
|                                                                                         | (Foster et al., 2019).                                             |
| achieves the optimal rate when the smoothness parameter is                              |                                                                    |
|                                                                                         | Finally,                                                           |
|                                                                                         | our method                                                         |
|                                                                                         | for                                                                |
|                                                                                         | (non-convex) GLMs                                                  |
|                                                                                         | uses                                                               |
|                                                                                         | the                                                                |
| not so large. We conclude the paper showing a black-box                                 |                                                                    |
|                                                                                         | Johnson-Lindenstrauss based dimensionality reduction tech-         |
| reduction that converts any DP method for finding station-                              |                                                                    |
|                                                                                         | nique similar to (Arora et al., 2022), which focused on the        |
| ary points of smooth and Lipschitz losses into a DP method                              |                                                                    |
|                                                                                         | convex setting. Moreover,                                          |
|                                                                                         | for population stationarity of                                     |
| with dimension-independent rates for the case of general-                               |                                                                    |
|                                                                                         | GLMs, we give a new uniform convergence result of gradi-           |
| ized linear models (GLM). Using our proposed method with                                |                                                                    |
|                                                                                         | ents of Lipschitz functions. This guarantee, unlike the prior      |
| Private Spiderboost as the base algorithm yields a rate of                              |                                                                    |
| (cid:17)(cid:17)                                                                        |                                                                    |
| (cid:16)(cid:2) √                                                                       |                                                                    |
| rank                                                                                    | work of (Foster et al., 2018), has only poly-logarithmic de-       |
| 1                                                                                       |                                                                    |
| (cid:3)2/3                                                                              |                                                                    |
| ˜                                                                                       |                                                                    |
| (cid:16) 1√                                                                             |                                                                    |
| O                                                                                       |                                                                    |
| ,                                                                                       |                                                                    |
| on population sta-                                                                      |                                                                    |
| nε                                                                                      |                                                                    |
| n + min                                                                                 |                                                                    |
| (nε)2/5                                                                                 |                                                                    |
|                                                                                         | pendence on the radius of the constraint set, which is crucial     |
| tionarity.                                                                              |                                                                    |
| This improves upon the result of                                                        |                                                                    |
| (Song et al.,                                                                           |                                                                    |
|                                                                                         | for our analysis.                                                  |
| rank                                                                                    |                                                                    |
| 2021) which proposed a method with ˜O(cid:0)(cid:2) √                                   |                                                                    |
| (cid:3)1/2(cid:1) em-                                                                   |                                                                    |
| nε                                                                                      |                                                                    |
| pirical stationarity2.                                                                  |                                                                    |
|                                                                                         | 1.3. Related Work                                                  |
|                                                                                         | The current work fits within the literature of differentially      |
| 1.2. Our Techniques                                                                     |                                                                    |
|                                                                                         | private optimization, which has primarily focused on the           |
| Our methods combine multiple techniques from optimiza-                                  |                                                                    |
|                                                                                         | convex case (Chaudhuri et al., 2011; Jain et al., 2012; Kifer      |
| tion and differential privacy in novel ways. The lower bound                            |                                                                    |
|                                                                                         | et al., 2012; Bassily et al., 2014; Talwar et al., 2014; Jain      |
| for the empirical norm of the gradient uses fingerprinting                              |                                                                    |
|                                                                                         | & Thakurta, 2014; Talwar et al., 2015; Bassily et al., 2019;       |
| codes to a loss similar to that used for Differentially Private-                        |                                                                    |
|                                                                                         | Feldman et al., 2020; Asi et al., 2021; Bassily et al., 2021b).    |
| Empirical Risk Minimization (DP-ERM)                                                    |                                                                    |
| (Bassily et al.,                                                                        |                                                                    |
|                                                                                         | The culmination of this line of work for the convex smooth         |
| 2014), crafted to work in the unconstrained case. This lower                            |                                                                    |
|                                                                                         | case showed that optimal rates are achievable in linear time       |
| bound can be extended to the population gradient norm by a                              |                                                                    |
|                                                                                         | (Feldman et al., 2020; Asi et al., 2021; Bassily et al., 2021b).   |
| known re-sampling argument (Bassily et al., 2019). We also                              |                                                                    |
|                                                                                         | Our work shows that                                                |
|                                                                                         | in the convex case similar rates are                               |
| √                                                                                       |                                                                    |
| give a non-private lower bound of Ω (1/                                                 |                                                                    |
| n) on population                                                                        |                                                                    |
|                                                                                         | achievable for the norm of the gradient:                           |
|                                                                                         | this result is useful,                                             |
| stationarity with n samples which holds even in dimension                               |                                                                    |
|                                                                                         | e.g., for dual formulations of linearly constrained convex         |
| 1, as opposed to previous results (Foster et al., 2019).                                |                                                                    |
|                                                                                         | programs (Nesterov, 2012), and moreover it has become a            |
|                                                                                         | problem of independent interest (Allen-Zhu, 2018; Foster           |
| Efficient algorithms for                                                                |                                                                    |
| (both empirical and population)                                                         |                                                                    |
|                                                                                         | et al., 2019).3                                                    |
| norm of the gradient are derived using noisy versions of                                |                                                                    |
| variance-reduced stochastic first order methods, which have                             |                                                                    |
|                                                                                         | Regarding stationary points for nonconvex losses, work             |
| proved remarkably useful in DP stochastic optimization (Asi                             |                                                                    |
|                                                                                         | in DP is                                                           |
|                                                                                         | far more recent, and primarily focused on the                      |
| et al., 2021; Bassily et al., 2021b;a). In the case of the empir-                       |                                                                    |
|                                                                                         | empirical stationarity (Wang et al., 2017; Zhang et al., 2017;     |
| ical risk, we use a noisy version of SpiderBoost (Wang et al.,                          |                                                                    |
|                                                                                         | 3To provide a specific example, consider the dual of the reg-      |
| 2019c). We remark that our methods can achieve compara-                                 |                                                                    |
|                                                                                         | ularized discrete optimal transport problem, as discussed in (Di-  |
| ble rates when applied to similar algorithms such as Spider                             |                                                                    |
|                                                                                         | akonikolas & Guzm´an, 2023), Section 5.6. If the marginals µ, ν in |
| (Fang et al., 2018) and Storm (Cutkosky & Orabona, 2019),                               |                                                                    |
|                                                                                         | that model are accessed through i.i.d. samples, then this becomes  |
| but SpiderBoost allows for a larger learning rate which is                              | an SCO problem. Moreover,                                          |
|                                                                                         | it                                                                 |
|                                                                                         | is argued in that reference that                                   |
|                                                                                         | approximate stationary points provide approximately feasible and   |
| 1We consider for complexity the first-order oracle model, stan-                         |                                                                    |
|                                                                                         | optimal transports through duality arguments. Hence, the result is |
| dard for continuous optimization (Nemirovsky & Yudin, 1983).                            |                                                                    |
|                                                                                         | an SCO problem where we require approximate stationary points.     |
| 2This is the rate obtained after fixing a mistake in the proof of                       |                                                                    |
| Theorem 4.1 in (Song et al., 2021). Specifically, in their proof, the                   |                                                                    |
| last term in Eq. (14) is missing a factor of T .                                        |                                                                    |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Setting Convergence Our Rate Previous best-known rate

Non-convex

Empirical

� √ d nε � 2 / 3 (Thm.   4.2 )

� √ d nε � 1 / 2 ( Wang et al. ,  2017 )

Population 1 n 1 / 3   + � √ d nε � 1 / 2 (Thm.   3.2 )

√

dε  + � √ d nε � 1 / 2 ( Zhou et al. ,  2020 )

Convex Population 1 √ n   + √

d nε (Thm.   5.1 ) None

Non-convex GLM

Empirical

� √ rank

nε � 2 / 3 ∧ 1 ( nϵ ) 2 / 5 (Cor.   6.2 )

� √ rank

nε � 1 / 2 ( Song et al. ,  2021 )

Population 1 √ n   + � √ rank nε � 2 / 3 ∧ 1 ( nϵ ) 2 / 5 (Cor.   6.2 ) None

Convex GLM Population 1 √ n   + √

rank nε ∧ 1 √ nϵ (Cor.   6.2 ) None

Table 1.  Results summary:   We omit log factors and function-class parameters.   The symbol  ∧ stands for minimum of the quantities.

Wang   &   Xu ,   2019 ;   Wang   et   al. ,   2019a ) 4 .   Under   similar assumptions   to   ours   these   works   approximate   stationary points with rate   ˜ O �� √ d nε � 1 / 2 � , which is slower than ours.

Works   addressing   population   guarantees   for   the   norm   of the gradient under DP are scarce.   ( Zhou et al. ,  2020 ) pro- posed a noisy gradient method, whose population guaran- tee   is   obtained   by   generalization   properties   of   DP.   How- ever,   the   best   guarantee   obtainable   with   their   analysis   is O �� √ d nε � 1 / 2   + √

dε � 5 . Note   that   for   any   ε   this   rate   is Ω � [ d/n ] 1 / 3 � .   Under   additional   assumptions   (on   the   Hes- sian), ( Wang & Xu ,  2019 ) obtains a rate of   ˜ O ( � d/ ( nε ))  by uniform convergence of gradients, which is sharper when  ε is constant.   By contrast, our rate is much faster than both for  ε  = Θ(1) .   In particular, in this range, our rates are faster than   those   obtained   by   uniform   convergence,   O ( � d/n ) ( Foster   et   al. ,   2018 ).   Moreover,   our   method   runs   in   time linear   in   n .   On   the   other   hand,   in   the   much   more   restric- tive setting where the loss satisfies the Polyak- Ł ojasiewicz (PL) inequality, ( Zhang et al. ,  2021 ) provide  population risk bounds of   ˜ O ( d/ [ nε ] 2 )  under DP.

The work of ( Bassily et al. ,  2021a ) studies population guar- antees   for   stationarity   in   constrained   settings,   obtaining rates   O � 1 n 1 / 3   + � √ d nε � 2 / 5 � in   linear   time.   Notice   first   that these guarantees are based on the Frank-Wolfe gap,   mak- ing   those   results   incomparable   to   ours.   Despite   this   fact,

4 Another   work,   ( Wang   et   al. ,   2019b ),   claims   to   achieve   this with improved oracle complexity.   However, the analysis therein contains   an   error which   is   not   easily   fixed.   Specifically,   ( Wang et al. ,  2019b , proof of Theorem 4.1) uses  σ 2 0 b 2 0   >  0 . 7  to employ privacy amplification via subsampling.   This is not true as they set σ 0   = 1 / [ d 1 / 4 √ n ]  and  b 0   =   √ n/d 1 / 4 . 5 ( Zhou et al. ,  2020 ) omits the term √

dε , but this omission is only valid when  ε <  1 / [ n √

d ] 1 / 3 .

their rates are slower than ours. 6   On the other hand, they provide   results   for   (close   to   nearly)   stationary   points   in constrained/unconstrained settings,   for a broader class of weakly   convex   losses   (possibly   nonsmooth).   This   result is   then   more   general,   but   the   rate   of   O � 1 n 1 / 4   + � √ d nε � 1 / 3 �

is   substantially   slower   than   ours,   and   their   algorithm   has oracle complexity which is superlinear in  n .

The problem of stationary points in (nonprivate) stochastic optimization has drawn major attention recently ( Ghadimi & Lan ,  2013 ;  2016 ;  Fang et al. ,  2018 ;  Allen-Zhu ,  2018 ;  Foster et al. ,  2018 ;  2019 ;  Arjevani et al. ,  2019 ).   To the best of our knowledge, no lower bounds for the sample complexity 7   of this problem are known (beyond those known for the convex case ( Foster et al. ,  2019 )).   On the other hand, oracle com- plexity is by now understood:   in high dimensions, for (on average) smooth losses the optimal stochastic oracle com- plexity rate is  O (1 /n 1 / 3 )  ( Arjevani et al. ,  2019 ).   Although this provides some evidence of the sharpness of our results (see Appendix  B.2 ), note that these lower bounds require very high dimensional constructions (namely,  d  = Ω(1 /α 4 ) , where  α  is the rate), which limits their applicability in the private setting.

In an independent and concurrent work, ( Tran & Cutkosky , 2022 )   achieve   a   rate   of   O ( � √ d nϵ � 2 / 3   + 1 √ n )   on   the   empir-

ical   gradient   with   gradient   complexity   O ( n 7 / 3 ϵ 3 / 4 /d 2 / 3 ) using a DP tree aggregation method.   Note that our result removes the  1 / √ n  term and improves the oracle complexity to   ˜ O � max �� n 5 ε 2 d � 1 / 3 , � nε √

d � 2 �� , which is better whenever

6 We believe our methods can be extended to constrained set- tings using gradient mapping, a guarantee for which is stronger than for Frank-Wolfe gap ( Lan ,  2020 , Section 7.5.1).   We defer this extension to future work. 7 Sample   complexity   is   the   fundamental   limit   on   the   sample size   needed,   as   a   function   of   α ,   to   achieve   α   stationarity.   This is   different   from   the   oracle   complexity   as   one   is   not   limited   to first-order methods.

3



| 0          | 1          | 2             | 3                   |
|:-----------|:-----------|:--------------|:--------------------|
| Non-convex | Empirical  | (cid:16) √    | (cid:16) √          |
|            |            | (cid:17)2/3   | (cid:17)1/2         |
|            |            | d             | d                   |
|            |            | (Thm. 4.2)    | (Wang et al., 2017) |
|            |            | nε            | nε                  |
|            | Population | (cid:17)1/2   | √                   |
|            |            | (cid:16) √    | d                   |
|            |            | 1             | (cid:1)1/2          |
|            |            | d             | dε + (cid:0) √      |
|            |            | (Thm. 3.2)    | (Zhou et al., 2020) |
|            |            | nε            | nε                  |
|            |            | n1/3 +        |                     |
| Convex     | Population | √             | None                |
|            |            | 1             |                     |
|            |            | d             |                     |
|            |            | √             |                     |
|            |            | (Thm. 5.1)    |                     |
|            |            | nε            |                     |
|            |            | n +           |                     |
| Non-convex | Empirical  | rank          | (cid:16) √          |
| GLM        |            | 1             | (cid:17)1/2         |
|            |            | (cid:2) √     | rank                |
|            |            | (cid:3)2/3    | (Song et al., 2021) |
|            |            | ∧             | nε                  |
|            |            | (Cor. 6.2)    |                     |
|            |            | nε            |                     |
|            |            | (nϵ)2/5       |                     |
|            | Population | rank          | None                |
|            |            | 1             |                     |
|            |            | 1             |                     |
|            |            | √             |                     |
|            |            | (cid:3)2/3 ∧  |                     |
|            |            | (Cor. 6.2)    |                     |
|            |            | nε            |                     |
|            |            | n + (cid:2) √ |                     |
|            |            | (nϵ)2/5       |                     |
| Convex GLM | Population | √             | None                |
|            |            | rank          |                     |
|            |            | 1             |                     |
|            |            | √             |                     |
|            |            | 1√            |                     |
|            |            | ∧             |                     |
|            |            | (Cor. 6.2)    |                     |
|            |            | nε            |                     |
|            |            | n +           |                     |
|            |            | nϵ            |                     |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

d  ≤ n 2 ϵ 1 / 4   (i.e. essentially whenever the error is nontrivial). Further, we accomplish this with a much simpler analysis.

2. Preliminaries

Let   f :   R d   ×   X → R   denote   a   (loss)   function   tak- ing   as   input,   the   model   parameter   w   and   data   point   x   ∈ X . We   assume   that   the   function   w   �→ f ( w ;  x )   is   L 0 - Lipschitz   and   L 1 -smooth.   That   is,   for   all   x   ∈X   and w 1 , w 2   ∈ R d ,   | f ( w 1 ;  x )  − f ( w 2 ;  x ) |   ≤ L 0  ∥ w 1  − w 2 ∥ and   ∥∇ f ( w 1 ;  x )  −∇ f ( w 2 ;  x ) ∥≤ L 1  ∥ w 1  − w 2 ∥ .   Given a   dataset   S   ∈X   n   of   n   points,   we   define   the   empirical risk   as   F ( w ;  S )   = 1 n � n i =1   f ( w ;  x i ) .   Assuming   that   the data   points   are   sampled   i.i.d. from   an   unknown   distri- bution   D ,   the   population   risk,   denoted   as   F ( w ;  D )   is   de- fined as  F ( w ;  D ) =  E x ∼D f ( w ;  x ) .   Furthermore, we define F 0   =  F (0;  S )  − min w ∈ R d  { F ( w ;  S ) }  when discussing the empirical case and similarly for the population loss when discussing stationary points of the population loss.   We use w ∗ to denote the population risk minimizer.   Finally, we use the notation  I d  to denote the  d  ×  d  identity matrix and use [ a ]  to denote the set  { 1 ,  2 , ..., a }  for  a  ≥ 1 .

Stationary points: Given a dataset  S , our goal is to find an   α -stationary   point   ¯ w   of   either   empirical   or   population risk;   formally,   ∥∇ F ( ¯ w ;  S ) ∥≤ α   or   ∥∇ F ( ¯ w ;  D ) ∥≤ α , respectively.

Differential Privacy (DP) ( Dwork et al. ,  2006 ): An algo- rithm   A   is   ( ε, δ ) -differentially   private   if   for   all   datasets S   and   S ′   differing   in   one   data   point   and   all   events E   in   the   range   of   the   A ,   we   have,   P  ( A ( S )  ∈E )   ≤ e ε P  ( A ( S ′ )  ∈E ) +  δ .

Generalized Linear Models (GLMs): For data domain X   ⊆ R d   and  Y   ⊆ R , a loss function  f   :  R d ×X ×Y   → R  is a GLM if  f ( w ; ( x, y )) =  ϕ y  ( ⟨ w, x ⟩ )  for some function  ϕ y . Our result for GLMs uses random matrices which satisfy the Johnson-Lindenstrauss (JL) property, defined as follows.

Definition 2.1  ( ( γ, β ) -JL property) .   A random matrix  Φ  ∈ R k × d   satisfies   ( γ, β ) -JL   property   if   for   any   u, v   ∈ R d , P  [ |⟨ Φ u,  Φ v ⟩−⟨ u, v ⟩|  > γ  ∥ u ∥∥ v ∥ ]  ≤ β.

3. Stationary Points of Population Risk

For   the   population   gradient,   we   provide   a   linear   time   al- gorithm;   see   Algorithm   1   for   pseudocode.   It   is   a   noisy variant of SPIDER ( Fang et al. ,  2018 ), and utilizes a vari- ance reduction technique tailored to an underlying binary tree   structure.   Namely,   we   run   T   rounds,   where   at   the beginning   of   round   t   we   build   a   binary   tree   of   depth   D , whose nodes are denoted by  u t,s , where  s  ∈{ 0 ,  1 } D .   Every node   u t,s   is   associated   with   a   parameter   vector   w t,s   and a gradient estimate  ∇ t,s .   Next, we perform a Depth-First-

Search traversal of the tree.   We denote by DFS [ D ]  the set of nodes in the visiting order excluding the root, for example: DFS [2]   =   { u 0 , u 00 , u 01 , u 1 , u 10 , u 11 } .   When   a   left   child node is visited, it receives the same parameter vector and gradient estimator of the parent node.

Algorithm 1  Tree-based Private Spider Input:   S   =   ( x 1 , . . . , x n )   ∈X   n :   private   dataset,   ( ε, δ ) : privacy parameters,  T :   number of rounds,  b :   batch size at beginning of each round,  D :   depth of trees at each round,  β :   step-size parameter,   ˜ α :   accuracy parameter. 1:   w 0 ,ℓ (2 D − 1)   = 0 2:   for  t  = 1  to  T   do 3: Set  w t, ∅ =  w t − 1 ,ℓ (2 D − 1) 4: Draw a batch  S t, ∅ of  b  data points, set  S   ← S  \ S t, ∅ .

5: Set  σ 2 t, ∅ :=   8 L 2 0   log(1 . 25 /δ )

b 2 ε 2 . 6: ∇ t, ∅ = 1 b �

x ∈ S t, ∅ ∇ f  ( w t, ∅ ;  x )   +   g t, ∅ ,   where g t, ∅ ∼N � 0 ,  I d σ 2 t, ∅ � . 7: for  u t,s   ∈ DFS [ D ]  do 8: Let  s  =  � sc , where  c  ∈{ 0 ,  1 } . 9: if  c  = 0  then 10: ∇ t,s   =  ∇ t, � s 11: w t,s   =  w t, � s 12: else 13: Draw a batch  S t,s   of b 2 | s |   data points, set  S   ← S  \  S t,s .

14: Set noise variance  σ 2 t,s   :=   8 · 2 D β 2  log(1 . 25 /δ )

b 2 ε 2 .

15: ∆ t,s   =   2 | s |

b �

x ∈ S t,s ( ∇ f  ( w t,s ;  x ) −∇ f  ( w t, � s ;  x ))+

g t,s , where  g t,s ∼N � 0 ,  I d σ 2 t,s � . 16: ∇ t,s   =  ∇ t, � s  + ∆ t,s . 17: end if 18: if  | s |  =  D  (i.e,  u t,s   is a leaf)  then 19: if  ∥∇ t,s ∥≤ 2˜ α  then 20: Return  w t,s 21: end if 22: Let  u t,s +   be the next vertex in  DFS[ D ] .

23: Set  η t,s   := β 2 D/ 2 L 1 ∥∇ t,s ∥ 24: w t,s +   =  w t,s  − η t,s ∇ t,s . 25: end if 26: end for 27:   end for 28:   Return  w , chosen uniformly at random from  { w t,s   :  t  ∈ [ T ] , u t,s   is a leaf } .

On   the   other   hand,   when   a   right   child   node   is   visited,   it receives   a   fresh   set   of   samples   and   uses   it   to   update   the gradient   estimator   coming   from   the   parent   node.   Every time   a   leaf   node   is   reached,   a   gradient   step   is   performed using the gradient estimator associated to the leaf.   Finally, the parameter vector of a right child node comes from the gradient step performed at the right-most leaf in the left sub-

4



| 0                                                           | 1                                  | 2                                                       |
|:------------------------------------------------------------|:-----------------------------------|:--------------------------------------------------------|
| X . We assume that                                          | Input: S = (x1, . . . , xn) ∈ X n: | private dataset, (ε, δ):                                |
| the function w (cid:55)→ f (w; x)                           |                                    |                                                         |
| is L0-                                                      |                                    |                                                         |
| That                                                        |                                    | privacy parameters, T : number of rounds, b: batch size |
| is,                                                         |                                    |                                                         |
| for all x ∈ X and                                           |                                    |                                                         |
| Lipschitz and L1-smooth.                                    |                                    |                                                         |
| ∈ Rd,                                                       |                                    | at beginning of each round, D: depth of trees at each   |
| w1, w2                                                      |                                    |                                                         |
| |f (w1; x) − f (w2; x)| ≤ L0 ∥w1 − w2∥                      |                                    |                                                         |
| and ∥∇f (w1; x) − ∇f (w2; x)∥ ≤ L1 ∥w1 − w2∥. Given         |                                    | round, β: step-size parameter, ˜α: accuracy parameter.  |
| a dataset S ∈ X n of n points, we define the empirical      |                                    |                                                         |
|                                                             | 1: w0,ℓ(2D−1) = 0                  |                                                         |
| (cid:80)n                                                   |                                    |                                                         |
| risk as F (w; S) = 1                                        |                                    |                                                         |
| the                                                         |                                    |                                                         |
| i=1 f (w; xi). Assuming that                                | 2:                                 | for t = 1 to T do                                       |
| n                                                           |                                    |                                                         |
| data points are sampled i.i.d.                              |                                    |                                                         |
| from an unknown distri-                                     |                                    |                                                         |
|                                                             | 3:                                 | Set wt,∅ = wt−1,ℓ(2D−1)                                 |
| bution D,                                                   | 4:                                 | Draw a batch St,∅ of b data points, set S ← S \ St,∅.   |
| the population risk, denoted as F (w; D) is de-             |                                    |                                                         |
| fined as F (w; D) = Ex∼Df (w; x). Furthermore, we define    |                                    |                                                         |
| F0 = F (0; S) − minw∈Rd {F (w; S)} when discussing the      | 5:                                 | 0 log(1.25/δ)                                           |
|                                                             |                                    | .                                                       |
|                                                             |                                    | Set σ2                                                  |
|                                                             |                                    | t,∅ := 8L2                                              |
|                                                             |                                    | b2ε2                                                    |
| empirical case and similarly for the population loss when   |                                    | (cid:80)                                                |
|                                                             |                                    | ∇t,∅ =                                                  |
|                                                             | 6:                                 | 1b                                                      |
|                                                             |                                    | x∈St,∅ ∇f (wt,∅; x) + gt,∅, where                       |
| discussing stationary points of the population loss. We use |                                    |                                                         |
|                                                             |                                    | (cid:1).                                                |
|                                                             |                                    | gt,∅ ∼ N (cid:0)0, Idσ2                                 |
|                                                             |                                    | t,∅                                                     |
| w∗ to denote the population risk minimizer. Finally, we use |                                    |                                                         |
|                                                             | 7:                                 | for ut,s ∈ DFS [D] do                                   |
| the notation Id to denote the d × d identity matrix and use |                                    |                                                         |
|                                                             | 8:                                 | Let s = (cid:98)sc, where c ∈ {0, 1}.                   |
| [a] to denote the set {1, 2, ..., a} for a ≥ 1.             |                                    |                                                         |
|                                                             | 9:                                 | if c = 0 then                                           |
|                                                             | 10:                                | ∇t,s = ∇t,(cid:98)s                                     |




| 0                                                            | 1   | 2                                                         |
|:-------------------------------------------------------------|:----|:----------------------------------------------------------|
| discussing stationary points of the population loss. We use  |     |                                                           |
|                                                              |     | (cid:1).                                                  |
|                                                              |     | gt,∅ ∼ N (cid:0)0, Idσ2                                   |
|                                                              |     | t,∅                                                       |
| w∗ to denote the population risk minimizer. Finally, we use  |     |                                                           |
|                                                              | 7:  | for ut,s ∈ DFS [D] do                                     |
| the notation Id to denote the d × d identity matrix and use  |     |                                                           |
|                                                              | 8:  | Let s = (cid:98)sc, where c ∈ {0, 1}.                     |
| [a] to denote the set {1, 2, ..., a} for a ≥ 1.              |     |                                                           |
|                                                              | 9:  | if c = 0 then                                             |
|                                                              | 10: | ∇t,s = ∇t,(cid:98)s                                       |
| Stationary points:                                           |     |                                                           |
| Given a dataset S, our goal is to find                       |     |                                                           |
|                                                              | 11: | wt,s = wt,(cid:98)s                                       |
| an α-stationary point                                        |     |                                                           |
| w of either empirical or population                          |     |                                                           |
|                                                              | 12: | else                                                      |
| risk;                                                        |     | b                                                         |
| formally, ∥∇F ( ¯w; S)∥ ≤ α or ∥∇F ( ¯w; D)∥ ≤ α,            |     |                                                           |
|                                                              | 13: | 2|s| data points, set S ←                                 |
| respectively.                                                |     |                                                           |
|                                                              |     | S \ St,s.                                                 |
|                                                              | 14: | Set noise variance σ2                                     |
|                                                              |     | .                                                         |
|                                                              |     | t,s := 8·2Dβ2 log(1.25/δ)                                 |
| Differential Privacy (DP) (Dwork et al., 2006):              |     |                                                           |
| An algo-                                                     |     |                                                           |
|                                                              |     | (cid:80)                                                  |
|                                                              | 15: | ∆t,s = 2|s|                                               |
| rithm A is (ε, δ)-differentially private if                  |     | b                                                         |
| for all datasets                                             |     | (∇f (wt,s; x)−∇f (wt,(cid:98)s; x))+                      |
|                                                              |     | x∈St,s                                                    |
| S                                                            |     |                                                           |
| and S′                                                       |     |                                                           |
| differing                                                    |     |                                                           |
| in                                                           |     |                                                           |
| one                                                          |     |                                                           |
| data                                                         |     |                                                           |
| point                                                        |     |                                                           |
| and                                                          |     |                                                           |
| all                                                          |     |                                                           |
| events                                                       |     |                                                           |
|                                                              |     | (cid:1) .                                                 |
|                                                              |     | gt,s, where gt,s∼ N (cid:0)0, Idσ2                        |
| E                                                            |     | t,s                                                       |
| ≤                                                            |     |                                                           |
| in                                                           |     |                                                           |
| the                                                          |     |                                                           |
| range                                                        |     |                                                           |
| of                                                           |     |                                                           |
| the A, we                                                    |     |                                                           |
| have, P (A(S) ∈ E)                                           |     |                                                           |
| eεP (A(S′) ∈ E) + δ.                                         | 16: | ∇t,s = ∇t,(cid:98)s + ∆t,s.                               |
|                                                              | 17: | end if                                                    |
|                                                              | 18: | if |s| = D (i.e, ut,s is a leaf) then                     |
| Generalized Linear Models (GLMs):                            |     |                                                           |
| For data domain                                              |     |                                                           |
|                                                              | 19: | if ∥∇t,s∥ ≤ 2˜α then                                      |
| X ⊆ Rd and Y ⊆ R, a loss function f : Rd ×X ×Y → R is        |     |                                                           |
|                                                              | 20: | Return wt,s                                               |
| a GLM if f (w; (x, y)) = ϕy (⟨w, x⟩) for some function ϕy.   |     |                                                           |
|                                                              | 21: | end if                                                    |
| Our result for GLMs uses random matrices which satisfy       |     |                                                           |
|                                                              | 22: | Let ut,s+ be the next vertex in DFS[D].                   |
| the Johnson-Lindenstrauss (JL) property, defined as follows. |     |                                                           |
|                                                              |     | β                                                         |
|                                                              | 23: | Set ηt,s :=                                               |
| Definition 2.1 ((γ, β)-JL property). A random matrix Φ ∈     |     | 2D/2L1∥∇t,s∥                                              |
|                                                              | 24: | wt,s+ = wt,s − ηt,s∇t,s.                                  |
| satisfies                                                    |     |                                                           |
| (γ, β)-JL property if                                        |     |                                                           |
| for any u, v ∈ Rd,                                           |     |                                                           |
|                                                              | 25: | end if                                                    |
| P [|⟨Φu, Φv⟩ − ⟨u, v⟩| > γ ∥u∥ ∥v∥] ≤ β.                     |     |                                                           |
|                                                              | 26: | end for                                                   |
|                                                              | 27: | end for                                                   |
| 3. Stationary Points of Population Risk                      |     |                                                           |
|                                                              |     | 28: Return w, chosen uniformly at random from {wt,s : t ∈ |
|                                                              |     | [T ], ut,s is a leaf}.                                    |
| For                                                          |     |                                                           |
| the population gradient, we provide a linear                 |     |                                                           |
| time al-                                                     |     |                                                           |




| 0                                                                                       | 1                                                               |
|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                                 |
| d ≤ n2ϵ1/4 (i.e. essentially whenever the error is nontrivial).                         | Search traversal of the tree. We denote by DFS[D] the set of    |
| Further, we accomplish this with a much simpler analysis.                               | nodes in the visiting order excluding the root, for example:    |
|                                                                                         | DFS[2] = {u0, u00, u01, u1, u10, u11}. When a left child        |
|                                                                                         | node is visited,                                                |
|                                                                                         | it receives the same parameter vector and                       |
| 2. Preliminaries                                                                        |                                                                 |
|                                                                                         | gradient estimator of the parent node.                          |
| Let f                                                                                   |                                                                 |
| : Rd × X → R denote                                                                     |                                                                 |
| a                                                                                       |                                                                 |
| (loss)                                                                                  |                                                                 |
| function tak-                                                                           |                                                                 |
| ing as input,                                                                           | Algorithm 1 Tree-based Private Spider                           |
| the model parameter w and data point x ∈                                                |                                                                 |
| X . We assume that                                                                      | private dataset, (ε, δ):                                        |
| the function w (cid:55)→ f (w; x)                                                       | Input: S = (x1, . . . , xn) ∈ X n:                              |
| is L0-                                                                                  |                                                                 |
| That                                                                                    | privacy parameters, T : number of rounds, b: batch size         |
| is,                                                                                     |                                                                 |
| for all x ∈ X and                                                                       |                                                                 |
| Lipschitz and L1-smooth.                                                                |                                                                 |
| ∈ Rd,                                                                                   | at beginning of each round, D: depth of trees at each           |
| w1, w2                                                                                  |                                                                 |
| |f (w1; x) − f (w2; x)| ≤ L0 ∥w1 − w2∥                                                  |                                                                 |
| and ∥∇f (w1; x) − ∇f (w2; x)∥ ≤ L1 ∥w1 − w2∥. Given                                     | round, β: step-size parameter, ˜α: accuracy parameter.          |
| a dataset S ∈ X n of n points, we define the empirical                                  |                                                                 |
|                                                                                         | 1: w0,ℓ(2D−1) = 0                                               |
| (cid:80)n                                                                               |                                                                 |
| risk as F (w; S) = 1                                                                    |                                                                 |
| the                                                                                     |                                                                 |
| i=1 f (w; xi). Assuming that                                                            | 2:                                                              |
| n                                                                                       | for t = 1 to T do                                               |
| data points are sampled i.i.d.                                                          |                                                                 |
| from an unknown distri-                                                                 |                                                                 |
|                                                                                         | 3:                                                              |
|                                                                                         | Set wt,∅ = wt−1,ℓ(2D−1)                                         |
| bution D,                                                                               | 4:                                                              |
| the population risk, denoted as F (w; D) is de-                                         | Draw a batch St,∅ of b data points, set S ← S \ St,∅.           |
| fined as F (w; D) = Ex∼Df (w; x). Furthermore, we define                                |                                                                 |
| F0 = F (0; S) − minw∈Rd {F (w; S)} when discussing the                                  | 0 log(1.25/δ)                                                   |
|                                                                                         | 5:                                                              |
|                                                                                         | .                                                               |
|                                                                                         | Set σ2                                                          |
|                                                                                         | t,∅ := 8L2                                                      |
|                                                                                         | b2ε2                                                            |
| empirical case and similarly for the population loss when                               | (cid:80)                                                        |
|                                                                                         | ∇t,∅ =                                                          |
|                                                                                         | 1b                                                              |
|                                                                                         | 6:                                                              |
|                                                                                         | x∈St,∅ ∇f (wt,∅; x) + gt,∅, where                               |
| discussing stationary points of the population loss. We use                             |                                                                 |
|                                                                                         | (cid:1).                                                        |
|                                                                                         | gt,∅ ∼ N (cid:0)0, Idσ2                                         |
|                                                                                         | t,∅                                                             |
| w∗ to denote the population risk minimizer. Finally, we use                             |                                                                 |
|                                                                                         | 7:                                                              |
|                                                                                         | for ut,s ∈ DFS [D] do                                           |
| the notation Id to denote the d × d identity matrix and use                             |                                                                 |
|                                                                                         | 8:                                                              |
|                                                                                         | Let s = (cid:98)sc, where c ∈ {0, 1}.                           |
| [a] to denote the set {1, 2, ..., a} for a ≥ 1.                                         |                                                                 |
|                                                                                         | 9:                                                              |
|                                                                                         | if c = 0 then                                                   |
|                                                                                         | 10:                                                             |
|                                                                                         | ∇t,s = ∇t,(cid:98)s                                             |
| Stationary points:                                                                      |                                                                 |
| Given a dataset S, our goal is to find                                                  |                                                                 |
|                                                                                         | 11:                                                             |
|                                                                                         | wt,s = wt,(cid:98)s                                             |
| an α-stationary point                                                                   |                                                                 |
| w of either empirical or population                                                     |                                                                 |
|                                                                                         | 12:                                                             |
|                                                                                         | else                                                            |
| risk;                                                                                   | b                                                               |
| formally, ∥∇F ( ¯w; S)∥ ≤ α or ∥∇F ( ¯w; D)∥ ≤ α,                                       |                                                                 |
|                                                                                         | 13:                                                             |
|                                                                                         | 2|s| data points, set S ←                                       |
| respectively.                                                                           |                                                                 |
|                                                                                         | S \ St,s.                                                       |
|                                                                                         | 14:                                                             |
|                                                                                         | Set noise variance σ2                                           |
|                                                                                         | .                                                               |
|                                                                                         | t,s := 8·2Dβ2 log(1.25/δ)                                       |
| Differential Privacy (DP) (Dwork et al., 2006):                                         |                                                                 |
| An algo-                                                                                |                                                                 |
|                                                                                         | (cid:80)                                                        |
|                                                                                         | 15:                                                             |
|                                                                                         | ∆t,s = 2|s|                                                     |
| rithm A is (ε, δ)-differentially private if                                             | b                                                               |
| for all datasets                                                                        | (∇f (wt,s; x)−∇f (wt,(cid:98)s; x))+                            |
|                                                                                         | x∈St,s                                                          |
| S                                                                                       |                                                                 |
| and S′                                                                                  |                                                                 |
| differing                                                                               |                                                                 |
| in                                                                                      |                                                                 |
| one                                                                                     |                                                                 |
| data                                                                                    |                                                                 |
| point                                                                                   |                                                                 |
| and                                                                                     |                                                                 |
| all                                                                                     |                                                                 |
| events                                                                                  |                                                                 |
|                                                                                         | (cid:1) .                                                       |
|                                                                                         | gt,s, where gt,s∼ N (cid:0)0, Idσ2                              |
| E                                                                                       | t,s                                                             |
| ≤                                                                                       |                                                                 |
| in                                                                                      |                                                                 |
| the                                                                                     |                                                                 |
| range                                                                                   |                                                                 |
| of                                                                                      |                                                                 |
| the A, we                                                                               |                                                                 |
| have, P (A(S) ∈ E)                                                                      |                                                                 |
| eεP (A(S′) ∈ E) + δ.                                                                    | 16:                                                             |
|                                                                                         | ∇t,s = ∇t,(cid:98)s + ∆t,s.                                     |
|                                                                                         | 17:                                                             |
|                                                                                         | end if                                                          |
|                                                                                         | 18:                                                             |
|                                                                                         | if |s| = D (i.e, ut,s is a leaf) then                           |
| Generalized Linear Models (GLMs):                                                       |                                                                 |
| For data domain                                                                         |                                                                 |
|                                                                                         | 19:                                                             |
|                                                                                         | if ∥∇t,s∥ ≤ 2˜α then                                            |
| X ⊆ Rd and Y ⊆ R, a loss function f : Rd ×X ×Y → R is                                   |                                                                 |
|                                                                                         | 20:                                                             |
|                                                                                         | Return wt,s                                                     |
| a GLM if f (w; (x, y)) = ϕy (⟨w, x⟩) for some function ϕy.                              |                                                                 |
|                                                                                         | 21:                                                             |
|                                                                                         | end if                                                          |
| Our result for GLMs uses random matrices which satisfy                                  |                                                                 |
|                                                                                         | 22:                                                             |
|                                                                                         | Let ut,s+ be the next vertex in DFS[D].                         |
| the Johnson-Lindenstrauss (JL) property, defined as follows.                            |                                                                 |
|                                                                                         | β                                                               |
|                                                                                         | 23:                                                             |
|                                                                                         | Set ηt,s :=                                                     |
| Definition 2.1 ((γ, β)-JL property). A random matrix Φ ∈                                | 2D/2L1∥∇t,s∥                                                    |
|                                                                                         | 24:                                                             |
|                                                                                         | wt,s+ = wt,s − ηt,s∇t,s.                                        |
| satisfies                                                                               |                                                                 |
| (γ, β)-JL property if                                                                   |                                                                 |
| for any u, v ∈ Rd,                                                                      |                                                                 |
|                                                                                         | 25:                                                             |
|                                                                                         | end if                                                          |
| P [|⟨Φu, Φv⟩ − ⟨u, v⟩| > γ ∥u∥ ∥v∥] ≤ β.                                                |                                                                 |
|                                                                                         | 26:                                                             |
|                                                                                         | end for                                                         |
|                                                                                         | 27:                                                             |
|                                                                                         | end for                                                         |
| 3. Stationary Points of Population Risk                                                 |                                                                 |
|                                                                                         | 28: Return w, chosen uniformly at random from {wt,s : t ∈       |
|                                                                                         | [T ], ut,s is a leaf}.                                          |
| For                                                                                     |                                                                 |
| the population gradient, we provide a linear                                            |                                                                 |
| time al-                                                                                |                                                                 |
| gorithm;                                                                                |                                                                 |
| see Algorithm 1 for pseudocode.                                                         |                                                                 |
| It                                                                                      |                                                                 |
| is a noisy                                                                              |                                                                 |
| variant of SPIDER (Fang et al., 2018), and utilizes a vari-                             | On the other hand, when a right child node is visited,          |
|                                                                                         | it                                                              |
| ance reduction technique tailored to an underlying binary                               | receives a fresh set of samples and uses it                     |
|                                                                                         | to update the                                                   |
| tree structure. Namely, we run T rounds, where at                                       | gradient estimator coming from the parent node.                 |
| the                                                                                     | Every                                                           |
| beginning of                                                                            | time a leaf node is reached, a gradient step is performed       |
| round t we build a binary tree of depth D,                                              |                                                                 |
| whose nodes are denoted by ut,s, where s ∈ {0, 1}D. Every                               | using the gradient estimator associated to the leaf. Finally,   |
| node ut,s                                                                               | the parameter vector of a right child node comes from the       |
| is associated with a parameter vector wt,s and                                          |                                                                 |
| a gradient estimate ∇t,s. Next, we perform a Depth-First-                               | gradient step performed at the right-most leaf in the left sub- |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

tree of it.   The use of the binary tree structure is benefitial because every gradient estimator is updated at most  D  times within a round of  2 D   optimization steps, as opposed to the original SPIDER algorithm where the gradient estimators are updated at every optimization step. This way, we are able to perform the same number of optimization steps but adding substantially smaller amounts of noise, leading to a faster rate than the one we would get without using the tree.   In the following, we denote by  ℓ ( k )  the binary representation of any number  k   ∈ [0 ,  2 D   − 1]  and by  | s |  the depth of  u t,s   for any  t  ∈ [ T ] .

The   proposed   algorithm   is   similar   to   the   one   in   Section 5   of   ( Bassily   et   al. ,   2021b )   for   constrained   Differentially Private-Stochastic   Convex   Optimization   (DP-SCO),   with the   key   difference   that   Algorithm   1   executes   each   round with   fixed   depth   trees,   which   is   key   for   our   convergence analysis, whereas the prior work leverages convexity to con- struct   trees   that   increase   depth   by   one   at   each   round.   In addition, to choose the step-size in ( Bassily et al. ,  2021b ) the authors leverage the bounded diameter of the domain, while our step-size is chosen as that of ( Fang et al. ,  2018 ), i.e. normalized by the norm of the gradient estimator and proportional to the target accuracy.   This choice is crucial for controlling the sensitivity of the gradient variation esti- mator in the unconstrained setting, and consequently for the privacy analysis as well.   Our results are presented below and the proofs are deferred to Appendix  C .

Theorem   3.1   (Privacy   guarantee) .   For   any   ε, δ   ∈ [0 ,  1] , Algorithm  1  is  ( ε, δ ) -DP.

Theorem 3.2  (Accuracy guarantee) .   Let  p  ∈ (0 ,  1) ,  ε, δ   >

0 ,   b   =   max � n 2 / 3 , √ nd 1 / 4

√ ε � ,   D   be   such   that   D 2 D +1   =

b ,   T = n b ( D/ 2+1) ,   α   = √

2 L 0  max � 1 n 1 / 3  , � √ d nε � 1 / 2 � ,

β = α  min { 1 , √

bε √

d   } ,   and   ˜ α = ˜ Cα ,   where ˜ C =

256 log � 1 . 25 δ � log � 2 T  2 D +1

p � +   8 L 1 F 0 √

2 D ( D/ 2+1) 2 L 2 0 .   Then,

for any  n  ≥ max { √

d (   D

2   + 1) 2 /ε,  (  D

2   + 1) 3 } , with proba- bility  1 − p , Algorithm  1  ends in line  20 , returning an iterate w t,s  with

∥∇ F ( w t,s ;  D ) ∥≤ 3 √

2 L 0   ˜ C  max � 1 n 1 / 3  , � √ d nε

� 1 / 2 � .

Furthermore, Algorithm  1  has oracle complexity of  n .

4. Stationary Points of Empirical Risk

4.1. Efficient Algorithm with Faster Rate

The algorithm for our upper bound is a noisy version of the SpiderBoost algorithm ( Wang et al. ,  2019c ) 8 .   The algorithm

8 SpiderBoost itself is essentially the Spider algorithm ( Fang et al. ,  2018 ) with a different learning rate and analysis.

works by running a series of phases of length  q .   Each phase starts with a minibatch estimate of the gradient, and subse- quent gradient estimates within the phase are then computed by adding an estimate of the gradient variation.   The key to the analysis is to bound the error in the gradient estimate at each iteration.   Towards this end, we have the following generalization of the ( Wang et al. ,  2019c ) Lemma 1, which follows directly from ( Fang et al. ,  2018 ) Proposition 1. Lemma   4.1.   Consider   Algorithm   2 ,   and   for   any   t   ∈ { 0 , .., T }   let   s t = � t q � q . If   each   ∇ t   computed   in line   9   is   an   unbiased   estimate   of   ∇ F ( w t ;  S )   satisfying

E � ∥∇ s t   −∇ F ( w s t ;  S ) ∥ 2 � ≤ τ   2 1   and   each   ∆ t   computed in line  13  is an unbiased estimate of the gradient variation satisfying   E � ∥ ∆ t  − [ ∇ F ( w t ;  S )  −∇ F ( w t − 1 ;  S )] ∥ 2 � ≤

τ   2 2   ∥ w t   − w t − 1 ∥ 2 .   Then   for   any   t   ≥ s t   + 1 ,   the   iterates of Algorithm  2  satisfy

E � ∥∇ t  −∇ F ( w t ) ∥ 2 � ≤ τ   2 2

t �

k = s t +1 E � ∥ w k  − w k − 1 ∥ 2 � +  τ   2 1   .

For privacy, using smoothness we observe the sensitivity of the gradient variation estimate at iteration  t  is proportional to   β  ∥ w t  − w t − 1 ∥ .   Thus   we   can   apply   the   above   lemma with  τ   2 1   =   L 2 0 b 1   +  L 2 0 σ 2 1   and  τ  2 2   =   L 2 1 b 2   +  L 2 1 σ 2 2   (note the Gaus- sian noise in line  13  is drawn with variance scale at most σ 2 2   ∥ w t   − w t − 1 ∥ 2 ).   By   carefully   balancing   the   algorithm parameters, we are then able to obtain the following result. The full proof is deferred to Appendix  B.1 . Theorem 4.2  (Private Spiderboost ERM) .   Let  ε, δ   ∈ [0 ,  1] .

Let   n   ≥ max � ( L 0 ε ) 2

F 0 L 1 d  log(1 /δ ) ,

√

d  max { 1 , √ L 1 F 0 /L 0 }

ε

� .   Al-

gorithm   2   is   ( ε, δ ) -DP.   Further,   there   exist   settings   of T, η, q, b 1 , b 2   such   that   Algorithm   2   has   E  [ ∥∇ F ( ¯ w ;  S ) ∥ ] bounded as

O





� √ F 0 L 1 L 0 � d  log (1 /δ ) nε

� 2 / 3

+   L 0 � d  log (1 /δ )

nε





and oracle complexity   ˜ O � max �� n 5 / 3 ε 2 / 3

d 1 / 3 � , � nε √

d

� 2 �� .

Note   that   the   restriction   on   n   in   the   theorem   statement is   essentially   trivial   when   the   upper   bound   is   nontrivial. We   remark   that   the   case   where   the   dominant   error   term is  α   =   ˜ O �� √ d nε � 2 / 3 � ,   then we approximately have oracle

complexity   ˜ O � max � 1 α 3  ,   n

α �� .

4.2. Lower Bound

We now show a lower bound for the sample complexity of finding a stationary point under differential privacy in the un-

constrained setting, which shows that the  O � L 0 √ d  log(1 /δ )

nε �

5



| 0                                                                                       | 1                                                              |
|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                                |
| tree of it. The use of the binary tree structure is benefitial                          | works by running a series of phases of length q. Each phase    |
| because every gradient estimator is updated at most D times                             | starts with a minibatch estimate of the gradient, and subse-   |
| within a round of 2D optimization steps, as opposed to the                              | quent gradient estimates within the phase are then computed    |
| original SPIDER algorithm where the gradient estimators                                 | by adding an estimate of the gradient variation. The key to    |
| are updated at every optimization step. This way, we are able                           | the analysis is to bound the error in the gradient estimate    |
| to perform the same number of optimization steps but adding                             | at each iteration. Towards this end, we have the following     |
| substantially smaller amounts of noise, leading to a faster                             | generalization of the (Wang et al., 2019c) Lemma 1, which      |
| rate than the one we would get without using the tree. In the                           | follows directly from (Fang et al., 2018) Proposition 1.       |
| following, we denote by ℓ(k) the binary representation of                               | and for any                                                    |
|                                                                                         | t                                                              |
|                                                                                         | ∈                                                              |
|                                                                                         | Lemma 4.1. Consider Algorithm 2,                               |
| any number k ∈ [0, 2D − 1] and by |s| the depth of ut,s for                             | (cid:106) t                                                    |
|                                                                                         | let                                                            |
|                                                                                         | If                                                             |
|                                                                                         | computed                                                       |
|                                                                                         | in                                                             |
|                                                                                         | {0, .., T }                                                    |
|                                                                                         | =                                                              |
|                                                                                         | q.                                                             |
|                                                                                         | st                                                             |
|                                                                                         | each ∇t                                                        |
|                                                                                         | q                                                              |
| any t ∈ [T ].                                                                           |                                                                |
|                                                                                         | satisfying                                                     |
|                                                                                         | line 9 is an unbiased estimate of ∇F (wt; S)                   |
|                                                                                         | (cid:104)                                                      |
| The proposed algorithm is similar                                                       | E                                                              |
| to the one in Section                                                                   | ≤ τ 2                                                          |
|                                                                                         | ∥∇st − ∇F (wst; S)∥2(cid:105)                                  |
|                                                                                         | 1 and each ∆t computed                                         |
| 5 of (Bassily et al., 2021b) for constrained Differentially                             |                                                                |
|                                                                                         | in line 13 is an unbiased estimate of the gradient variation   |
| Private-Stochastic Convex Optimization (DP-SCO), with                                   |                                                                |
|                                                                                         | satisfying E                                                   |
|                                                                                         | ≤                                                              |
|                                                                                         | ∥∆t − [∇F (wt; S) − ∇F (wt−1; S)]∥2(cid:105)                   |
| the key difference that Algorithm 1 executes each round                                 |                                                                |
| with fixed depth trees, which is key for our convergence                                | the iterates                                                   |
|                                                                                         | τ 2                                                            |
|                                                                                         | Then for any t ≥ st + 1,                                       |
|                                                                                         | 2 ∥wt − wt−1∥2.                                                |
| analysis, whereas the prior work leverages convexity to con-                            | of Algorithm 2 satisfy                                         |
| struct                                                                                  |                                                                |
| trees that                                                                              |                                                                |
| increase depth by one at each round.                                                    |                                                                |
| In                                                                                      |                                                                |
|                                                                                         | t(cid:88)                                                      |
| addition,                                                                               | E (cid:2)∥∇t − ∇F (wt)∥2(cid:3) ≤ τ 2                          |
| to choose the step-size in (Bassily et al., 2021b)                                      | E (cid:2)∥wk − wk−1∥2(cid:3) + τ 2                             |
|                                                                                         | 1 .                                                            |
| the authors leverage the bounded diameter of the domain,                                | k=st+1                                                         |
| while our step-size is chosen as that of (Fang et al., 2018),                           |                                                                |
| i.e. normalized by the norm of the gradient estimator and                               | For privacy, using smoothness we observe the sensitivity of    |
| proportional to the target accuracy. This choice is crucial                             | the gradient variation estimate at iteration t is proportional |
| for controlling the sensitivity of the gradient variation esti-                         | to β ∥wt − wt−1∥. Thus we can apply the above lemma            |
| mator in the unconstrained setting, and consequently for the                            | + L2                                                           |
|                                                                                         | + L2                                                           |
|                                                                                         | with τ 2                                                       |
|                                                                                         | 1 = L2                                                         |
|                                                                                         | 0σ2                                                            |
|                                                                                         | 2 = L2                                                         |
|                                                                                         | 1σ2                                                            |
|                                                                                         | 1 and τ 2                                                      |
|                                                                                         | 2 (note the Gaus-                                              |
|                                                                                         | b1                                                             |
|                                                                                         | b2                                                             |
| privacy analysis as well. Our results are presented below                               |                                                                |
|                                                                                         | sian noise in line 13 is drawn with variance scale at most     |
| and the proofs are deferred to Appendix C.                                              |                                                                |
|                                                                                         | σ2                                                             |
|                                                                                         | 2 ∥wt − wt−1∥2). By carefully balancing the algorithm          |
| Theorem 3.1 (Privacy guarantee). For any ε, δ ∈ [0, 1],                                 | parameters, we are then able to obtain the following result.   |
| Algorithm 1 is (ε, δ)-DP.                                                               | The full proof is deferred to Appendix B.1.                    |
| Theorem 3.2 (Accuracy guarantee). Let p ∈ (0, 1), ε, δ >                                | Theorem 4.2 (Private Spiderboost ERM). Let ε, δ ∈ [0, 1].      |
|                                                                                         | √                                                              |
| √                                                                                       | √                                                              |
| (cid:110)                                                                               | (cid:26)                                                       |
| (cid:111)                                                                               | (cid:27)                                                       |
| nd1/4                                                                                   | L1F0/L0}                                                       |
|                                                                                         | d max{1,                                                       |
|                                                                                         | (L0ε)2                                                         |
| √                                                                                       | . Al-                                                          |
| 0, b = max                                                                              | Let n ≥ max                                                    |
| n2/3,                                                                                   |                                                                |
| , D be such that D2D+1 =                                                                |                                                                |
| ε                                                                                       | ε                                                              |
|                                                                                         | F0L1d log(1/δ) ,                                               |
| √                                                                                       |                                                                |
| n                                                                                       | gorithm 2 is                                                   |
| d                                                                                       | there                                                          |
| (cid:1)1/2(cid:9),                                                                      | exist                                                          |
| b, T                                                                                    | settings of                                                    |
| =                                                                                       | (ε, δ)-DP. Further,                                            |
| 2L0 max (cid:8)                                                                         |                                                                |
| b(D/2+1) , α =                                                                          |                                                                |
| nε                                                                                      |                                                                |
| n1/3 , (cid:0) √                                                                        |                                                                |
| √                                                                                       |                                                                |
| ˜                                                                                       | such that Algorithm 2 has E [∥∇F ( ¯w; S)∥]                    |
| ˜                                                                                       | T, η, q, b1, b2                                                |
| bε√                                                                                     |                                                                |
| and                                                                                     |                                                                |
| where                                                                                   |                                                                |
| },                                                                                      |                                                                |
| β                                                                                       |                                                                |
| =                                                                                       |                                                                |
| α min{1,                                                                                |                                                                |
| α                                                                                       |                                                                |
| =                                                                                       |                                                                |
| Cα,                                                                                     |                                                                |
| C                                                                                       |                                                                |
| =                                                                                       |                                                                |
| d                                                                                       |                                                                |
| √                                                                                       |                                                                |
| (cid:17)                                                                                | bounded as                                                     |
| (cid:16) 2T 2D+1                                                                        |                                                                |
| 2D(D/2+1)                                                                               |                                                                |
| .                                                                                       |                                                                |
| Then,                                                                                   |                                                                |
| 256 log (cid:0) 1.25                                                                    |                                                                |
| (cid:1) log                                                                             |                                                                |
| + 8L1F0                                                                                 |                                                                |
| δ                                                                                       |                                                                |
| p                                                                                       |                                                                |
| 2L2                                                                                     |                                                                |
| 0                                                                                       | (cid:33)2/3                                                    |
| √                                                                                       | (cid:32) √                                                     |
|                                                                                         |                                                              |
|                                                                                         |                                                              |
|                                                                                         | (cid:112)d log (1/δ)                                           |
|                                                                                         | (cid:112)d log (1/δ)                                           |
|                                                                                         | F0L1L0                                                         |
|                                                                                         | L0                                                             |
| for any n ≥ max{                                                                        |                                                                |
| 2 + 1)2/ε, ( D                                                                          |                                                                |
| 2 + 1)3}, with proba-                                                                   |                                                                |
|                                                                                         | +                                                              |
|                                                                                         | O                                                              |
| bility 1 − p, Algorithm 1 ends in line 20, returning an iterate                         | nε                                                             |
|                                                                                         | nε                                                             |
| wt,s with                                                                               |                                                                |
|                                                                                         | (cid:17)                                                       |
|                                                                                         | (cid:26)(cid:16) n5/3ε2/3                                      |
| √                                                                                       | (cid:16) nε√                                                   |
|                                                                                         | and oracle complexity ˜O                                       |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Algorithm 2   Private SpiderBoost

Input:   Dataset:   S   ∈X   n ,   Function:   f   :   R d   × X   �→ R , Learning   Rate:   η ,   Phase   Size:   q ,   Batch   Sizes   b 1 , b 2 , Privacy Parameters:   ( ε, δ ) , Iterations:   T 1:   w 0   = 0

2:   σ 1   = cL 0 √

log(1 /δ )

ε max � 1 b 1   , √

T √ qn � ,   where  c  is a uni- versal constant.

3:   σ 2   = cL 1 √

log(1 /δ )

ε max � 1 b 2   , √

T n �

4:   � σ 2   = 2 cL 0 √

log(1 /δ ) ε max � 1 b 2   , √

T n �

5:   for  t  = 0 , . . . , T   do 6: if mod   ( t, q ) = 0  then

7: Sample batch  S t  of size  b 1 8: Sample  g t   ∼N (0 ,  I d σ 2 1 )

9: ∇ t   = 1 b 1 �

x ∈ S t   ∇ f ( w t ;  x ) +  g t 10: else 11: Sample batch  S t  of size  b 2 12: g t   ∼N � 0 ,  I d  min � σ 2 2   ∥ w t   − w t − 1 ∥ 2  ,  � σ 2 2 ��

13: ∆ t   = 1 b 2 �

x ∈ S t   [ ∇ f ( w t ;  x )  −∇ f ( w t − 1 ;  x )]+ g t

14: ∇ t   =  ∇ t − 1  + ∆ t 15: end if 16: w t +1   =  w t  − η ∇ t 17:   end for 18:   return   ¯ w  uniformly at random from  { w 1 , . . . , w T  }

term in the rate given in Theorem  4.2  is necessary.   Further- more, as our lower bound holds for all levels of smoothness, it also shows that our rate in Theorem  4.2  is optimal in the (admittedly   uncommon)   regime   where   L 1   ≤

√

dL 2 0 F 0 nε   .   Our lower bound in fact holds even for convex functions.   Fur- thermore, this result implies the same lower bound (up to log factors) for the population gradient using the technique in ( Bassily et al. ,  2019 ), Appendix C.

Theorem 4.3.   Given  L 0 , L 1 , n, ε   =   O (1) ,  2 − Ω( n )   ≤ δ   ≤ 1 /n 1+Ω(1) ,   there   exists   an   L 0 -Lispchitz,   L 1 -smooth   (con- vex)   loss   f   :   R d   × X   → R   and   a   dataset   S   of   n   points such that any  ( ε, δ ) -DP algorithm run on  S   with output   ¯ w satisfies,

∥∇ F ( ¯ w ;  S ) ∥ = Ω

�

L 0  min

�

1 ,

� d  log (1 /δ )

nε

��

.

The proof is based on a reduction to DP mean estimation. Specifically, we consider a instance of the Huber loss func- tion for which the minimizer is the empirical mean of the dataset.   We   then   argue   that   close   to   the   minimizer,   the empirical   stationarity   is   lower   bounded   by   DP   mean   esti- mation bound ( Steinke & Ullman ,  2015 ), and far away, by construction, the empirical stationarity is  L 0 .

Proof of Theorem  4.3 .   For   any   r   >   0 ,   let   W r   denote   the ball of radius  r  centered at the origin. Let  B   =   L 0

L 1   . Consider the loss function:

f ( w ;  x ) =

� L 1 2   ∥ w  − x ∥ 2 if  ∥ w  − x ∥≤ B

L 0  ∥ w  − x ∥− L 2 0 2 L 1 otherwise

The   function   f ( w ;  x )   is   convex,   L 1 -smooth   and   L 0 - Lispchitz in  R d .   We restrict to datasets  S   =  { x i } n i =1   where x i   ∈W B/ 4   for all  i , and let  F ( w ;  S )   =   1

n � n i =1   f ( w ;  x i ) be the empirical risk on  S .   The unconstrained minimizer of F ( w ;  S )  is  w ∗ =   1

n � n i =1   x i   which lies in  W B/ 4 .

For any  w   ∈W 3 B/ 4 ,  w  lies in the quadratic region around all data points.   Hence, from  L 1 -strong convexity of  w   �→ F ( w ;  S )  on  W 3 B/ 4 , we have that whenever   ¯ w   ∈W 3 B/ 4 ,

∥∇ F ( ¯ w ;  S ) ∥∥ ¯ w  − w ∗ ∥≥⟨∇ F ( ¯ w ;  S ) , w ∗ − ¯ w ⟩

≥ F ( ¯ w ;  S )  − F ( w ∗ ;  S )

≥ L 1

2   ∥ ¯ w  − w ∗ ∥ 2  .

Let  E   be the event that   ¯ w   ∈W 3 B/ 4  and let  E E   denote the conditional expectation (conditioned on event  E ) operator. Then,

E E ∥∇ F ( ¯ w ;  S ) ∥≥ L 1

2   E  ∥ ¯ w  − w ∗ ∥

≥ L 1

2   Ω

�� L 0 4 L 1

� min

�

1 ,

� d  log (1 /δ )

nε

��

.

where the last inequality follows from known lower bounds for DP mean estimation ( Steinke & Ullman ,  2015 ;  Kamath & Ullman ,  2020 ).   We remark that the lower bound in the referenced work is for algorithms which produce outputs in   the   ball   of   the   same   radius   as   the   dataset,   i.e.   W B/ 4 . However,   a   simple   post-processing   argument   shows   that the same lower bound applies to algorithms which produce output in  W 3 B/ 4 .   Specifically, assuming the contrary, we simply   project   the   output   in   W 3 B/ 4   to   W B/ 4 :   privacy   is preserved by post-processing and the distance to the mean cannot increase by the non-expansiveness property of pro- jection   to   convex   sets,   hence   a   contradiction.   This   gives us,

E E  [ ∥∇ F ( ¯ w ;  S ) ∥ ]  ≥ Ω

�

L 0  min

�

1 ,

� d  log (1 /δ )

nε

��

Let   ˜ W   =   { w   :  ∥ w  − w ∗ ∥≤ B/ 2 } .   Since   ˜ W   ⊆W 3 B/ 4 , we have that the above conditional lower bound applies for ¯ w   ∈ ˜ W   as   well.   We   now   consider   ¯ w   ̸∈ ˜ W .   Let   w ′   be any   point   on   the   boundary   of   ˜ W ,   denoted   as   ∂ W .   Note that   w ′   lies   in   the   region   where,   for   any   data   point,   the

6



| 0                                    | 1                                                    | 2                                                               |
|:-------------------------------------|:-----------------------------------------------------|:----------------------------------------------------------------|
| Algorithm 2 Private SpiderBoost      |                                                      | Proof of Theorem 4.3. For any r > 0,                            |
|                                      |                                                      | let Wr denote the                                               |
|                                      |                                                      | ball of radius r centered at the origin. Let B = L0             |
|                                      |                                                      | . Consider                                                      |
| Input: Dataset: S ∈ X n, Function: f | : Rd × X (cid:55)→ R,                                | L1                                                              |
|                                      |                                                      | the loss function:                                              |
|                                      | Learning Rate:                                       |                                                                 |
|                                      | η, Phase Size:                                       |                                                                 |
|                                      | q, Batch Sizes b1, b2,                               |                                                                 |
|                                      | Privacy Parameters: (ε, δ), Iterations: T            |                                                                 |
|                                      |                                                      | (cid:40) L1                                                     |
|                                      |                                                      | ∥w − x∥2                                                        |
|                                      |                                                      | if ∥w − x∥ ≤ B                                                  |
| 1: w0 = 0                            |                                                      | 2                                                               |
|                                      | √                                                    | f (w; x) =                                                      |
|                                      | √                                                    |                                                                 |
|                                      | (cid:111)                                            | otherwise                                                       |
|                                      | log(1/δ)                                             | L0 ∥w − x∥ − L2                                                 |
|                                      | cL0                                                  |                                                                 |
|                                      | (cid:110) 1                                          |                                                                 |
|                                      | T√                                                   | 2L1                                                             |
|                                      | ,                                                    |                                                                 |
|                                      | max                                                  |                                                                 |
|                                      | , where c is a uni-                                  |                                                                 |
| 2: σ1 =                              |                                                      |                                                                 |
|                                      | b1                                                   |                                                                 |
|                                      | versal constant.                                     |                                                                 |
|                                      | √                                                    |                                                                 |
|                                      | (cid:111)                                            |                                                                 |
|                                      | log(1/δ)                                             | f (w; x)                                                        |
|                                      | cL1                                                  | The                                                             |
|                                      | (cid:110) 1                                          | function                                                        |
|                                      | T                                                    | is                                                              |
|                                      |                                                      | convex,                                                         |
|                                      |                                                      | L1-smooth                                                       |
|                                      |                                                      | and L0-                                                         |
| 3: σ2 =                              | max                                                  |                                                                 |
|                                      | ,                                                    |                                                                 |
|                                      | ε                                                    |                                                                 |
|                                      | n                                                    |                                                                 |
|                                      | b2                                                   | Lispchitz in Rd. We restrict to datasets S = {xi}n              |
|                                      | √                                                    |                                                                 |
|                                      | √                                                    | i=1 where                                                       |
|                                      | (cid:111)                                            |                                                                 |
|                                      | log(1/δ)                                             | (cid:80)n                                                       |
|                                      | 2cL0                                                 |                                                                 |
|                                      | (cid:110) 1                                          |                                                                 |
|                                      | T                                                    | xi ∈ WB/4 for all i, and let F (w; S) = 1                       |
|                                      | max                                                  | i=1 f (w; xi)                                                   |
|                                      | ,                                                    |                                                                 |
| 4: (cid:98)σ2 =                      | ε                                                    | n                                                               |
|                                      | n                                                    |                                                                 |
|                                      | b2                                                   |                                                                 |
|                                      |                                                      | be the empirical risk on S. The unconstrained minimizer of      |
| 5:                                   | for t = 0, . . . , T do                              |                                                                 |
|                                      |                                                      | (cid:80)n                                                       |
|                                      |                                                      | F (w; S) is w∗ = 1                                              |
| 6:                                   | if                                                   | i=1 xi which lies in WB/4.                                      |
|                                      | mod (t, q) = 0 then                                  | n                                                               |
| 7:                                   | Sample batch St of size b1                           |                                                                 |
|                                      |                                                      | For any w ∈ W3B/4, w lies in the quadratic region around        |
|                                      | Sample gt ∼ N (0, Idσ2                               | all data points. Hence, from L1-strong convexity of w (cid:55)→ |
| 8:                                   | 1)                                                   |                                                                 |
|                                      | (cid:80)                                             | F (w; S) on W3B/4, we have that whenever ¯w ∈ W3B/4,            |
| 9:                                   | ∇f (wt; x) + gt                                      |                                                                 |
|                                      | ∇t = 1                                               |                                                                 |
|                                      | x∈St                                                 |                                                                 |
|                                      | b1                                                   |                                                                 |
| 10:                                  | else                                                 |                                                                 |
|                                      |                                                      | ∥∇F ( ¯w; S)∥ ∥ ¯w − w∗∥ ≥ ⟨∇F ( ¯w; S), w∗ − ¯w⟩               |
| 11:                                  | Sample batch St of size b2                           |                                                                 |
|                                      | (cid:16)                                             |                                                                 |
|                                      | (cid:110)                                            |                                                                 |
|                                      | (cid:111)(cid:17)                                    |                                                                 |
|                                      |                                                      | ≥ F ( ¯w; S) − F (w∗; S)                                        |
| 12:                                  | σ2                                                   |                                                                 |
|                                      | gt ∼ N                                               |                                                                 |
|                                      | 0, Id min                                            |                                                                 |
|                                      | 2 ∥wt − wt−1∥2 , (cid:98)σ2                          |                                                                 |
|                                      | (cid:80)                                             | L1                                                              |
| 13:                                  | ∆t = 1                                               | ≥                                                               |
|                                      | [∇f (wt; x) − ∇f (wt−1; x)]+gt                       | ∥ ¯w − w∗∥2 .                                                   |
|                                      | x∈St                                                 |                                                                 |
|                                      | b2                                                   |                                                                 |
|                                      |                                                      | 2                                                               |
| 14:                                  | ∇t = ∇t−1 + ∆t                                       | Let E be the event that ¯w ∈ W3B/4 and let EE denote the        |
| 15:                                  | end if                                               | conditional expectation (conditioned on event E) operator.      |
| 16:                                  | wt+1 = wt − η∇t                                      | Then,                                                           |
| 17:                                  | end for                                              |                                                                 |
|                                      |                                                      | L1                                                              |
| 18:                                  | return ¯w uniformly at random from {w1, . . . , wT } |                                                                 |
|                                      |                                                      | E ∥ ¯w − w∗∥                                                    |
|                                      |                                                      | EE∥∇F ( ¯w; S)∥ ≥                                               |
|                                      |                                                      | 2                                                               |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

corresponding loss is a quadratic function.   Hence, by direct computation,  ∇ F ( w ′ ;  S ) =  L 1  ( w ′   − w ∗ ) .   Therefore,

⟨∇ F ( w ′ ) , w ′   − w ∗ ⟩ =  L 1  ∥ w ′   − w ∗ ∥ 2   =   L 1 B 2

4 .

We now apply gradient monotonicity to obtain the following (see Lemma  A.1 , Appendix  A ),

E E c  ∥∇ F ( ¯ w ;  S ) ∥≥ L 1 B 2

4 ·   2 B   =   L 0

2   ,

where  E c   denotes the complement set of  E . We combine the above bounds using the law of total expectation as follows,

E [ ∥∇ F ( ¯ w ;  S ) ∥ ] =  E E [ ∥∇ F ( ¯ w ;  S ) ∥ ] P {  ¯ w   ∈ E }  +  E E c [ ∥∇ F ( ¯ w ;  S ) ∥ ] P {  ¯ w   ∈ E c }

= Ω � L 0  min � 1 ,

� d  log (1 /δ )

nε

�� P ( ¯ w   ∈ E ) + Ω( L 0 ) P ( ¯ w   ∈ E c )

= Ω � L 0  min � 1 ,

� d  log (1 /δ )

nε

�� .

This completes the proof.

Challenges for Further Rate Improvements: Given the above   lower   bound,   the   question   arises   as   to   whether   the ˜ O �� √ d nε   ] 2 / 3 � term can be improved.   An informal argument using the oracle complexity lower bound of ( Arjevani et al. , 2019 ) suggests several major challenges in obtaining further rate improvements. A more detailed version of the following discussion can be found in Appendix  B.2 .

Consider methods which ensure privacy by directly priva- tizing the gradient/gradient variation queries.   The aim of such methods is to design some private stochastic first or- der   oracle,   O ε ′ ,δ ′ ,   such   that   a   set   of   G   queries   to   O ε ′ ,δ ′ satisfies   ( ε, δ ) -DP,   and   use   this   oracle   in   some   optimiza- tion   algorithm   A ( O ε ′ ,δ ′ ) .   Such   a   setup   encapsulates   nu- merous results in the convex setting ( Bassily et al. ,  2019 ; Kulkarni et al. ,  2021 ), and is even more dominant in non- convex settings ( Wang et al. ,  2017 ;  Zhou et al. ,  2020 ;  Abadi et   al. ,   2016 ).   Under   advanced   composition   based   argu- ments, to make  G  calls to such a private oracle one needs ε ′   ≤ ε/ √

G .   Now, standard fingerprinting code arguments suggest lower bounds on the level of accuracy of any such private oracle ( Steinke & Ullman ,  2015 ).   Specifically, with- out leveraging further problem structure beyond Lipschitz- ness, one needs the gradient estimation error to be at least

τ 1   = Ω � L 0 √ Gd  log(1 /δ )

nε � .   A similar argument suggests the error in the gradient variation between iterates  w, w ′   must

at   least   τ 2  ∥ w  − w ′ ∥ =   Ω � L 1 ∥ w − w ′ ∥ √ Gd  log(1 /δ ) nε � .   Now consider some optimization algorithm,  A , which takes as input   a   stochastic   oracle   O   for   some   smooth   function   L . The lower bound of ( Arjevani et al. ,  2019 ) suggests that if  A makes at most  G  queries to  O  (as a black box) the algorithm

satisfies   E  [ ∥∇L ( A ( O )) ∥ ]   =   Ω �� F 0 τ 2 τ 1 G � 1 / 3  + τ 1 √

G

� .   If O   is   a   private   oracle   satisfying   the   previously   mentioned conditions, we would then have under the setting of  τ 1  and τ 2   suggested by privacy that the convergence guarantee for E  [ ∥∇L ( A ( O )) ∥ ]  is lower bounded as

Ω





� √ F 0 L 1 L 0 � d  log (1 /δ ) nε

� 2 / 3

+   L 0 � d  log (1 /δ )

nε



 .

This indicates a substantial challenge for future rate improve- ments, as alternative methods which avoid private gradients (see e.g.   ( Feldman et al. ,  2020 )) rely crucially on stability guarantees arising from convexity.

5. Stationary Points in the Convex Setting

Algorithm 3  Recursive Regularization Input:   Dataset  S , loss function  f , steps  T ,  { λ t } t ,  { R t } t , PrivateSubRoutine ,   number   of   steps   of   sub-routine { K t } , selector functions  {S t ( · ) } t , step size  { η t } t , noise variances  { σ t } t 1:   w 0   = 0 ,  n 0   = 1 2:   Define   function   ( w, x )   �→ f   (0) ( w ;  x )   =   f ( w ;  x )   + λ 0

2   ∥ w  − w 0 ∥ 2

3:   for  t  = 1  to  T   − 1  do 4: n t   =  n t − 1  + � | S |

T �

5: ¯ w t = PrivateSubRoutine ( S n t − 1 : n t , f   ( t − 1) , R t , K t , η t ,  S t ( · ) , σ t ) 6: Define function ( w, x ) �→ f   ( t ) ( w ;  x ) = f   ( t − 1) ( w ;  x ) +   λ t

2   ∥ w  − ¯ w t ∥ 2

7:   end for Output:   ¯ w   =   ¯ w T

In this section, we additionally assume that the loss function is convex.   The motivation for this is two-fold:   firstly, this setting   has   recently   gained   attention   in   a   non-private   set- ting ( Nesterov ,  2012 ;  Allen-Zhu ,  2018 ;  Foster et al. ,  2019 ). Secondly, in this setting we are able to establish tightly the sample complexity of approximate stationary points.

Our method is based on the recursive regularization tech- nique proposed in ( Allen-Zhu ,  2018 ), and further improved by ( Foster et al. ,  2019 ). The main idea, as the name suggests, is to recursively regularize the objective and optimize it via some   solver.   For   the   DP   setting,   the   key   idea   is   to   use   a private sub-routine as the inner solver.   Furthermore, while a solver for the unconstrained problem suffices non-privately, we need to carefully increase the radius of the constrained set over which the solver operates.

Theorem   5.1.   Let   L 0 , L 1 , ε, δ   >   0 ,   d, n   ∈ N .   Let   w   �→ f ( w ;  x )  be an  L 0 -Lipschitz  L 1 -smooth convex function for

7



| 0                                                            | 1                                                                                       | 2                                                          | 3          | 4     | 5   | 6   |
|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------|:-----------|:------|:----|:----|
|                                                              | Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                            |            |       |     |     |
|                                                              |                                                                                         | (cid:16)(cid:0) F0τ2τ1                                     | (cid:1)1/3 |       |     |     |
| corresponding loss is a quadratic function. Hence, by direct | satisfies E [∥∇L(A(O))∥] = Ω                                                            |                                                            |            | + τ1√ | .   | If  |
|                                                              |                                                                                         | G                                                          |            | G     |     |     |
| computation, ∇F (w′; S) = L1 (w′ − w∗). Therefore,           |                                                                                         |                                                            |            |       |     |     |
|                                                              |                                                                                         | O is a private oracle satisfying the previously mentioned  |            |       |     |     |
|                                                              |                                                                                         | conditions, we would then have under the setting of τ1 and |            |       |     |     |
| L1B2                                                         |                                                                                         |                                                            |            |       |     |     |
| .                                                            |                                                                                         | τ2 suggested by privacy that the convergence guarantee for |            |       |     |     |
| ⟨∇F (w′), w′ − w∗⟩ = L1 ∥w′ − w∗∥2 =                         |                                                                                         |                                                            |            |       |     |     |
| 4                                                            |                                                                                         |                                                            |            |       |     |     |
|                                                              | E [∥∇L(A(O))∥] is lower bounded as                                                      |                                                            |            |       |     |     |
| We now apply gradient monotonicity to obtain the following   |                                                                                         |                                                            |            |       |     |     |




| 0                                                            | 1       | 2                                                         | 3                                         | 4      | 5                                  | 6           | 7           |
|:-------------------------------------------------------------|:--------|:----------------------------------------------------------|:------------------------------------------|:-------|:-----------------------------------|:------------|:------------|
|                                                              |         |                                                           | loss function f , steps T , {λt}t, {Rt}t, |        |                                    |             |             |
| This completes the proof.                                    |         | PrivateSubRoutine, number of                              |                                           |        | steps of                           |             | sub-routine |
|                                                              |         | {Kt}, selector functions {St(·)}t, step size {ηt}t, noise |                                           |        |                                    |             |             |
| Challenges for Further Rate Improvements:                    |         |                                                           |                                           |        |                                    |             |             |
| Given the                                                    |         |                                                           |                                           |        |                                    |             |             |
|                                                              |         | variances {σt}t                                           |                                           |        |                                    |             |             |
| above lower bound,                                           |         | 1: w0 = 0, n0 = 1                                         |                                           |        |                                    |             |             |
| the question arises as to whether the                        |         |                                                           |                                           |        |                                    |             |             |
| O(cid:0)(cid:2) √                                            |         |                                                           |                                           |        |                                    |             |             |
| nε ]2/3(cid:1) term can be improved. An informal argument    |         | 2: Define function (w, x)                                 |                                           |        | (cid:55)→ f (0)(w; x) = f (w; x) + |             |             |
| using the oracle complexity lower bound of (Arjevani et al., |         | λ0                                                        |                                           |        |                                    |             |             |
|                                                              |         | ∥w − w0∥2                                                 |                                           |        |                                    |             |             |
|                                                              |         | 2                                                         |                                           |        |                                    |             |             |
| 2019) suggests several major challenges in obtaining further | 3:      | for t = 1 to T − 1 do                                     |                                           |        |                                    |             |             |
|                                                              |         | (cid:106) |S|                                             | (cid:107)                                 |        |                                    |             |             |
| rate improvements. A more detailed version of the following  |         |                                                           |                                           |        |                                    |             |             |
|                                                              | 4:      | nt = nt−1 +                                               |                                           |        |                                    |             |             |
|                                                              |         | T                                                         |                                           |        |                                    |             |             |
| discussion can be found in Appendix B.2.                     |         |                                                           |                                           |        |                                    |             |             |
|                                                              | 5:      | =                                                         | PrivateSubRoutine(Snt−1:nt, f (t−1), Rt,  |        |                                    |             |             |
|                                                              |         | wt                                                        |                                           |        |                                    |             |             |
| Consider methods which ensure privacy by directly priva-     |         | Kt, ηt, St(·), σt)                                        |                                           |        |                                    |             |             |
| tizing the gradient/gradient variation queries. The aim of   | 6:      | Define                                                    |                                           | (w, x) | (cid:55)→                          | f (t)(w; x) | =           |
|                                                              |         | function                                                  |                                           |        |                                    |             |             |
| such methods is to design some private stochastic first or-  |         | f (t−1)(w; x) + λt                                        | 2 ∥w − ¯wt∥2                              |        |                                    |             |             |
| der oracle, Oε′,δ′, such that a set of G queries to Oε′,δ′   | 7:      | end for                                                   |                                           |        |                                    |             |             |
| satisfies (ε, δ)-DP, and use this oracle in some optimiza-   | Output: | w = ¯wT                                                   |                                           |        |                                    |             |             |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

all   x .   Let   R t   = � √ 2 � t  ∥ w ∗ ∥ , λ t   =   2 t λ ,   η t   = log( K t )

λ t K t   ,

T   = � log 2 � L 1 λ �� ,  σ 2 t   =   64 L 2 0 K 2 t   log(1 /δ ) n 2 ε 2 , and  S t ( { w k } k ) = 1 � Kt k =1 (1 − η t λ t ) − k � K t k =1   (1  − η t λ t ) − k  w k .

1.   (Optimal   rate)   Algorithm   3   run   with   NoisyGD (Algorithm 7 in Appendix D ) as the Pri- vateSubRoutine with above parameter set- tings and λ = L 2 0 L 1 ∥ w ∗ ∥ min � 1 n , d n 2 ε 2 � and

K t = max � L 1 + λ t

λ t log � L 1 + λ t

λ t

� , n 2 ε 2 � L 2 0 λ + L 3 / 2 1 �

T   2 λdL 2 0   log(1 /δ )

�

satisfies   ( ε, δ ) -DP,   and   given   a   dataset   S   of   n   i.i.d. samples from  D , outputs   ¯ w  such that

E  ∥∇ F ( ¯ w ;  D ) ∥ =   ˜ O

� L 0 √ n   +   L 0 √

d nε

�

.

Furthermore,   the   above   rate   is   tight   up   to   poly- logarithmic factors. 2.   (Linear time rate) Algorithm 3 run with PhasedSGD (Algorithm 5 ) as the PrivateSub- Routine   with   with   above   parameter   settings   and λ = max � L 2 0 L 1 ∥ w ∗ ∥ 2   min � 1 n , d n 2 ε 2 � ,   L 1  log( n )

n � and

K t   =  ⌊ n

T   ⌋ satisfies  ( ε, δ ) -DP and given a dataset  S   of  n i.i.d.   samples from  D , in linear time, outputs   ¯ w  with

E  ∥∇ F ( ¯ w ;  D ) ∥ =   ˜ O

� L 0 √ n   +   L 0 √

d nε +   L 1  ∥ w ∗ ∥ √ n

�

.

The   proof   of   the   above   result   is   deferred   to   Appendix   D . For   the   tightness   of   the   rate,   the   necessity   of   the   second term   L 0 √

d nε is   due   to   our   DP   empirical   stationarity   lower bound, Theorem  4.3 .   For the first “non-private” term   L 0 √ n , even   though   ( Foster   et   al. ,   2019 )   proved   a   sample   com- plexity lower bound, their instance is not Lipschitz and has d  = Ω( n  log ( n )) , hence not applicable. To remedy this, we give a new lower bound construction with a Lispchitz func- tion in  d   =   1 , Theorem  A.2  in Appendix  A . The polylog dependence on  L 1   and  ∥ w ∗ ∥ in the upper bounds, is consis- tent with the non-private sample complexity in ( Foster et al. , 2019 ).

The   second   result   is   a   linear   time   method   which   has   an additional   L 1  ∥ w ∗ ∥ / √ n   term.   Firstly,   if   the   smoothness parameter is  small enough , then there is no overhead; this small-enough smoothness is precisely the regime in which we have linear time methods with optimal rates for smooth DP-SCO ( Feldman et al. ,  2020 ).   More importantly, ( Fos- ter   et   al. ,   2019 )   showed   that   even   in   the   non-private   set- ting,   a   polynomial   dependence   on   L 1  ∥ w ∗ ∥ is   necessary in the stochastic oracle model.   However, the optimal non- private term, shown in ( Foster et al. ,  2019 ), is  L 1  ∥ w ∗ ∥ /n 2 , achieved   by   accelerated   methods.   Improving   this   depen- dency, if possible, is an interesting direction for future work.

6. Generalized Linear Models

In this section, we assume that the loss function is a general- ized linear model (GLM),  f ( w ; ( x, y )) =  ϕ y  ( ⟨ w, x ⟩ ) . Also, assume the norm of data points  x  are bounded by  ∥X∥ and the function  ϕ y   :   R   → R  is  L 0 -Lipschitz and  L 1 -smooth for all  y .   Furthermore, let  rank  denote the rank of design matrix  X   ∈ R n × d .

Algorithm 4  JL method Input:   Dataset  S , function  ( z, y )  �→ ϕ y ( z ) , Algorithm  A , JL matrix  Φ  ∈ R k × d ,  L 0 ,  L 1 ,  ∥X∥ 1:   ˜ w = A (( z, y ) �→ ϕ y ( z ) ,  { (Φ x i , y i ) } n i =1   , 2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2) Output:   ¯ w   = Φ ⊤ ˜ w

Algorithm   4   is   a   generic   method   which   converts   any   for smooth Lipschitz losses with an empirical stationarity guar- antee to get dimension-independent rates on population sta- tionarity for smooth Lipschitz GLMs.   This algorithm is the JL   method   from   ( Arora   et   al. ,   2022 )   used   therein   to   give excess   risk   bounds   for   convex   GLM.   We   note   that   while the   JL   method   there   is   limited   to   the   Noisy   GD   method, ours is a black-box reduction.   Furthermore, unlike ( Arora et al. ,  2022 ), we show that the JL method gives finer rank based guarantees by leveraging the fact it acts as an oblivi- ous approximate subspace embedding (see Definition  E.1  in Appendix  E ).

Theorem 6.1.   Let  A  be an  ( ε, δ ) -DP algorithm which when run   on   a   L 1 -smooth   L 0 -Lipschitz   function   on   a   dataset S   =   { ( x i , y i ) } n i =1   where   x i   ∈X   ⊆ R d ,   guarantees E  [ ∥∇ F ( A ( S );  S ) ∥ ]  ≤ g ( d, n, L 1 , L 0 , ε, δ )  and  ∥A ( S ) ∥≤ poly ( n, d, L 0 , L 1 )  with probability at least  1  − 1 √ n .   Then, Algorithm  4  run with

k   = � min � arg min j ∈ N

� g ( j, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2)

+   L 0  ∥X∥ log ( n ) √ j

� ,  rank  log � 2 n δ

���

on a  L 0 -Lipschitz,  L 1 -smooth GLM loss, is  ( ε, δ ) -DP. Fur- thermore,   given   a   dataset   of   n   i.i.d   samples   from   D ,   its output   ¯ w  has  E  [ ∥∇ F ( ¯ w ;  D ) ∥ ]  bounded as

˜ O � L 0  ∥X∥ √ n +  g ( k, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2) �

The expression for  k   above comes from the subspace em- bedding property of JL, and from balancing the dimension of   the   embedding   with   respect   to   the   error   of   A   and   the approximation   error   of   the   JL   embedding.   The   proof   is based on the properties of JL matrices:   oblivious subspace embedding and preservation of norms, together with a new

8



| 0                                                                                       | 1                                                               |
|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                                 |
| ,                                                                                       | 6. Generalized Linear Models                                    |
| all x.                                                                                  |                                                                 |
| 2(cid:1)t                                                                               |                                                                 |
| Let Rt = (cid:0)√                                                                       |                                                                 |
| ∥w∗∥ , λt = 2tλ, ηt = log(Kt)                                                           |                                                                 |
| λtKt                                                                                    |                                                                 |
| log(1/δ)                                                                                |                                                                 |
| 0K2                                                                                     |                                                                 |
| (cid:0) L1                                                                              |                                                                 |
| (cid:1)(cid:5), σ2                                                                      |                                                                 |
| T = (cid:4)log2                                                                         |                                                                 |
| t = 64L2                                                                                |                                                                 |
| , and St({wk}k) =                                                                       | In this section, we assume that the loss function is a general- |
| λ                                                                                       |                                                                 |
| n2ε2                                                                                    |                                                                 |
| 1                                                                                       |                                                                 |
| (cid:80)Kt                                                                              |                                                                 |
| (cid:80)Kt                                                                              | ized linear model (GLM), f (w; (x, y)) = ϕy (⟨w, x⟩). Also,     |
| k=1 (1 − ηtλt)−k wk.                                                                    |                                                                 |
| k=1(1−ηtλt)−k                                                                           |                                                                 |
|                                                                                         | assume the norm of data points x are bounded by ∥X ∥ and        |
| 1.                                                                                      | the function ϕy                                                 |
| (Optimal                                                                                | : R → R is L0-Lipschitz and L1-smooth                           |
| rate)                                                                                   |                                                                 |
| Algorithm                                                                               |                                                                 |
| 3                                                                                       |                                                                 |
| run                                                                                     |                                                                 |
| with                                                                                    |                                                                 |
| NoisyGD                                                                                 |                                                                 |
| (Algorithm                                                                              | for all y. Furthermore, let rank denote the rank of design      |
| 7                                                                                       |                                                                 |
| in                                                                                      |                                                                 |
| Appendix                                                                                |                                                                 |
| D)                                                                                      |                                                                 |
| as                                                                                      |                                                                 |
| the                                                                                     |                                                                 |
| Pri-                                                                                    |                                                                 |
| vateSubRoutine                                                                          | matrix X ∈ Rn×d.                                                |
| with                                                                                    |                                                                 |
| above                                                                                   |                                                                 |
| parameter                                                                               |                                                                 |
| set-                                                                                    |                                                                 |
| d                                                                                       |                                                                 |
| (cid:1)                                                                                 |                                                                 |
| tings                                                                                   |                                                                 |
| and                                                                                     |                                                                 |
| and                                                                                     |                                                                 |
| λ                                                                                       |                                                                 |
| =                                                                                       |                                                                 |
| n ,                                                                                     |                                                                 |
| n2ε2                                                                                    |                                                                 |
| L1∥w∗∥ min (cid:0) 1                                                                    |                                                                 |
|                                                                                         | Algorithm 4 JL method                                           |
| (cid:19)                                                                                |                                                                 |
| (cid:18)                                                                                |                                                                 |
| n2ε2(cid:16)                                                                            |                                                                 |
| L2                                                                                      |                                                                 |
| (cid:17)                                                                                |                                                                 |
| 0λ+L3/2                                                                                 |                                                                 |
| L1+λt                                                                                   |                                                                 |
| (cid:16) L1+λt                                                                          |                                                                 |
| log                                                                                     | Input: Dataset S, function (z, y) (cid:55)→ ϕy(z), Algorithm A, |
| ,                                                                                       |                                                                 |
| =                                                                                       |                                                                 |
| max                                                                                     |                                                                 |
| Kt                                                                                      |                                                                 |
| T 2λdL2                                                                                 |                                                                 |
| λt                                                                                      |                                                                 |
| λt                                                                                      |                                                                 |
| 0 log(1/δ)                                                                              |                                                                 |
|                                                                                         | JL matrix Φ ∈ Rk×d, L0, L1, ∥X ∥                                |
| satisfies                                                                               |                                                                 |
| (ε, δ)-DP, and given a dataset S of n i.i.d.                                            |                                                                 |
|                                                                                         | w                                                               |
|                                                                                         | =                                                               |
|                                                                                         | A((z, y)                                                        |
|                                                                                         | (cid:55)→                                                       |
|                                                                                         | ϕy(z), {(Φxi, yi)}n                                             |
|                                                                                         | 1:                                                              |
|                                                                                         | i=1 ,                                                           |
| samples from D, outputs ¯w such that                                                    |                                                                 |
|                                                                                         | 2L0 ∥X ∥ , 2L1 ∥X ∥2 , ε, δ/2)                                  |
| √                                                                                       |                                                                 |
| (cid:32)                                                                                |                                                                 |
| (cid:33)                                                                                |                                                                 |
| d                                                                                       | w = Φ⊤ ˜w                                                       |
|                                                                                         | Output:                                                         |
| L0                                                                                      |                                                                 |
| L0√                                                                                     |                                                                 |
| E ∥∇F ( ¯w; D)∥ = ˜O                                                                    |                                                                 |
| .                                                                                       |                                                                 |
| +                                                                                       |                                                                 |
| nε                                                                                      |                                                                 |
| n                                                                                       |                                                                 |
|                                                                                         | Algorithm 4 is a generic method which converts any for          |
| Furthermore,                                                                            |                                                                 |
| the                                                                                     |                                                                 |
| above                                                                                   |                                                                 |
| rate                                                                                    |                                                                 |
| is                                                                                      |                                                                 |
| tight                                                                                   |                                                                 |
| up                                                                                      |                                                                 |
| to                                                                                      |                                                                 |
| poly-                                                                                   |                                                                 |
|                                                                                         | smooth Lipschitz losses with an empirical stationarity guar-    |
| logarithmic factors.                                                                    |                                                                 |
|                                                                                         | antee to get dimension-independent rates on population sta-     |
| 2.                                                                                      |                                                                 |
| (Linear                                                                                 |                                                                 |
| time                                                                                    |                                                                 |
| rate)                                                                                   |                                                                 |
| Algorithm                                                                               |                                                                 |
| 3                                                                                       |                                                                 |
| run                                                                                     |                                                                 |
| with                                                                                    |                                                                 |
|                                                                                         | tionarity for smooth Lipschitz GLMs. This algorithm is the      |
| (Algorithm                                                                              |                                                                 |
| 5)                                                                                      |                                                                 |
| as                                                                                      |                                                                 |
| the                                                                                     |                                                                 |
| PrivateSub-                                                                             |                                                                 |
| PhasedSGD                                                                               |                                                                 |
|                                                                                         | JL method from (Arora et al., 2022) used therein to give        |
| Routine                                                                                 |                                                                 |
| with                                                                                    |                                                                 |
| with                                                                                    |                                                                 |
| above                                                                                   |                                                                 |
| parameter                                                                               |                                                                 |
| settings                                                                                |                                                                 |
| and                                                                                     |                                                                 |
| (cid:17)                                                                                | excess risk bounds for convex GLM. We note that while           |
| (cid:16)                                                                                |                                                                 |
| d                                                                                       |                                                                 |
| and                                                                                     | the JL method there is limited to the Noisy GD method,          |
| λ                                                                                       |                                                                 |
| =                                                                                       |                                                                 |
| max                                                                                     |                                                                 |
| n ,                                                                                     |                                                                 |
| n2ε2                                                                                    |                                                                 |
| n                                                                                       |                                                                 |
| L1∥w∗∥2 min (cid:0) 1                                                                   |                                                                 |
| Kt = ⌊ n                                                                                | ours is a black-box reduction. Furthermore, unlike (Arora       |
| T ⌋ satisfies (ε, δ)-DP and given a dataset S of n                                      |                                                                 |
| i.i.d. samples from D, in linear time, outputs ¯w with                                  | et al., 2022), we show that the JL method gives finer rank      |
| √                                                                                       | based guarantees by leveraging the fact it acts as an oblivi-   |
| (cid:32)                                                                                |                                                                 |
| (cid:33)                                                                                |                                                                 |
| d                                                                                       |                                                                 |
| L0                                                                                      |                                                                 |
| L1 ∥w∗∥                                                                                 |                                                                 |
| L0√                                                                                     | ous approximate subspace embedding (see Definition E.1 in       |
| √                                                                                       |                                                                 |
| E ∥∇F ( ¯w; D)∥ = ˜O                                                                    |                                                                 |
| +                                                                                       |                                                                 |
| +                                                                                       |                                                                 |
| .                                                                                       |                                                                 |
| nε                                                                                      | Appendix E).                                                    |
| n                                                                                       |                                                                 |
| n                                                                                       |                                                                 |
|                                                                                         | Theorem 6.1. Let A be an (ε, δ)-DP algorithm which when         |
| The proof of the above result                                                           |                                                                 |
| is deferred to Appendix D.                                                              |                                                                 |
|                                                                                         | run on a L1-smooth L0-Lipschitz function on a dataset           |
| For                                                                                     |                                                                 |
| the tightness of                                                                        |                                                                 |
| the rate,                                                                               |                                                                 |
| the necessity of                                                                        |                                                                 |
| the second                                                                              |                                                                 |
|                                                                                         | ∈ X ⊆ Rd, guarantees                                            |
|                                                                                         | S = {(xi, yi)}n                                                 |
| √                                                                                       | i=1 where xi                                                    |
| term L0                                                                                 |                                                                 |
| is due to our DP empirical stationarity lower                                           | E [∥∇F (A(S); S)∥] ≤ g(d, n, L1, L0, ε, δ) and ∥A(S)∥ ≤         |
| nε                                                                                      |                                                                 |
| bound, Theorem 4.3. For the first “non-private” term L0√                                | poly(n, d, L0, L1) with probability at least 1 − 1√             |
| n ,                                                                                     | n . Then,                                                       |
| even though (Foster et al., 2019) proved a sample com-                                  | Algorithm 4 run with                                            |
| plexity lower bound, their instance is not Lipschitz and has                            |                                                                 |
|                                                                                         | (cid:24)                                                        |
|                                                                                         | (cid:18)                                                        |
|                                                                                         | (cid:18)                                                        |
| d = Ω (n log (n)), hence not applicable. To remedy this, we                             |                                                                 |
|                                                                                         | k =                                                             |
|                                                                                         | min                                                             |
|                                                                                         | arg min                                                         |
|                                                                                         | g(j, n, 2L0 ∥X ∥ , 2L1 ∥X ∥2 , ε, δ/2)                          |
| give a new lower bound construction with a Lispchitz func-                              | j∈N                                                             |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

uniform convergence result for gradients of Lipschitz GLMs. The full proof is deferred to Appendix  E .

Below,   we   instantiate   the   above   with   our   proposed   algo- rithms.

Corollary 6.2.   Under the assumptions of Theorem  6.1 , Al- gorithm  4  run with  A  as

1.   Private   Spiderboost   (Alg.   2 )   yields   ∥∇ F ( ¯ w ;  D ) ∥ =

˜ O � 1 √ n   + min �� √ rank

nε � 2 / 3 , 1 ( nε ) 2 / 5

�� .

2.   Algorithm  3  with NoisyGD as PrivateSubRoutine, un- der   the   additional   assumption   that   w   �→ f ( w ; ( x, y )) is convex for all x, y , yields ∥∇ F ( ¯ w ;  D ) ∥ = ˜ O � 1 √ n   + min � √ rank

nε , 1 √ nε �� .

We remark that the above technique also gives bounds on empirical stationarity.   In particular, the first term 1 √ n , in the above   guarantees,   is   the   uniform   convergence   bound   and the second term is the bound on empirical stationarity.

Acknowledgements

RA   and   EU   are   supported,   in   part,   by   NSF   BIGDATA award IIS-1838139 and NSF CAREER award IIS-1943251. RB’s   and   MM’s   research   is   supported   by   NSF   CAREER Award   2144532   and   NSF   Award   AF-1908281.   CG   and TG’s research was partially supported by INRIA Associate Teams project, FONDECYT 1210362 grant, ANID Anillo ACT210005 grant, and National Center for Artificial Intelli- gence CENIA FB210017, Basal ANID.

References

Abadi,   M.,   Chu,   A.,   Goodfellow,   I.,   McMahan,   H.   B., Mironov,   I.,   Talwar,   K.,   and   Zhang,   L. Deep   learn- ing   with   differential   privacy. In   23rd   ACM   Confer- ence on Computer and Communications Security , CCS ’16, pp. 308–318, New York, NY, USA, 2016. Associa- tion for Computing Machinery.   ISBN 9781450341394. doi:   10.1145/2976749.2978318.   URL  https://doi. org/10.1145/2976749.2978318 .

Allen-Zhu, Z.   How to make the gradients small stochasti- cally:   Even faster convex and nonconvex sgd.   Advances in Neural Information Processing Systems , 31, 2018.

Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N.,   and Woodworth,   B.   Lower bounds for non-convex stochastic optimization, 2019.

Arora,   R.,   Bassily,   R.,   Guzm ´ an,   C.,   Menart,   M.,   and   Ul- lah, E.   Differentially private generalized linear models revisited.   arXiv preprint arXiv:2205.03014 , 2022.

Asi,   H.,   Feldman,   V.,   Koren,   T.,   and   Talwar,   K.   Private stochastic convex optimization:   Optimal rates in l1 geom- etry.   In  International Conference on Machine Learning , pp. 393–403. PMLR, 2021.

Bassily, R., Smith, A., and Thakurta, A.   Private empirical risk   minimization:   Efficient   algorithms   and   tight   error bounds.   In  2014 IEEE 55th Annual Symposium on Foun- dations of Computer Science , pp. 464–473. IEEE, 2014.

Bassily,   R.,   Feldman,   V.,   Talwar,   K.,   and Guha Thakurta, A.   Private stochastic convex optimization with optimal rates. In   Wallach,   H.,   Larochelle,   H.,   Beygelz- imer,   A.,   d'Alch ´ e-Buc,   F.,   Fox,   E.,   and   Garnett, R.   (eds.), Advances   in   Neural   Information   Pro- cessing Systems , volume 32. Curran Associates, Inc., 2019. URL https://proceedings. neurips.cc/paper/2019/file/ 3bd8fdb090f1f5eb66a00c84dbc5ad51-Paper. pdf .

Bassily,   R.,   Guzm ´ an,   C.,   and   Menart,   M.   Differentially private   stochastic   optimization:   New   results   in   convex and non-convex settings.  Advances in Neural Information Processing Systems , 34, 2021a.

Bassily, R., Guzman, C., and   Nandi, A. Non- euclidean   differentially   private   stochastic   convex   op- timization. In   Belkin,   M.   and   Kpotufe,   S.   (eds.), Proceedings   of   Thirty   Fourth   Conference   on   Learn- ing   Theory , volume   134   of   Proceedings   of   Ma- chine   Learning   Research ,   pp.   474–499.   PMLR,   15–19 Aug   2021b. URL   https://proceedings.mlr. press/v134/bassily21a.html .

Bousquet, O. and Elisseeff, A.   Stability and generalization. The Journal of Machine Learning Research , 2:499–526, 2002.

Bun,   M.,   Dwork,   C.,   Rothblum,   G.   N.,   and   Steinke,   T. Composable and versatile privacy via truncated cdp.   In Proceedings   of   the   50th   Annual   ACM   SIGACT   Sympo- sium on Theory of Computing , STOC 2018, pp. 74–86, New   York,   NY,   USA,   2018.   Association   for   Comput- ing   Machinery.   ISBN   9781450355599.   doi:   10.1145/ 3188745.3188946. URL   https://doi.org/10. 1145/3188745.3188946 .

Carmon, Y., Duchi, J. C., Hinder, O., and Sidford, A.   ”con- vex until proven guilty”:   Dimension-free acceleration of gradient descent on non-convex functions.   In  Proceed- ings   of   the   34th   International   Conference   on   Machine Learning - Volume 70 , ICML’17, pp. 654–663. JMLR.org, 2017.

Chaudhuri, K., Monteleoni, C., and Sarwate, A. D.   Differ- entially private empirical risk minimization.   Journal of Machine Learning Research , 12(Mar):1069–1109, 2011.

9



| 0                                                                                       | 1                                                            |
|:----------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                              |
| uniform convergence result for gradients of Lipschitz GLMs.                             | Asi, H., Feldman, V., Koren, T., and Talwar, K.              |
|                                                                                         | Private                                                      |
| The full proof is deferred to Appendix E.                                               | stochastic convex optimization: Optimal rates in l1 geom-    |
|                                                                                         | etry.                                                        |
|                                                                                         | In International Conference on Machine Learning,             |
| Below, we instantiate the above with our proposed algo-                                 |                                                              |
|                                                                                         | pp. 393–403. PMLR, 2021.                                     |
| rithms.                                                                                 |                                                              |
|                                                                                         | Bassily, R., Smith, A., and Thakurta, A. Private empirical   |
| Corollary 6.2. Under the assumptions of Theorem 6.1, Al-                                |                                                              |
|                                                                                         | risk minimization: Efficient algorithms and tight error      |
| gorithm 4 run with A as                                                                 |                                                              |
|                                                                                         | bounds.                                                      |
|                                                                                         | In 2014 IEEE 55th Annual Symposium on Foun-                  |
|                                                                                         | dations of Computer Science, pp. 464–473. IEEE, 2014.        |
| 1. Private Spiderboost                                                                  |                                                              |
| (Alg. 2) yields ∥∇F ( ¯w; D)∥ =                                                         |                                                              |
| (cid:18)(cid:16) √                                                                      |                                                              |
| (cid:17)2/3                                                                             |                                                              |
| rank                                                                                    | Bassily, R., Feldman, V., Talwar, K., and Guha Thakurta,     |
| 1                                                                                       |                                                              |
| 1                                                                                       |                                                              |
| ˜                                                                                       |                                                              |
| √                                                                                       |                                                              |
| nε                                                                                      |                                                              |
| n + min                                                                                 |                                                              |
| (nε)2/5                                                                                 |                                                              |
|                                                                                         | A. Private stochastic convex optimization with optimal       |
| 2. Algorithm 3 with NoisyGD as PrivateSubRoutine, un-                                   | rates.                                                       |
|                                                                                         | In Wallach,                                                  |
|                                                                                         | H.,                                                          |
|                                                                                         | Larochelle,                                                  |
|                                                                                         | H.,                                                          |
|                                                                                         | Beygelz-                                                     |
| der the additional assumption that w (cid:55)→ f (w; (x, y))                            | imer,                                                        |
|                                                                                         | A.,                                                          |
|                                                                                         | d'Alch´e-Buc,                                                |
|                                                                                         | F.,                                                          |
|                                                                                         | Fox,                                                         |
|                                                                                         | E.,                                                          |
|                                                                                         | and Garnett,                                                 |
| is                                                                                      | Advances                                                     |
| convex                                                                                  | in                                                           |
| for                                                                                     | Neural                                                       |
| all                                                                                     | Information                                                  |
| yields                                                                                  | Pro-                                                         |
| x, y,                                                                                   | R.                                                           |
| ∥∇F ( ¯w; D)∥                                                                           | (eds.),                                                      |
| =                                                                                       |                                                              |
| (cid:16) √                                                                              |                                                              |
| (cid:17)(cid:17)                                                                        |                                                              |
| rank                                                                                    | cessing                                                      |
| ˜                                                                                       | Systems,                                                     |
| (cid:16) 1√                                                                             | volume                                                       |
| 1√                                                                                      | 32.                                                          |
|                                                                                         | Curran                                                       |
|                                                                                         | Associates,                                                  |
| .                                                                                       |                                                              |
| ,                                                                                       |                                                              |
| nε                                                                                      |                                                              |
| n + min                                                                                 |                                                              |
| nε                                                                                      |                                                              |
|                                                                                         | https://proceedings.                                         |
|                                                                                         | Inc.,                                                        |
|                                                                                         | 2019.                                                        |
|                                                                                         | URL                                                          |
|                                                                                         | neurips.cc/paper/2019/file/                                  |
| We remark that the above technique also gives bounds on                                 |                                                              |
|                                                                                         | 3bd8fdb090f1f5eb66a00c84dbc5ad51-Paper.                      |
| empirical stationarity. In particular, the first term 1√                                |                                                              |
| n , in the                                                                              | pdf.                                                         |
| above guarantees,                                                                       |                                                              |
| is the uniform convergence bound and                                                    |                                                              |
| the second term is the bound on empirical stationarity.                                 | Bassily, R., Guzm´an, C., and Menart, M. Differentially      |
|                                                                                         | private stochastic optimization: New results in convex       |
|                                                                                         | and non-convex settings. Advances in Neural Information      |
| Acknowledgements                                                                        |                                                              |
|                                                                                         | Processing Systems, 34, 2021a.                               |
| RA and EU are supported,                                                                |                                                              |
| in part, by NSF BIGDATA                                                                 |                                                              |
|                                                                                         | Bassily,                                                     |
|                                                                                         | R.,                                                          |
|                                                                                         | Guzman,                                                      |
|                                                                                         | C.,                                                          |
|                                                                                         | and                                                          |
|                                                                                         | Nandi,                                                       |
|                                                                                         | A.                                                           |
|                                                                                         | Non-                                                         |
| award IIS-1838139 and NSF CAREER award IIS-1943251.                                     |                                                              |
|                                                                                         | euclidean differentially private                             |
|                                                                                         | stochastic                                                   |
|                                                                                         | convex op-                                                   |
| RB’s and MM’s research is supported by NSF CAREER                                       |                                                              |
|                                                                                         | timization.                                                  |
|                                                                                         | In Belkin, M.                                                |
|                                                                                         | and Kpotufe,                                                 |
|                                                                                         | S.                                                           |
|                                                                                         | (eds.),                                                      |
| Award 2144532 and NSF Award AF-1908281. CG and                                          |                                                              |
|                                                                                         | Proceedings                                                  |
|                                                                                         | of Thirty Fourth Conference                                  |
|                                                                                         | on Learn-                                                    |
| TG’s research was partially supported by INRIA Associate                                |                                                              |
|                                                                                         | ing                                                          |
|                                                                                         | Proceedings                                                  |
|                                                                                         | of Ma-                                                       |
|                                                                                         | Theory,                                                      |
|                                                                                         | volume                                                       |
|                                                                                         | 134                                                          |
|                                                                                         | of                                                           |
| Teams project, FONDECYT 1210362 grant, ANID Anillo                                      |                                                              |
|                                                                                         | chine Learning Research, pp. 474–499. PMLR, 15–19            |
| ACT210005 grant, and National Center for Artificial Intelli-                            |                                                              |
|                                                                                         | Aug 2021b.                                                   |
|                                                                                         | URL https://proceedings.mlr.                                 |
| gence CENIA FB210017, Basal ANID.                                                       |                                                              |
|                                                                                         | press/v134/bassily21a.html.                                  |
|                                                                                         | Bousquet, O. and Elisseeff, A. Stability and generalization. |
| References                                                                              |                                                              |
|                                                                                         | The Journal of Machine Learning Research, 2:499–526,         |
| Abadi, M., Chu, A., Goodfellow,                                                         | 2002.                                                        |
| I., McMahan, H. B.,                                                                     |                                                              |
| Mironov,                                                                                |                                                              |
| I., Talwar, K.,                                                                         |                                                              |
| and Zhang, L.                                                                           |                                                              |
| Deep learn-                                                                             |                                                              |
|                                                                                         | Bun, M., Dwork, C., Rothblum, G. N., and Steinke, T.         |
| ing with differential privacy.                                                          |                                                              |
| In 23rd ACM Confer-                                                                     |                                                              |
|                                                                                         | Composable and versatile privacy via truncated cdp.          |
|                                                                                         | In                                                           |
| ence on Computer and Communications Security, CCS                                       |                                                              |
|                                                                                         | Proceedings of                                               |
|                                                                                         | the 50th Annual ACM SIGACT Sympo-                            |
| ’16, pp. 308–318, New York, NY, USA, 2016. Associa-                                     |                                                              |
|                                                                                         | sium on Theory of Computing, STOC 2018, pp. 74–86,           |
| tion for Computing Machinery.                                                           |                                                              |
| ISBN 9781450341394.                                                                     |                                                              |
|                                                                                         | New York, NY, USA, 2018. Association for Comput-             |
| doi: 10.1145/2976749.2978318. URL https://doi.                                          |                                                              |
|                                                                                         | ing Machinery.                                               |
|                                                                                         | ISBN 9781450355599.                                          |
|                                                                                         | doi: 10.1145/                                                |
| org/10.1145/2976749.2978318.                                                            |                                                              |
|                                                                                         | 3188745.3188946.                                             |
|                                                                                         | URL https://doi.org/10.                                      |
|                                                                                         | 1145/3188745.3188946.                                        |
| Allen-Zhu, Z. How to make the gradients small stochasti-                                |                                                              |
| cally: Even faster convex and nonconvex sgd. Advances                                   |                                                              |
|                                                                                         | Carmon, Y., Duchi, J. C., Hinder, O., and Sidford, A. ”con-  |
| in Neural Information Processing Systems, 31, 2018.                                     |                                                              |
|                                                                                         | vex until proven guilty”: Dimension-free acceleration of     |
|                                                                                         | gradient descent on non-convex functions.                    |
|                                                                                         | In Proceed-                                                  |
| Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro,                          |                                                              |
|                                                                                         | ings of                                                      |
|                                                                                         | the 34th International Conference on Machine                 |
| N., and Woodworth, B. Lower bounds for non-convex                                       |                                                              |
|                                                                                         | Learning - Volume 70, ICML’17, pp. 654–663. JMLR.org,        |
| stochastic optimization, 2019.                                                          |                                                              |
|                                                                                         | 2017.                                                        |
| Arora, R., Bassily, R., Guzm´an, C., Menart, M., and Ul-                                | Chaudhuri, K., Monteleoni, C., and Sarwate, A. D. Differ-    |
| lah, E. Differentially private generalized linear models                                | entially private empirical risk minimization. Journal of     |
| revisited. arXiv preprint arXiv:2205.03014, 2022.                                       | Machine Learning Research, 12(Mar):1069–1109, 2011.          |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Cohen, M. B.   Nearly tight oblivious subspace embeddings by trace inequalities. In  Proceedings of the twenty-seventh annual ACM-SIAM symposium on Discrete algorithms , pp. 278–287. SIAM, 2016.

Cutkosky,   A.   and   Orabona,   F. Momentum-based   vari- ance   reduction   in   non-convex   sgd. In   Wallach,   H., Larochelle,   H.,   Beygelzimer,   A.,   d'Alch ´ e-Buc,   F., Fox,   E.,   and   Garnett,   R.   (eds.),   Advances   in   Neural Information Processing Systems , volume 32. Curran As- sociates, Inc., 2019.   URL  https://proceedings. neurips.cc/paper/2019/file/ b8002139cdde66b87638f7f91d169d96-Paper. pdf .

Diakonikolas, J. and Guzm ´ an, C.   Complementary compos- ite minimization, small gradients in general norms, and applications, 2023.

Duchi, J. Lecture notes for statistics 311/elec- trical engineering 377. URL: https://stanford. edu/class/stats311/Lectures/full   notes. pdf. Last visited on , 2:23, 2016.

Dwork,   C.   and   Roth,   A.   The   algorithmic   foundations   of differential privacy.   Foundations and Trends® in Theo- retical Computer Science , 9(3–4):211–407, 2014.

Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrat- ing noise to sensitivity in private data analysis.   In  Theory of cryptography conference , pp. 265–284. Springer, 2006.

Fang,   C.,   Li,   C.   J.,   Lin,   Z.,   and   Zhang,   T. Spider: Near-optimal   non-convex   optimization   via   stochastic path-integrated   differential   estimator. In   Bengio,   S., Wallach,   H.,   Larochelle,   H.,   Grauman,   K.,   Cesa- Bianchi, N., and Garnett, R. (eds.),  Advances in Neural Information Processing Systems , volume 31. Curran As- sociates, Inc., 2018.   URL  https://proceedings. neurips.cc/paper/2018/file/ 1543843a4723ed2ab08e18053ae6dc5b-Paper. pdf .

Feldman, V., Koren, T., and Talwar, K.   Private stochastic convex optimization:   optimal rates in linear time.   In  Pro- ceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , pp. 439–449, 2020.

Foster,   D.   J.,   Sekhari,   A.,   and   Sridharan,   K. Uniform convergence   of   gradients   for   non-convex   learning   and optimization.   In   Bengio,   S.,   Wallach,   H.,   Larochelle, H.,   Grauman,   K.,   Cesa-Bianchi,   N.,   and   Garnett, R.   (eds.), Advances   in   Neural   Information   Pro- cessing Systems , volume 31. Curran Associates, Inc., 2018. URL https://proceedings. neurips.cc/paper/2018/file/ 59ab3ba90ae4b4ab84fe69de7b8e3f5f-Paper. pdf .

Foster,   D. J.,   Sekhari,   A.,   Shamir,   O.,   Srebro,   N.,   Sridha- ran, K., and Woodworth, B.   The complexity of making the gradient small in stochastic convex optimization.   In Conference on Learning Theory , pp. 1319–1345. PMLR, 2019.

Ge, R., Lee, J. D., and Ma, T.   Matrix completion has no spu- rious local minimum. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and Garnett, R. (eds.),  Advances in Neural Information Processing Systems , volume 29. Curran As- sociates, Inc., 2016.   URL  https://proceedings. neurips.cc/paper/2016/file/ 7fb8ceb3bd59c7956b1df66729296a4c-Paper. pdf .

Ghadimi, S. and Lan, G.   Stochastic first-and zeroth-order methods for nonconvex stochastic programming.   SIAM Journal on Optimization , 23(4):2341–2368, 2013.

Ghadimi,   S.   and   Lan,   G.   Accelerated   gradient   methods for   nonconvex   nonlinear   and   stochastic   programming. Mathematical Programming , 156(1):59–99, 2016.

Jain, P. and Thakurta, A.   (near) dimension independent risk bounds for differentially private learning.   In  ICML , 2014.

Jain, P., Kothari, P., and Thakurta, A.   Differentially private online learning.   In  25th Annual Conference on Learning Theory (COLT) , pp. 24.1–24.34, 2012.

Jin, C., Netrapalli, P., Ge, R., Kakade, S. M., and Jordan, M.   I. A   short   note   on   concentration   inequalities   for random vectors with subgaussian norm.   arXiv preprint arXiv:1902.03736 , 2019.

Kamath, G. and Ullman, J.   A primer on private statistics. arXiv preprint arXiv:2005.00010 , 2020.

Kifer, D., Smith, A., and Thakurta, A.   Private convex empir- ical risk minimization and high-dimensional regression. In  Conference on Learning Theory , pp. 25–1, 2012.

Kulkarni,   J.,   Lee,   Y.   T.,   and   Liu,   D.   Private   non-smooth erm   and   sco   in   subquadratic   steps. In   Ranzato,   M., Beygelzimer,   A.,   Dauphin,   Y.,   Liang,   P.,   and Vaughan, J. W. (eds.),  Advances in Neural Information Processing Systems ,   volume   34,   pp.   4053–4064.   Curran   Asso- ciates,   Inc.,   2021. URL   https://proceedings. neurips.cc/paper/2021/file/ 211c1e0b83b9c69fa9c4bdede203c1e3-Paper. pdf .

Lan, G.   First-order and stochastic optimization methods for machine learning .   Springer, 2020.

Ma,   C.,   Wang,   K.,   Chi,   Y.,   and   Chen,   Y.   Implicit   regu- larization in nonconvex statistical estimation:   Gradient descent converges linearly for phase retrieval and matrix

10



| 0                                                                                       | 1                                                              |
|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                                |
| Cohen, M. B. Nearly tight oblivious subspace embeddings                                 | Foster, D. J., Sekhari, A., Shamir, O., Srebro, N., Sridha-    |
| by trace inequalities. In Proceedings of the twenty-seventh                             | ran, K., and Woodworth, B. The complexity of making            |
| annual ACM-SIAM symposium on Discrete algorithms,                                       | the gradient small in stochastic convex optimization.          |
|                                                                                         | In                                                             |
| pp. 278–287. SIAM, 2016.                                                                | Conference on Learning Theory, pp. 1319–1345. PMLR,            |
|                                                                                         | 2019.                                                          |
| Cutkosky, A. and Orabona, F.                                                            |                                                                |
| Momentum-based vari-                                                                    |                                                                |
| ance                                                                                    | Ge, R., Lee, J. D., and Ma, T. Matrix completion has no spu-   |
| reduction in non-convex sgd.                                                            |                                                                |
| In Wallach, H.,                                                                         |                                                                |
| Larochelle,                                                                             | rious local minimum. In Lee, D., Sugiyama, M., Luxburg,        |
| H.,                                                                                     |                                                                |
| Beygelzimer,                                                                            |                                                                |
| A.,                                                                                     |                                                                |
| d'Alch´e-Buc,                                                                           |                                                                |
| F.,                                                                                     |                                                                |
| in Neural                                                                               | U., Guyon, I., and Garnett, R. (eds.), Advances in Neural      |
| Fox, E.,                                                                                |                                                                |
| and Garnett, R.                                                                         |                                                                |
| (eds.), Advances                                                                        |                                                                |
| Information Processing Systems, volume 32. Curran As-                                   | Information Processing Systems, volume 29. Curran As-          |
| sociates, Inc., 2019. URL https://proceedings.                                          | sociates, Inc., 2016. URL https://proceedings.                 |
| neurips.cc/paper/2019/file/                                                             | neurips.cc/paper/2016/file/                                    |
| b8002139cdde66b87638f7f91d169d96-Paper.                                                 | 7fb8ceb3bd59c7956b1df66729296a4c-Paper.                        |
| pdf.                                                                                    | pdf.                                                           |
| Diakonikolas, J. and Guzm´an, C. Complementary compos-                                  |                                                                |
|                                                                                         | Ghadimi, S. and Lan, G. Stochastic first-and zeroth-order      |
| ite minimization, small gradients in general norms, and                                 |                                                                |
|                                                                                         | methods for nonconvex stochastic programming. SIAM             |
| applications, 2023.                                                                     |                                                                |
|                                                                                         | Journal on Optimization, 23(4):2341–2368, 2013.                |
| Duchi,                                                                                  |                                                                |
| J.                                                                                      |                                                                |
| Lecture                                                                                 |                                                                |
| notes                                                                                   |                                                                |
| for                                                                                     |                                                                |
| statistics                                                                              |                                                                |
| 311/elec-                                                                               |                                                                |
|                                                                                         | Ghadimi, S. and Lan, G. Accelerated gradient methods           |
| URL:                                                                                    |                                                                |
| https://stanford.                                                                       |                                                                |
| trical                                                                                  |                                                                |
| engineering                                                                             |                                                                |
| 377.                                                                                    |                                                                |
|                                                                                         | for nonconvex nonlinear and stochastic programming.            |
| edu/class/stats311/Lectures/full notes.                                                 |                                                                |
| pdf.                                                                                    |                                                                |
| Last                                                                                    |                                                                |
|                                                                                         | Mathematical Programming, 156(1):59–99, 2016.                  |
| visited on, 2:23, 2016.                                                                 |                                                                |
|                                                                                         | Jain, P. and Thakurta, A.                                      |
|                                                                                         | (near) dimension independent risk                              |
| Dwork, C. and Roth, A. The algorithmic foundations of                                   |                                                                |
|                                                                                         | bounds for differentially private learning.                    |
|                                                                                         | In ICML, 2014.                                                 |
| differential privacy. Foundations and Trends® in Theo-                                  |                                                                |
| retical Computer Science, 9(3–4):211–407, 2014.                                         | Jain, P., Kothari, P., and Thakurta, A. Differentially private |
|                                                                                         | online learning.                                               |
|                                                                                         | In 25th Annual Conference on Learning                          |
| Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrat-                            |                                                                |
|                                                                                         | Theory (COLT), pp. 24.1–24.34, 2012.                           |
| ing noise to sensitivity in private data analysis.                                      |                                                                |
| In Theory                                                                               |                                                                |
| of cryptography conference, pp. 265–284. Springer, 2006.                                |                                                                |
|                                                                                         | Jin, C., Netrapalli, P., Ge, R., Kakade, S. M., and Jordan,    |
|                                                                                         | M.                                                             |
|                                                                                         | I.                                                             |
|                                                                                         | A short note on concentration inequalities                     |
|                                                                                         | for                                                            |
| Fang, C., Li, C.                                                                        |                                                                |
| J., Lin, Z.,                                                                            |                                                                |
| and Zhang, T.                                                                           |                                                                |
| Spider:                                                                                 |                                                                |
|                                                                                         | random vectors with subgaussian norm. arXiv preprint           |
| Near-optimal non-convex optimization via                                                |                                                                |
| stochastic                                                                              |                                                                |
|                                                                                         | arXiv:1902.03736, 2019.                                        |
| path-integrated differential estimator.                                                 |                                                                |
| In Bengio, S.,                                                                          |                                                                |
| Wallach, H.,                                                                            |                                                                |
| Larochelle, H., Grauman, K.,                                                            |                                                                |
| Cesa-                                                                                   |                                                                |
|                                                                                         | Kamath, G. and Ullman, J. A primer on private statistics.      |
| Bianchi, N., and Garnett, R. (eds.), Advances in Neural                                 |                                                                |
|                                                                                         | arXiv preprint arXiv:2005.00010, 2020.                         |
| Information Processing Systems, volume 31. Curran As-                                   |                                                                |
| sociates, Inc., 2018. URL https://proceedings.                                          | Kifer, D., Smith, A., and Thakurta, A. Private convex empir-   |
| neurips.cc/paper/2018/file/                                                             | ical risk minimization and high-dimensional regression.        |
| 1543843a4723ed2ab08e18053ae6dc5b-Paper.                                                 | In Conference on Learning Theory, pp. 25–1, 2012.              |
| pdf.                                                                                    |                                                                |
|                                                                                         | Kulkarni, J., Lee, Y. T., and Liu, D. Private non-smooth       |
| Feldman, V., Koren, T., and Talwar, K. Private stochastic                               |                                                                |
|                                                                                         | erm and sco in subquadratic steps.                             |
|                                                                                         | In Ranzato, M.,                                                |
| convex optimization: optimal rates in linear time.                                      |                                                                |
| In Pro-                                                                                 |                                                                |
|                                                                                         | Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan,          |
| ceedings of the 52nd Annual ACM SIGACT Symposium                                        |                                                                |
|                                                                                         | J. W. (eds.), Advances in Neural Information Processing        |
| on Theory of Computing, pp. 439–449, 2020.                                              |                                                                |
|                                                                                         | Systems,                                                       |
|                                                                                         | volume                                                         |
|                                                                                         | 34,                                                            |
|                                                                                         | pp.                                                            |
|                                                                                         | 4053–4064. Curran Asso-                                        |
|                                                                                         | ciates,                                                        |
|                                                                                         | Inc., 2021.                                                    |
|                                                                                         | URL https://proceedings.                                       |
| Foster, D.                                                                              |                                                                |
| J., Sekhari, A.,                                                                        |                                                                |
| and Sridharan, K.                                                                       |                                                                |
| Uniform                                                                                 |                                                                |
|                                                                                         | neurips.cc/paper/2021/file/                                    |
| convergence of gradients for non-convex learning and                                    |                                                                |
|                                                                                         | 211c1e0b83b9c69fa9c4bdede203c1e3-Paper.                        |
| optimization.                                                                           |                                                                |
| In Bengio, S., Wallach, H., Larochelle,                                                 |                                                                |
|                                                                                         | pdf.                                                           |
| H., Grauman, K.,                                                                        |                                                                |
| Cesa-Bianchi, N.,                                                                       |                                                                |
| and Garnett,                                                                            |                                                                |
| Advances                                                                                |                                                                |
| in                                                                                      |                                                                |
| Neural                                                                                  |                                                                |
| Information                                                                             |                                                                |
| Pro-                                                                                    |                                                                |
| R.                                                                                      |                                                                |
| (eds.),                                                                                 |                                                                |
|                                                                                         | Lan, G. First-order and stochastic optimization methods for    |
| cessing                                                                                 |                                                                |
| Systems,                                                                                |                                                                |
| volume                                                                                  |                                                                |
| 31.                                                                                     |                                                                |
| Curran                                                                                  |                                                                |
| Associates,                                                                             |                                                                |
|                                                                                         | machine learning. Springer, 2020.                              |
| https://proceedings.                                                                    |                                                                |
| Inc.,                                                                                   |                                                                |
| 2018.                                                                                   |                                                                |
| URL                                                                                     |                                                                |
| neurips.cc/paper/2018/file/                                                             | Ma, C., Wang, K., Chi, Y., and Chen, Y.                        |
|                                                                                         | Implicit                                                       |
|                                                                                         | regu-                                                          |
| 59ab3ba90ae4b4ab84fe69de7b8e3f5f-Paper.                                                 | larization in nonconvex statistical estimation: Gradient       |
| pdf.                                                                                    | descent converges linearly for phase retrieval and matrix      |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

completion.   In   Dy,   J.   and   Krause,   A.   (eds.),   Proceed- ings   of   the   35th   International   Conference   on   Machine Learning , volume 80 of  Proceedings of Machine Learn- ing   Research ,   pp.   3345–3354.   PMLR,   10–15   Jul   2018. URL  https://proceedings.mlr.press/v80/ ma18c.html .

Nemirovsky, A. S. and Yudin, D. B.   Problem complexity and method efficiency in optimization . Wiley-Interscience, 1983.

Nesterov,   Y.   How   to   make   the   gradients   small.   Optima. Mathematical Optimization Society Newsletter , (88):10– 11, 2012.

Nesterov, Y. and Polyak, B.   Cubic regularization of new- ton   method   and   its   global   performance.   Mathematical Programming , 108:177–205, 2006.

Rudelson, M. and Vershynin, R.   Non-asymptotic theory of random matrices:   extreme singular values.   In  Proceed- ings   of   the   International   Congress   of   Mathematicians 2010   (ICM   2010)   (In   4   Volumes)   Vol.   I:   Plenary   Lec- tures and Ceremonies Vols. II–IV: Invited Lectures ,   pp. 1576–1602. World Scientific, 2010.

Song, S., Steinke, T., Thakkar, O., and Thakurta, A. Evading the curse of dimensionality in unconstrained private glms. In  International Conference on Artificial Intelligence and Statistics , pp. 2638–2646. PMLR, 2021.

Steinke, T. and Ullman, J.   Between pure and approximate differential privacy.   Journal of Privacy and Confidential- ity , 7, 01 2015.   doi:   10.29012/jpc.v7i2.648.

Sun,   J.,   Qu,   Q.,   and   Wright,   J.   A   geometric   analysis   of phase retrieval.   In  2016 IEEE International Symposium on Information Theory (ISIT) , pp. 2379–2383, 2016.   doi: 10.1109/ISIT.2016.7541725.

Talwar, K., Thakurta, A., and Zhang, L.   Private empirical risk minimization beyond the worst case: The effect of the constraint set geometry.   arXiv preprint arXiv:1411.5417 , 2014.

Talwar,   K.,   Thakurta,   A.,   and   Zhang,   L.   Nearly   optimal private lasso.   In  NIPS , 2015.

Tran,   H.   and   Cutkosky,   A.   Momentum   aggregation   for private   non-convex   erm. In   Advances   in   Neural   In- formation   Processing   Systems ,   volume   35.   Curran   As- sociates,   Inc.,   2022.   URL   https://openreview. net/pdf?id=x56v-UN7BjD .

Wang,   D. and Xu,   J.   Differentially private empirical risk minimization with smooth non-convex loss functions:   A non-stationary view.   In  Proceedings of the AAAI Confer- ence on Artificial Intelligence , volume 33, pp. 1182–1189, 2019.

Wang, D., Ye, M., and Xu, J.   Differentially private empiri- cal risk minimization revisited:   Faster and more general. Advances in Neural Information Processing Systems , 30, 2017.

Wang, D., Chen, C., and Xu, J.   Differentially private em- pirical risk minimization with non-convex loss functions. In  Proceedings of the 36th International Conference on Machine   Learning ,   volume   97   of   Proceedings   of   Ma- chine   Learning   Research ,   pp.   6526–6535.   PMLR,   09– 15 Jun 2019a.   URL  https://proceedings.mlr. press/v97/wang19c.html .

Wang,   L.,   Jayaraman,   B.,   Evans,   D.,   and   Gu,   Q. Effi- cient privacy-preserving nonconvex optimization.   CoRR , abs/1910.13659, 2019b.   URL  http://arxiv.org/ abs/1910.13659 .

Wang,   Z.,   Ji,   K.,   Zhou,   Y.,   Liang,   Y.,   and   Tarokh, V. Spiderboost   and   momentum: Faster   variance reduction   algorithms. In   Wallach,   H.,   Larochelle, H.,   Beygelzimer,   A.,   d'Alch ´ e-Buc,   F.,   Fox,   E.,   and Garnett,   R.   (eds.),   Advances   in   Neural   Information Processing   Systems ,   volume   32.   Curran   Associates, Inc., 2019c. URL https://proceedings. neurips.cc/paper/2019/file/ 512c5cad6c37edb98ae91c8a76c3a291-Paper. pdf .

Zhang,   J.,   Zheng,   K.,   Mou,   W.,   and   Wang,   L.   Efficient private erm for smooth objectives.   In  Proceedings of the 26th   International   Joint   Conference   on   Artificial   Intel- ligence ,   IJCAI’17,   pp.   3922–3928.   AAAI   Press,   2017. ISBN 9780999241103.

Zhang, Q., Ma, J., Lou, J., and Xiong, L.   Private stochastic non-convex optimization with improved utility rates.   In Proceedings of the Thirtieth International Joint Confer- ence on Artificial Intelligence, IJCAI-21 , pp. 3370–3376, 2021.

Zhou, Y., Chen, X., Hong, M., Wu, Z. S., and Banerjee, A. Private   stochastic   non-convex   optimization:   Adaptive algorithms   and   tighter   generalization   bounds. CoRR , abs/2006.13501, 2020.   URL  https://arxiv.org/ abs/2006.13501 .

11



| 0                                                                                       | 1                                                            |
|:----------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                              |
| completion.                                                                             | Wang, D., Ye, M., and Xu, J. Differentially private empiri-  |
| In Dy, J. and Krause, A. (eds.), Proceed-                                               |                                                              |
| ings of                                                                                 | cal risk minimization revisited: Faster and more general.    |
| the 35th International Conference on Machine                                            |                                                              |
| Learning, volume 80 of Proceedings of Machine Learn-                                    | Advances in Neural Information Processing Systems, 30,       |
| ing Research, pp. 3345–3354. PMLR, 10–15 Jul 2018.                                      | 2017.                                                        |
| URL https://proceedings.mlr.press/v80/                                                  |                                                              |
|                                                                                         | Wang, D., Chen, C., and Xu, J. Differentially private em-    |
| ma18c.html.                                                                             |                                                              |
|                                                                                         | pirical risk minimization with non-convex loss functions.    |
| Nemirovsky, A. S. and Yudin, D. B. Problem complexity                                   | In Proceedings of the 36th International Conference on       |
| and method efficiency in optimization. Wiley-Interscience,                              | Machine Learning, volume 97 of Proceedings of Ma-            |
| 1983.                                                                                   | chine Learning Research, pp. 6526–6535. PMLR, 09–            |
|                                                                                         | 15 Jun 2019a. URL https://proceedings.mlr.                   |
| Nesterov, Y. How to make the gradients small. Optima.                                   |                                                              |
|                                                                                         | press/v97/wang19c.html.                                      |
| Mathematical Optimization Society Newsletter, (88):10–                                  |                                                              |
| 11, 2012.                                                                               |                                                              |
|                                                                                         | Wang, L.,                                                    |
|                                                                                         | Jayaraman, B., Evans, D., and Gu, Q.                         |
|                                                                                         | Effi-                                                        |
|                                                                                         | cient privacy-preserving nonconvex optimization. CoRR,       |
| Nesterov, Y. and Polyak, B. Cubic regularization of new-                                |                                                              |
|                                                                                         | abs/1910.13659, 2019b. URL http://arxiv.org/                 |
| ton method and its global performance. Mathematical                                     |                                                              |
|                                                                                         | abs/1910.13659.                                              |
| Programming, 108:177–205, 2006.                                                         |                                                              |
|                                                                                         | Wang,                                                        |
|                                                                                         | Z.,                                                          |
|                                                                                         | Ji, K.,                                                      |
|                                                                                         | Zhou, Y.,                                                    |
|                                                                                         | Liang, Y.,                                                   |
|                                                                                         | and Tarokh,                                                  |
| Rudelson, M. and Vershynin, R. Non-asymptotic theory of                                 |                                                              |
|                                                                                         | V                                                            |
|                                                                                         | .                                                            |
|                                                                                         | Spiderboost                                                  |
|                                                                                         | and momentum:                                                |
|                                                                                         | Faster                                                       |
|                                                                                         | variance                                                     |
| random matrices: extreme singular values.                                               |                                                              |
| In Proceed-                                                                             |                                                              |
|                                                                                         | reduction                                                    |
|                                                                                         | algorithms.                                                  |
|                                                                                         | In Wallach, H.,                                              |
|                                                                                         | Larochelle,                                                  |
| ings of                                                                                 |                                                              |
| the International Congress of Mathematicians                                            |                                                              |
|                                                                                         | H., Beygelzimer, A.,                                         |
|                                                                                         | d'Alch´e-Buc, F., Fox, E.,                                   |
|                                                                                         | and                                                          |
| 2010 (ICM 2010)                                                                         |                                                              |
| (In 4 Volumes) Vol.                                                                     |                                                              |
| I: Plenary Lec-                                                                         |                                                              |
|                                                                                         | in Neural                                                    |
|                                                                                         | Information                                                  |
|                                                                                         | Garnett, R.                                                  |
|                                                                                         | (eds.), Advances                                             |
| tures and Ceremonies Vols. II–IV: Invited Lectures, pp.                                 |                                                              |
|                                                                                         | Processing                                                   |
|                                                                                         | Systems,                                                     |
|                                                                                         | volume                                                       |
|                                                                                         | 32. Curran Associates,                                       |
| 1576–1602. World Scientific, 2010.                                                      |                                                              |
|                                                                                         | https://proceedings.                                         |
|                                                                                         | Inc.,                                                        |
|                                                                                         | 2019c.                                                       |
|                                                                                         | URL                                                          |
| Song, S., Steinke, T., Thakkar, O., and Thakurta, A. Evading                            | neurips.cc/paper/2019/file/                                  |
| the curse of dimensionality in unconstrained private glms.                              | 512c5cad6c37edb98ae91c8a76c3a291-Paper.                      |
| In International Conference on Artificial Intelligence and                              | pdf.                                                         |
| Statistics, pp. 2638–2646. PMLR, 2021.                                                  |                                                              |
|                                                                                         | Zhang, J., Zheng, K., Mou, W., and Wang, L.                  |
|                                                                                         | Efficient                                                    |
| Steinke, T. and Ullman, J. Between pure and approximate                                 |                                                              |
|                                                                                         | private erm for smooth objectives.                           |
|                                                                                         | In Proceedings of the                                        |
| differential privacy. Journal of Privacy and Confidential-                              |                                                              |
|                                                                                         | 26th International Joint Conference on Artificial Intel-     |
| ity, 7, 01 2015. doi: 10.29012/jpc.v7i2.648.                                            |                                                              |
|                                                                                         | ligence,                                                     |
|                                                                                         | IJCAI’17, pp. 3922–3928. AAAI Press, 2017.                   |
|                                                                                         | ISBN 9780999241103.                                          |
| Sun, J., Qu, Q., and Wright, J. A geometric analysis of                                 |                                                              |
| phase retrieval.                                                                        |                                                              |
| In 2016 IEEE International Symposium                                                    |                                                              |
|                                                                                         | Zhang, Q., Ma, J., Lou, J., and Xiong, L. Private stochastic |
| on Information Theory (ISIT), pp. 2379–2383, 2016. doi:                                 |                                                              |
|                                                                                         | non-convex optimization with improved utility rates.         |
|                                                                                         | In                                                           |
| 10.1109/ISIT.2016.7541725.                                                              |                                                              |
|                                                                                         | Proceedings of the Thirtieth International Joint Confer-     |
|                                                                                         | ence on Artificial Intelligence, IJCAI-21, pp. 3370–3376,    |
| Talwar, K., Thakurta, A., and Zhang, L. Private empirical                               |                                                              |
|                                                                                         | 2021.                                                        |
| risk minimization beyond the worst case: The effect of the                              |                                                              |
| constraint set geometry. arXiv preprint arXiv:1411.5417,                                |                                                              |
|                                                                                         | Zhou, Y., Chen, X., Hong, M., Wu, Z. S., and Banerjee, A.    |
| 2014.                                                                                   |                                                              |
|                                                                                         | Private stochastic non-convex optimization: Adaptive         |
|                                                                                         | algorithms and tighter generalization bounds.                |
|                                                                                         | CoRR,                                                        |
| Talwar, K., Thakurta, A., and Zhang, L. Nearly optimal                                  |                                                              |
|                                                                                         | abs/2006.13501, 2020. URL https://arxiv.org/                 |
| private lasso.                                                                          |                                                              |
| In NIPS, 2015.                                                                          |                                                              |
|                                                                                         | abs/2006.13501.                                              |
| Tran, H. and Cutkosky, A. Momentum aggregation for                                      |                                                              |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

A. Lower bounds

A.1. Missing details from DP Empirical Stationarity Lower Bound

Proof of Theorem  4.3 .   For any  r   >  0 , let  W r   denote the ball of radius  r  centered at the origin.   Let  B   =   L 0

L 1   .   Consider the loss function:

f ( w ;  x ) =

� L 1 2   ∥ w  − x ∥ 2 if  ∥ w  − x ∥≤ B

L 0  ∥ w  − x ∥− L 2 0 2 L 1 otherwise

The function  f ( w ;  x )  is convex,  L 1 -smooth and  L 0 -Lispchitz in  R d .   We restrict to datasets  S   =  { x i } n i =1   where  x i   ∈W B/ 4 for   all   i ,   and   let   F ( w ;  S )   = 1 n � n i =1   f ( w ;  x i )   be   the   empirical   risk   on   S .   The   unconstrained   minimizer   of   F ( w ;  S )   is w ∗ =   1

n � n i =1   x i   which lies in  W B/ 4 .

For any  w   ∈W 3 B/ 4 ,  w  lies in the quadratic region around all data points.   Hence, from  L 1 -strong convexity of  w   �→ F ( w ;  S ) on  W 3 B/ 4 , we have that whenever   ¯ w   ∈W 3 B/ 4 ,

∥∇ F ( ¯ w ;  S ) ∥∥ ¯ w  − w ∗ ∥≥⟨∇ F ( ¯ w ;  S ) , w ∗ − ¯ w ⟩≥ F ( ¯ w ;  S )  − F ( w ∗ ;  S )  ≥ L 1

2   ∥ ¯ w  − w ∗ ∥ 2  .

Let  E   be the event that   ¯ w   ∈W 3 B/ 4   and let  E E   denote the conditional expectation (conditioned on event  E ) operator.   Then,

E E  ∥∇ F ( ¯ w ;  S ) ∥≥ L 1

2   E  ∥ ¯ w  − w ∗ ∥≥ L 1

2   Ω

�� L 0 4 L 1

� min

�

1 ,

� d  log (1 /δ )

nε

��

.

where the last inequality follows from known lower bounds for DP mean estimation ( Steinke & Ullman ,  2015 ;  Kamath & Ullman ,  2020 ).   We remark that the lower bound in the referenced work is for algorithms which produce outputs in the ball of the same radius as the dataset, i.e.   W B/ 4 .   However, a simple post-processing argument shows that the same lower bound applies to algorithms which produce output in  W 3 B/ 4 .   Specifically, assuming the contrary, we simply project the output in  W 3 B/ 4   to  W B/ 4 :   privacy is preserved by post-processing and the distance to the mean cannot increase by the non-expansiveness property of projection to convex sets, hence a contradiction.   This gives us,

E E  [ ∥∇ F ( ¯ w ;  S ) ∥ ]  ≥ Ω

�

L 0  min

�

1 ,

� d  log (1 /δ )

nε

��

Let   ˜ W   =   { w   :  ∥ w  − w ∗ ∥≤ B/ 2 } .   Since   ˜ W   ⊆W 3 B/ 4 ,   we   have   that   the   above   conditional   lower   bound   applies   for ¯ w   ∈ ˜ W   as well.   We now consider   ¯ w   ̸∈ ˜ W .   Let  w ′   be  any  point on the boundary of   ˜ W ,   denoted as  ∂ W .   Note that  w ′

lies in the region where, for any data point, the corresponding loss is a quadratic function.   Hence, by direct computation, ∇ F ( w ′ ;  S ) =  L 1  ( w ′   − w ∗ ) .   Therefore,

⟨∇ F ( w ′ ) , w ′   − w ∗ ⟩ =  L 1  ∥ w ′   − w ∗ ∥ 2   =   L 1 B 2

4 .

We now apply Lemma  A.1  which gives us,

E E c  ∥∇ F ( ¯ w ;  S ) ∥≥ L 1 B 2

4 ·   2 B   =   L 0

2   ,

where  E c   denotes the complement set of  E .   We combine the above bounds using the law of total expectation as follows,

E [ ∥∇ F ( ¯ w ;  S ) ∥ ] = E E [ ∥∇ F ( ¯ w ;  S ) ∥ ] P {  ¯ w   ∈ E }  +  E E c [ ∥∇ F ( ¯ w ;  S ) ∥ ] P {  ¯ w   ∈ E c }

= Ω � L 0  min � 1 ,

� d  log (1 /δ )

nε

�� P ( ¯ w   ∈ E ) + Ω( L 0 ) P ( ¯ w   ∈ E c )

= Ω � L 0  min � 1 ,

� d  log (1 /δ )

nε

�� .

This completes the proof.

12



| 0                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                           |
| A. Lower bounds                                                                                                                   |
| A.1. Missing details from DP Empirical Stationarity Lower Bound                                                                   |
| . Consider the                                                                                                                    |
| Proof of Theorem 4.3. For any r > 0, let Wr denote the ball of radius r centered at the origin. Let B = L0                        |
| L1                                                                                                                                |
| loss function:                                                                                                                    |
| (cid:40) L1                                                                                                                       |
| ∥w − x∥2                                                                                                                          |
| if ∥w − x∥ ≤ B                                                                                                                    |
| 2                                                                                                                                 |
| f (w; x) =                                                                                                                        |
| otherwise                                                                                                                         |
| L0 ∥w − x∥ − L2                                                                                                                   |
| 2L1                                                                                                                               |
| The function f (w; x) is convex, L1-smooth and L0-Lispchitz in Rd. We restrict to datasets S = {xi}n                              |
| i=1 where xi ∈ WB/4                                                                                                               |
| (cid:80)n                                                                                                                         |
| for all i, and let F (w; S) = 1                                                                                                   |
| i=1 f (w; xi) be the empirical risk on S. The unconstrained minimizer of F (w; S) is                                              |
| n                                                                                                                                 |
| (cid:80)n                                                                                                                         |
| w∗ = 1                                                                                                                            |
| i=1 xi which lies in WB/4.                                                                                                        |
| n                                                                                                                                 |
| For any w ∈ W3B/4, w lies in the quadratic region around all data points. Hence, from L1-strong convexity of w (cid:55)→ F (w; S) |
| on W3B/4, we have that whenever ¯w ∈ W3B/4,                                                                                       |
| L1                                                                                                                                |
| ∥∇F ( ¯w; S)∥ ∥ ¯w − w∗∥ ≥ ⟨∇F ( ¯w; S), w∗ − ¯w⟩ ≥ F ( ¯w; S) − F (w∗; S) ≥                                                      |
| ∥ ¯w − w∗∥2 .                                                                                                                     |
| 2                                                                                                                                 |
| Let E be the event that ¯w ∈ W3B/4 and let EE denote the conditional expectation (conditioned on event E) operator. Then,         |
| (cid:33)(cid:33)                                                                                                                  |
| (cid:32)                                                                                                                          |
| (cid:19)                                                                                                                          |
| (cid:112)d log (1/δ)                                                                                                              |
| L1                                                                                                                                |
| L1                                                                                                                                |
| (cid:32)(cid:18) L0                                                                                                               |
| E ∥ ¯w − w∗∥ ≥                                                                                                                    |
| Ω                                                                                                                                 |
| .                                                                                                                                 |
| min                                                                                                                               |
| 1,                                                                                                                                |
| EE ∥∇F ( ¯w; S)∥ ≥                                                                                                                |
| 2                                                                                                                                 |
| 2                                                                                                                                 |
| nε                                                                                                                                |
| 4L1                                                                                                                               |
| where the last inequality follows from known lower bounds for DP mean estimation (Steinke & Ullman, 2015; Kamath                  |
| & Ullman, 2020). We remark that the lower bound in the referenced work is for algorithms which produce outputs in the             |
| ball of the same radius as the dataset, i.e. WB/4. However, a simple post-processing argument shows that the same lower           |
| bound applies to algorithms which produce output in W3B/4. Specifically, assuming the contrary, we simply project the             |
| output in W3B/4 to WB/4: privacy is preserved by post-processing and the distance to the mean cannot increase by the              |
| non-expansiveness property of projection to convex sets, hence a contradiction. This gives us,                                    |
| (cid:33)(cid:33)                                                                                                                  |
| (cid:32)                                                                                                                          |
| (cid:32)                                                                                                                          |
| (cid:112)d log (1/δ)                                                                                                              |
| 1,                                                                                                                                |
| EE [∥∇F ( ¯w; S)∥] ≥ Ω                                                                                                            |
| L0 min                                                                                                                            |
| nε                                                                                                                                |
| ˜                                                                                                                                 |
| Let                                                                                                                               |
| W = {w : ∥w − w∗∥ ≤ B/2}.                                                                                                         |
| Since                                                                                                                             |
| the above conditional                                                                                                             |
| lower bound applies for                                                                                                           |
| W ⊆ W3B/4, we have that                                                                                                           |
| ˜                                                                                                                                 |
| w ∈                                                                                                                               |
| w ̸∈                                                                                                                              |
| W as well. We now consider                                                                                                        |
| W. Let w′ be any point on the boundary of                                                                                         |
| W, denoted as ∂W. Note that w′                                                                                                    |
| lies in the region where, for any data point, the corresponding loss is a quadratic function. Hence, by direct computation,       |
| ∇F (w′; S) = L1 (w′ − w∗). Therefore,                                                                                             |
| L1B2                                                                                                                              |
| .                                                                                                                                 |
| ⟨∇F (w′), w′ − w∗⟩ = L1 ∥w′ − w∗∥2 =                                                                                              |
| 4                                                                                                                                 |
| We now apply Lemma A.1 which gives us,                                                                                            |
| L1B2                                                                                                                              |
| L0                                                                                                                                |
| 2 B                                                                                                                               |
| ·                                                                                                                                 |
| =                                                                                                                                 |
| ,                                                                                                                                 |
| EEc ∥∇F ( ¯w; S)∥ ≥                                                                                                               |
| 4                                                                                                                                 |
| 2                                                                                                                                 |
| where Ec denotes the complement set of E. We combine the above bounds using the law of total expectation as follows,              |
| E[∥∇F ( ¯w; S)∥]                                                                                                                  |
| =                                                                                                                                 |
| EE[∥∇F ( ¯w; S)∥]P{ ¯w ∈ E} + EEc [∥∇F ( ¯w; S)∥]P{ ¯w ∈ Ec}                                                                      |
| (cid:16)                                                                                                                          |
| (cid:110)                                                                                                                         |
| (cid:111)(cid:17)                                                                                                                 |
| (cid:112)d log (1/δ)                                                                                                              |
| =                                                                                                                                 |
| Ω                                                                                                                                 |
| 1,                                                                                                                                |
| L0 min                                                                                                                            |
| P( ¯w ∈ E) + Ω(L0)P( ¯w ∈ Ec)                                                                                                     |
| nε                                                                                                                                |
| (cid:16)                                                                                                                          |
| (cid:110)                                                                                                                         |
| (cid:111)(cid:17)                                                                                                                 |
| (cid:112)d log (1/δ)                                                                                                              |
| =                                                                                                                                 |
| Ω                                                                                                                                 |
| 1,                                                                                                                                |
| .                                                                                                                                 |
| L0 min                                                                                                                            |
| nε                                                                                                                                |
| This completes the proof.                                                                                                         |
| 12                                                                                                                                |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Lemma A.1.   Let  G, R  ≥ 0 , d  ∈ N .   Let  W R ( w 0 )  denote the Euclidean ball around  w 0  of radius  R  and let  ∂ W R ( w 0 )  denote its boundary.   Let  f   :  R d   → R  be a differentiable convex function.   Suppose  w 0   ∈ R d   is such that for every  v   ∈ ∂ W R ( w 0 ) , ⟨∇ f ( v ) , v  − w 0 ⟩≥ G , then for any  w   ̸∈W R ( w 0 ) , we have  ∥∇ f ( w ) ∥≥ G

R .

Proof.   For a unit vector  u   ∈ R d , define directional directive  f   ′ u ( w )   =   ⟨∇ f ( w ) , u ⟩ .   We first show that for any  u   ∈ R d   : ∥ u ∥ = 1  and any  w ′   ∈ R d , the function  f   ′ u ( w ′  +  ru )  is non-decreasing in  r   ∈ R + .   This simply follows from monotonicity of gradients since  f   is convex.   In particular, for any  r ′   > r   >  0 , we have

f   ′ u ( w ′  +  r ′ u )  − f  ′ u ( w ′  +  ru ) =  ⟨∇ f ( w ′  +  r ′ u )  −∇ f ( w ′  +  ru ) , u ⟩

= 1 r ′   − r   ⟨∇ f ( w ′  +  r ′ u )  −∇ f ( w ′  +  ru ) , w ′  +  ru  − ( w ′  +  ru ) ⟩

>  0

We now prove the claim in the lemma statement.   Let  w   ̸∈ ∂W R   and define  u   = w − w 0 ∥ w − w 0 ∥ .   Then from Cauchy-Schwarz inequality and the above monotonicity property, we have,

∥∇ f ( w ) ∥≥⟨∇ f ( w ) , u ⟩ =  f   ′ u ( w )  ≥ f  ′ u ( w 0   +  Ru ) =  ⟨∇ f ( w 0   +  Ru ) , u ⟩

=   1

R   ⟨∇ f ( v ) , v  − w 0 ⟩≥ G

R

which finishes the proof.

A.2. Non-private Sample Complexity Lower Bound

Theorem A.2.   For any  L 0 , L 1 ,  n, d   ∈ N , there exists a distribution  D   over some set  X   and a  L 0 -Lipschitz,  L 1 -smooth (convex) loss function  w   �→ f ( w ;  x )  such that given  n  i.i.d samples from  D , the output   ¯ w  of any algorithm satisfies,

E  ∥∇ F ( ¯ w ;  D ) ∥ = Ω � L 0 √ n

�

Proof.   We construct a hard instance in  d  = 1  dimension.   Let  p  ∈ [0 ,  1]  be a parameter to be set later and let  v   ∈{− 1 ,  1 }  be chosen by an adversary.   Let the data domain  X   =  {− 1 ,  1 }  and consider the distribution  D  on  X   as follows:

x  =

� 1 with probability 1+ vp

2 − 1 with probability 1 − vp

2

Note that  E [ x ] =  vp .   Consider the loss function  f ( w ;  x )  as

f ( w ;  x ) =   L 0

2   wx  +   L 1

2   ∆( w )

where  ∆ is the Huber regularization function, defined as,

∆( w ) =

� | w | 2 if   | w | ≤ L 0 2 L 1 L 0 | w |

L 1 − L 2 0 4 L 2 1 otherwise

Note   that the   loss function   w   �→ f ( w ;  x )   is   convex,   L 0 -Lipschitz and   L 1 -smooth   in   R d ,   for all   x .   The   population risk function is,

F ( w ;  D ) =   L 0

2   wpv  +   L 1

2   ∆( w )

Let   ¯ w  be output some algorithm given  n  i.i.d.   samples from  D .   Consider two cases:

13



| 0                                                                                                                     |
|:----------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                               |
| Lemma A.1. Let G, R ≥ 0, d ∈ N. Let WR(w0) denote the Euclidean ball around w0 of radius R and let ∂WR(w0) denote     |
| its boundary. Let f : Rd → R be a differentiable convex function. Suppose w0 ∈ Rd is such that for every v ∈ ∂WR(w0), |
| ⟨∇f (v), v − w0⟩ ≥ G, then for any w ̸∈ WR(w0), we have ∥∇f (w)∥ ≥ G                                                  |
| R .                                                                                                                   |
| Proof. For a unit vector u ∈ Rd, define directional directive f ′                                                     |
| u(w) = ⟨∇f (w), u⟩. We first show that for any u ∈ Rd :                                                               |
| ∥u∥ = 1 and any w′ ∈ Rd, the function f ′                                                                             |
| u(w′ + ru) is non-decreasing in r ∈ R+. This simply follows from monotonicity                                         |
| of gradients since f is convex. In particular, for any r′ > r > 0, we have                                            |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Case 1:   |  ¯ w |  > L 0 2 L 1   : The gradient norm in this case is

|∇ F ( ¯ w ;  D ) | 2   = ���� L 0

2   vp  +   L 0  ¯ w

2  |  ¯ w |

����

2

=   L 2 0 p 2

4 +   L 2 0 4   +   L 2 0 2  |  ¯ w | vp  ¯ w

≥ L 2 0 4   − L 2 0 2   p

=   L 2 0 4   − L 2 0 8 √ n

≥ L 2 0 8

where the first inequality follows since  v   ¯ w |  ¯ w |   ≥− 1 , the third equality follows by setting  p  = 1 √

16 n   and the second inequality

follows since  n  ≥ 1 .   We therefore have that  E  |∇ F ( ¯ w ;  D ) | ≥ L 0 2 √

2 .

Case 2:   |  ¯ w | ≤ L 0 2 L 1   : In this case, the gradient norm is,

|∇ F ( ¯ w ;  D ) | 2   = ���� L 0

2   vp  +  L 1  ¯ w ����

2

Suppose there exists an algorithm with output   ¯ w , which, with  n  samples guarantees that  E  |∇ F ( ¯ w ;  D ) |  < o � L 0 √ n � .   Then

from Markov’s inequality, with probability at least  0 . 9 , we have that  |∇ F ( ¯ w ;  D ) | 2   < o � L 2 0 n � .   Let   ˜ w   =  − 2 L 1   ¯ w L 0 , then we have that with probability at least  0 . 9 ,

|∇ F ( ¯ w ;  D ) | 2   ≤ o � L 2 0 n

� ⇐⇒| vp  − ˜ w | 2   < o � 1 n

�

This contradicts the well-known bias estimation lower bounds, with  p   = 1 √

16 n , using Le Cam’s method (( Duchi ,  2016 ),

Example 7.7), hence  E  |∇ F ( ¯ w ;  D ) | ≥ Ω � L 0 √ n � .   Combining the two cases finishes the proof.

B. Missing Results for Empirical Stationary Points

B.1. Private Spiderboost

The   following   lemma   largely   follows   from   the   analysis   in   ( Wang   et   al. ,   2019c ).   We   present   a   full   proof   below   for completeness.

Lemma B.1.   Let the conditions of Lemma  4.1  be satisfied.   Let  η   ≤ 1 2 L 1   and  q   ≤ O � 1 τ   2 2   η 2 � .   Then the output of Private SpiderBoost,   ¯ w  satisfies

E  [ ∥∇ F ( ¯ w ;  S ) ∥ ] =  O

�� F 0 ηT   +  τ 1

�

. (1)

Proof.   In the following, for any  t  ∈ [ T ] , let  s t   = � t q � q  (i.e.   the index corresponding to the start of the phase containing iteration  t ).

By   a   standard   analysis   for   smooth   functions   we   have   (recalling   that   ∇ t   is   an   unbiased   estimate   of   ∇ F ( w t ;  S )   for   any t  ∈ [ T ] )

F ( w t +1 ;  S )  ≤ F ( w t ;  S ) +   η

2   ∥∇ F ( w t ;  S )  −∇ t ∥ 2  − � η 2   − L 1 η 2

2

� ∥∇ t ∥ 2   .

14



| 0                                                                                                                |
|:-----------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                          |
| :                                                                                                                |
| Case 1: | ¯w| > L0                                                                                               |
| The gradient norm in this case is                                                                                |
| 2L1                                                                                                              |
| 2                                                                                                                |
| L0                                                                                                               |
| L0 ¯w                                                                                                            |
| (cid:12)(cid:12)(cid:12)(cid:12)                                                                                 |
| (cid:12)(cid:12)(cid:12)(cid:12)                                                                                 |
| |∇F ( ¯w; D)|2 =                                                                                                 |
| vp +                                                                                                             |
| 2                                                                                                                |
| 2 | ¯w|                                                                                                          |
| L2                                                                                                               |
| L2                                                                                                               |
| L2                                                                                                               |
| 0p2                                                                                                              |
| 0                                                                                                                |
| 0                                                                                                                |
| =                                                                                                                |
| +                                                                                                                |
| +                                                                                                                |
| vp ¯w                                                                                                            |
| 4                                                                                                                |
| 4                                                                                                                |
| 2 | ¯w|                                                                                                          |
| L2                                                                                                               |
| L2                                                                                                               |
| 0                                                                                                                |
| 0                                                                                                                |
| ≥                                                                                                                |
| −                                                                                                                |
| p                                                                                                                |
| 4                                                                                                                |
| 2                                                                                                                |
| L2                                                                                                               |
| L2                                                                                                               |
| 0                                                                                                                |
| 0                                                                                                                |
| √                                                                                                                |
| =                                                                                                                |
| −                                                                                                                |
| 4                                                                                                                |
| 8                                                                                                                |
| n                                                                                                                |
| L2                                                                                                               |
| 0                                                                                                                |
| ≥                                                                                                                |
| 8                                                                                                                |
| 1√                                                                                                               |
| and the second inequality                                                                                        |
| | ¯w| ≥ −1, the third equality follows by setting p =                                                            |
| 16n                                                                                                              |
| follows since n ≥ 1. We therefore have that E |∇F ( ¯w; D)| ≥ L0                                                 |
| .                                                                                                                |
| 2                                                                                                                |
| 2                                                                                                                |
| Case 2: | ¯w| ≤ L0                                                                                               |
| :                                                                                                                |
| In this case, the gradient norm is,                                                                              |
| 2L1                                                                                                              |
| 2                                                                                                                |
| L0                                                                                                               |
| (cid:12)(cid:12)(cid:12)(cid:12)                                                                                 |
| (cid:12)(cid:12)(cid:12)(cid:12)                                                                                 |
| |∇F ( ¯w; D)|2 =                                                                                                 |
| vp + L1 ¯w                                                                                                       |
| 2                                                                                                                |
| (cid:16) L0√                                                                                                     |
| Suppose there exists an algorithm with output ¯w, which, with n samples guarantees that E |∇F ( ¯w; D)| < o      |
| . Then                                                                                                           |
| n                                                                                                                |
| (cid:17)                                                                                                         |
| (cid:16) L2                                                                                                      |
| 0                                                                                                                |
| from Markov’s inequality, with probability at least 0.9, we have that |∇F ( ¯w; D)|2 < o                         |
| . Let ˜w = − 2L1 ¯w                                                                                              |
| , then we                                                                                                        |
| n                                                                                                                |
| L0                                                                                                               |
| have that with probability at least 0.9,                                                                         |
| (cid:19)                                                                                                         |
| (cid:19)                                                                                                         |
| (cid:18) L2                                                                                                      |
| (cid:18) 1                                                                                                       |
| 0                                                                                                                |
| |∇F ( ¯w; D)|2 ≤ o                                                                                               |
| ⇐⇒ |vp − ˜w|2 < o                                                                                                |
| n                                                                                                                |
| n                                                                                                                |
| 1√                                                                                                               |
| This contradicts the well-known bias estimation lower bounds, with p =                                           |
| , using Le Cam’s method ((Duchi, 2016),                                                                          |
| 16n                                                                                                              |
| (cid:16) L0√                                                                                                     |
| Example 7.7), hence E |∇F ( ¯w; D)| ≥ Ω                                                                          |
| . Combining the two cases finishes the proof.                                                                    |
| n                                                                                                                |
| B. Missing Results for Empirical Stationary Points                                                               |
| B.1. Private Spiderboost                                                                                         |
| The following lemma largely follows from the analysis in (Wang et al., 2019c). We present a full proof below for |
| completeness.                                                                                                    |
| (cid:16)                                                                                                         |
| (cid:17)                                                                                                         |
| 1                                                                                                                |
| 1                                                                                                                |
| . Then the output of Private                                                                                     |
| and q ≤ O                                                                                                        |
| Lemma B.1. Let the conditions of Lemma 4.1 be satisfied. Let η ≤                                                 |
| τ 2                                                                                                              |
| 2L1                                                                                                              |
| 2 η2                                                                                                             |
| SpiderBoost, ¯w satisfies                                                                                        |
| (cid:32)(cid:115)                                                                                                |
| (cid:33)                                                                                                         |
| F0                                                                                                               |
| E [∥∇F ( ¯w; S)∥] = O                                                                                            |
| .                                                                                                                |
| (1)                                                                                                              |
| + τ1                                                                                                             |
| ηT                                                                                                               |
| (cid:106) t                                                                                                      |
| Proof.                                                                                                           |
| q (i.e.                                                                                                          |
| the index corresponding to the start of the phase containing                                                     |
| In the following, for any t ∈ [T ], let st =                                                                     |
| q                                                                                                                |
| iteration t).                                                                                                    |
| By a standard analysis for smooth functions we have (recalling that ∇t                                           |
| is an unbiased estimate of ∇F (wt; S) for any                                                                    |
| t ∈ [T ])                                                                                                        |
| (cid:18) η                                                                                                       |
| L1η2                                                                                                             |
| η 2                                                                                                              |
| −                                                                                                                |
| F (wt+1; S) ≤ F (wt; S) +                                                                                        |
| ∥∇F (wt; S) − ∇t∥2 −                                                                                             |
| ∥∇t∥2 .                                                                                                          |
| 2                                                                                                                |
| 2                                                                                                                |
| 14                                                                                                               |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Taking expectation we have the following manipulation using the update rule of Algorithm  2

E  [ F ( w t +1 ;  S )  − F ( w t ;  S )]  ≤ η

2 E � ∥∇ F ( w t ;  S )  −∇ t ∥ 2 � − � η 2   − L 1 η 2

2

� E � ∥∇ t ∥ 2 �

≤ ητ  2 2 2

t �

k = s t +1 E � ∥ w k +1  − w k ∥ 2 � +   η

2 E � ∥∇ s t   − F ( w s t ;  S ) ∥ 2 �

− � η 2   − L 1 η 2

2

� E � ∥∇ t ∥ 2 �

≤ η 3 τ  2 2 2

t �

k = s t +1 E � ∥∇ k ∥ 2 � +   ητ  2 1 2 − � η 2   − L 1 η 2

2

� E � ∥∇ t ∥ 2 � ,

where the second inequality follows from Lemma  4.1  and the last inequality follows from the update rule.   Note that if t  =  s t  the sum is empty.   Summing over a given phase we have

E  [ F ( w t +1 ;  S )  − F ( w s t ;  S )]  ≤ η 3 τ  2 2 2

t �

k = s t

k �

j = s t +1 E � ∥∇ j ∥ 2 � +

t �

k = s t

� ητ   2 1 2   − � η 2   − L 1 η 2

2 � E � ∥∇ k ∥ 2 ��

≤ η 3 τ  2 2   q 2

t �

k = s t E � ∥∇ k ∥ 2 � +

t �

k = s t

� ητ   2 1 2   − � η 2   − L 1 η 2

2 � E � ∥∇ k ∥ 2 ��

=  −

t �

k = s t

�� η 2   − L 1 η 2

2 − η 3 τ  2 2   q 2

�

� �� � A

E � ∥∇ k ∥ 2 � − ητ  2 1 2

�

, (2)

where the second inequality comes from the fact that each gradient appears at most  q  times in the sum.   We now sum over all phases.   Let  P   =  { p 0 , p 1 , ..., }  = � 0 , q,  2 q, ..., � T  − 1

q � q, T � .   We have

E  [ F ( w T  ;  S )  − F ( w 0 ;  S )]  ≤

| P  | �

i =1 E � F ( w p i ;  S )  − F ( w p i − 1 ;  S ) �

≤−

T �

t =0 A  E � ∥∇ k ∥ 2 � +   Tητ  2 1 2 .

Rearranging the above yields

1 T

T �

t =0 E � ∥∇ k ∥ 2 � ≤ F 0

TA   +   ητ  2 1 2 A  . (3)

Now let  i ∗ denote the index of   ¯ w  selected by the algorithm.   Note that

E � ∥∇ F ( w i ∗ ;  S ) ∥ 2 � ≤ 2 E � ∥∇ F ( w i ∗ ;  S )  −∇ i ∗ ∥ 2 � + 2 E � ∥∇ i ∗ ∥ 2 � . (4)

15



| 0                                                                                                                          | 1                                                                                       | 2                                                  |
|:---------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:---------------------------------------------------|
|                                                                                                                            | Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |                                                    |
| Taking expectation we have the following manipulation using the update rule of Algorithm 2                                 |                                                                                         |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:18) η                                                                              |                                                    |
|                                                                                                                            | L1η2                                                                                    |                                                    |
| E [F (wt+1; S) − F (wt; S)] ≤                                                                                              | η 2                                                                                     |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | ∥∇F (wt; S) − ∇t∥2(cid:105)                                                             |                                                    |
|                                                                                                                            | ∥∇t∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | t(cid:88)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | ητ 2                                                                                    |                                                    |
|                                                                                                                            | η 2                                                                                     |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | ≤                                                                                       |                                                    |
|                                                                                                                            | ∥wk+1 − wk∥2(cid:105)                                                                   |                                                    |
|                                                                                                                            | ∥∇st − F (wst; S)∥2(cid:105)                                                            |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | k=st+1                                                                                  |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:18) η                                                                              |                                                    |
|                                                                                                                            | L1η2                                                                                    |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | ∥∇t∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | t(cid:88)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | η3τ 2                                                                                   |                                                    |
|                                                                                                                            | ητ 2                                                                                    |                                                    |
|                                                                                                                            | (cid:18) η                                                                              |                                                    |
|                                                                                                                            | L1η2                                                                                    |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | ≤                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | ,                                                                                       |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | ∥∇t∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | k=st+1                                                                                  |                                                    |
| where the second inequality follows from Lemma 4.1 and the last                                                            |                                                                                         | inequality follows from the update rule. Note that |
|                                                                                                                            |                                                                                         | if                                                 |
| t = st the sum is empty. Summing over a given phase we have                                                                |                                                                                         |                                                    |
|                                                                                                                            | k(cid:88)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:17)                                                                                |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | η3τ 2                                                                                   |                                                    |
|                                                                                                                            | (cid:104) ητ 2                                                                          |                                                    |
|                                                                                                                            | (cid:16) η                                                                              |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | ∥∇j∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)(cid:105)                                                                 |                                                    |
| E [F (wt+1; S) − F (wst; S)] ≤                                                                                             | t(cid:88) k                                                                             |                                                    |
|                                                                                                                            | t(cid:88) k                                                                             |                                                    |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2 − L1η2                                                                                |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | j=st+1                                                                                  |                                                    |
|                                                                                                                            | =st                                                                                     |                                                    |
|                                                                                                                            | =st                                                                                     |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:17)                                                                                |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | η3τ 2                                                                                   |                                                    |
|                                                                                                                            | (cid:104) ητ 2                                                                          |                                                    |
|                                                                                                                            | (cid:16) η                                                                              |                                                    |
|                                                                                                                            | 2 q                                                                                     |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)(cid:105)                                                                 |                                                    |
|                                                                                                                            | t(cid:88) k                                                                             |                                                    |
|                                                                                                                            | t(cid:88) k                                                                             |                                                    |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | ≤                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2 − L1η2                                                                                |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | =st                                                                                     |                                                    |
|                                                                                                                            | =st                                                                                     |                                                    |
|                                                                                                                            | (cid:35)                                                                                |                                                    |
|                                                                                                                            | (cid:19)                                                                                |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:34) (cid:18) η                                                                     |                                                    |
|                                                                                                                            | η3τ 2                                                                                   |                                                    |
|                                                                                                                            | ητ 2                                                                                    |                                                    |
|                                                                                                                            | L1η2                                                                                    |                                                    |
|                                                                                                                            | 2 q                                                                                     |                                                    |
|                                                                                                                            | t(cid:88) k                                                                             | (2)                                                |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | = −                                                                                     |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | ,                                                                                       |                                                    |
|                                                                                                                            | −                                                                                       |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | =st                                                                                     |                                                    |
|                                                                                                                            | (cid:124)                                                                               |                                                    |
|                                                                                                                            | (cid:123)(cid:122)                                                                      |                                                    |
|                                                                                                                            | (cid:125)                                                                               |                                                    |
|                                                                                                                            | A                                                                                       |                                                    |
| where the second inequality comes from the fact that each gradient appears at most q times in the sum. We now sum over all |                                                                                         |                                                    |
|                                                                                                                            | (cid:110)                                                                               |                                                    |
|                                                                                                                            | (cid:107)                                                                               |                                                    |
|                                                                                                                            | (cid:111)                                                                               |                                                    |
|                                                                                                                            | (cid:106) T −1                                                                          |                                                    |
| phases. Let P = {p0, p1, ...,} =                                                                                           | 0, q, 2q, ...,                                                                          |                                                    |
|                                                                                                                            | q, T                                                                                    |                                                    |
|                                                                                                                            | . We have                                                                               |                                                    |
|                                                                                                                            | q                                                                                       |                                                    |
|                                                                                                                            | |P |                                                                                    |                                                    |
|                                                                                                                            | (cid:88) i                                                                              |                                                    |
|                                                                                                                            | E [F (wT ; S) − F (w0; S)] ≤                                                            |                                                    |
|                                                                                                                            | E (cid:2)F (wpi ; S) − F (wpi−1; S)(cid:3)                                              |                                                    |
|                                                                                                                            | =1                                                                                      |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | T ητ 2                                                                                  |                                                    |
|                                                                                                                            | T(cid:88) t                                                                             |                                                    |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | ≤ −                                                                                     |                                                    |
|                                                                                                                            | A E                                                                                     |                                                    |
|                                                                                                                            | .                                                                                       |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | 2                                                                                       |                                                    |
|                                                                                                                            | =0                                                                                      |                                                    |
| Rearranging the above yields                                                                                               |                                                                                         |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
|                                                                                                                            | ητ 2                                                                                    |                                                    |
|                                                                                                                            | F0                                                                                      |                                                    |
|                                                                                                                            | 1 T                                                                                     | (3)                                                |
|                                                                                                                            | T(cid:88) t                                                                             |                                                    |
|                                                                                                                            | 1                                                                                       |                                                    |
|                                                                                                                            | E                                                                                       |                                                    |
|                                                                                                                            | ≤                                                                                       |                                                    |
|                                                                                                                            | +                                                                                       |                                                    |
|                                                                                                                            | .                                                                                       |                                                    |
|                                                                                                                            | ∥∇k∥2(cid:105)                                                                          |                                                    |
|                                                                                                                            | T A                                                                                     |                                                    |
|                                                                                                                            | 2A                                                                                      |                                                    |
|                                                                                                                            | =0                                                                                      |                                                    |
| Now let i∗ denote the index of ¯w selected by the algorithm. Note that                                                     |                                                                                         |                                                    |
| (cid:104)                                                                                                                  | (cid:104)                                                                               |                                                    |
|                                                                                                                            | (cid:104)                                                                               |                                                    |
| E                                                                                                                          | ≤ 2E                                                                                    | (4)                                                |
|                                                                                                                            | + 2E                                                                                    |                                                    |
|                                                                                                                            | .                                                                                       |                                                    |
|                                                                                                                            | ∥∇F (wi∗ ; S)∥2(cid:105)                                                                |                                                    |
|                                                                                                                            | ∥∇F (wi∗ ; S) − ∇i∗ ∥2(cid:105)                                                         |                                                    |
|                                                                                                                            | ∥∇i∗ ∥2(cid:105)                                                                        |                                                    |
|                                                                                                                            | 15                                                                                      |                                                    |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

The second term above can be bounded via inequality ( 3 ).   To bound the first term we have by Lemma  4.1  that

E � ∥∇ i ∗ −∇ F ( w i ∗ ;  S ) ∥ 2 � ≤ τ   2 2

t ∗ �

k = s t ∗ +1 E � ∥ w k  − w k − 1 ∥ 2 � +  τ   2 1

=  η 2 τ   2 2

t ∗ �

k = s t ∗ +1 E � ∥∇ k ∥ 2 � +  τ   2 1

≤ qη 2 τ  2 2 T

T �

k =0 E � ∥∇ k ∥ 2 � +  τ   2 1

≤ τ  2 2   η 2 qF 0

TA +   η 3 qτ  2 2 2 A   τ  2 1   +  τ  2 1   ,

where the last inequality comes from inequality  ( 3 )  and the expectation over  i ∗ .   Plugging into inequality  ( 4 )  one can obtain

E � ∥∇ F ( w i ∗ ;  S ) ∥ 2 � ≤ 2 F 0

TA   (1 +  τ  2 2   η 2 q ) + � η A   + 2 +   τ  2 2   η 3 q

A

� τ   2 1   . (5)

Now recall  A  =   η

2   − L 1 η 2

2 − η 3 τ  2 2   q 2 .   Since  q   ≤ O � 1 τ   2 2   η 2 � and  η   ≤ 1 2 L 1   we have  A  = Θ( η ) .   Thus plugging into inequality

( 5 ) and again using the fact that  q   ≤ O � 1 τ   2 2   η 2 � we have

E � ∥∇ F ( w i ∗ ;  S ) ∥ 2 � =  O � F 0 Tη   (1 +  τ  2 2   η 2 q ) + � 3 +   τ  2 2   η 3 q

A

� τ   2 1

� =  O � F 0 Tη   +  τ  2 1

� .

The claim then follows from the Jensen inequality.

For privacy, we will rely on the moments accountant analysis of ( Abadi et al. ,  2016 ).   This roughly gives the same analysis as using privacy amplification via subsampling and the advanced composition theorem, but allows for improvements in log factors.   We provide the following theorem implicit in ( Abadi et al. ,  2016 ) Theorem 1 below.   The same result can be obtained using the analysis for ( Kulkarni et al. ,  2021 ) Theorem 3.1 which uses the truncated central differential privacy guarantees of the Gaussian mechanism ( Bun et al. ,  2018 ). Theorem B.2  (( Abadi et al. ,  2016 ;  Kulkarni et al. ,  2021 )) .   Let  ε, δ   ∈ (0 ,  1]  and  c  be a universal constant.   Let  D   ∈Y n   be a dataset over some domain  Y , and let  h 1 , ..., h T   :  Y   �→ R d   be a series of (possibly adaptive) queries such that for any  y   ∈Y ,

t  ∈ [ T ] ,  ∥ h t ( y ) ∥ 2   ≤ λ t .   Let  σ t   = cλ t √

log(1 /δ )

ε max � 1 b , √

T n � .   Then the algorithm which samples batches of size  B 1 , .., B t of size  b  uniformly at random and outputs   1

n �

y ∈ B t   h t ( y ) +  g t  for all  t  ∈ [ T ]  where  g t   ∼N (0 ,  I σ 2 t   ) , is  ( ε, δ ) -DP.

We note that the original statement of the Theorem in ( Abadi et al. ,  2016 ) requires  σ t   ≥ cλ t √

T   log(1 /δ )

nε and  T   ≥ n 2 ε

b 2   (or T   ≥ n 2

b 2   so long as  ε  ≤ 1 ).   However, in the case where  T   ≤ n 2

b 2  , one can simply consider the meta algorithm that does run T   ′   =   n 2

b 2   steps and only outputs the first  T   results.   This algorithm is at least as private as the algorithm which outputs every

result, and under the setting  T   ′   the scale of noise is 8 λ t √

log(1 /δ ) bε .

We can now prove the main result for Private Spiderboost, restated below.   We note that the setting of  b 2  given below will always be less than  n  under required conditions.   More details are provided in the proof below.

Theorem   B.3   (Private   Spiderboost) .   Let   n   ≥ max � ( L 0 ε ) 2

F 0 L 1 d  log(1 /δ ) ,

√

d  max { 1 , √ L 1 F 0 /L 0 }

ε

� . Private   Spiderboost

run   with   parameter   settings   η   = 1 2 L 1   ,   b 1   =   n ,   b 2   =

�

max

�� L 0 nε √

F 0 L 1 d  log(1 /δ )

� 2 / 3 ,   ( L 0 nd  log(1 /δ )) 1 / 3

( L 1 F 0 ) 1 / 6 ε 2 / 3

��

,   T   = �

max

�� ( F 0 L 1 ) 1 / 4 nε √

L 0 d  log(1 /δ )

� 4 / 3 , nε √

d  log(1 /δ )

��

, and  q   = � n 2 ε 2

L 2 1 T d  log(1 /δ ) � satisfies

E  [ ∥∇ F ( ˜ w ) ∥ ] =  O





�� F 0 L 1 L 0 d  log (1 /δ )

nε

� 2 / 3

+

� d  log (1 /δ ) L 0

nε





16



| 0                                                                                                                                    |
|:-------------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                              |
| The second term above can be bounded via inequality (3). To bound the first term we have by Lemma 4.1 that                           |
| t∗                                                                                                                                   |
| (cid:104)                                                                                                                            |
| (cid:104)                                                                                                                            |
| (cid:88)                                                                                                                             |
| E                                                                                                                                    |
| E                                                                                                                                    |
| ≤ τ 2                                                                                                                                |
| + τ 2                                                                                                                                |
| ∥∇i∗ − ∇F (wi∗ ; S)∥2(cid:105)                                                                                                       |
| ∥wk − wk−1∥2(cid:105)                                                                                                                |
| 2                                                                                                                                    |
| 1                                                                                                                                    |
| k=st∗ +1                                                                                                                             |
| t∗                                                                                                                                   |
| (cid:104)                                                                                                                            |
| (cid:88)                                                                                                                             |
| E                                                                                                                                    |
| = η2τ 2                                                                                                                              |
| + τ 2                                                                                                                                |
| ∥∇k∥2(cid:105)                                                                                                                       |
| 2                                                                                                                                    |
| 1                                                                                                                                    |
| k=st∗ +1                                                                                                                             |
| (cid:104)                                                                                                                            |
| qη2τ 2                                                                                                                               |
| T(cid:88) k                                                                                                                          |
| 2                                                                                                                                    |
| E                                                                                                                                    |
| ≤                                                                                                                                    |
| + τ 2                                                                                                                                |
| ∥∇k∥2(cid:105)                                                                                                                       |
| 1                                                                                                                                    |
| T                                                                                                                                    |
| =0                                                                                                                                   |
| τ 2                                                                                                                                  |
| η3qτ 2                                                                                                                               |
| 2 η2qF0                                                                                                                              |
| 2                                                                                                                                    |
| ≤                                                                                                                                    |
| +                                                                                                                                    |
| τ 2                                                                                                                                  |
| 1 + τ 2                                                                                                                              |
| 1 ,                                                                                                                                  |
| T A                                                                                                                                  |
| 2A                                                                                                                                   |
| where the last inequality comes from inequality (3) and the expectation over i∗. Plugging into inequality (4) one can obtain         |
| (cid:104)                                                                                                                            |
| (cid:18) η                                                                                                                           |
| τ 2                                                                                                                                  |
| 2F0                                                                                                                                  |
| 2 η3q                                                                                                                                |
| E                                                                                                                                    |
| ≤                                                                                                                                    |
| (1 + τ 2                                                                                                                             |
| + 2 +                                                                                                                                |
| τ 2                                                                                                                                  |
| (5)                                                                                                                                  |
| ∥∇F (wi∗ ; S)∥2(cid:105)                                                                                                             |
| 2 η2q) +                                                                                                                             |
| 1 .                                                                                                                                  |
| T A                                                                                                                                  |
| A                                                                                                                                    |
| A                                                                                                                                    |
| (cid:16)                                                                                                                             |
| (cid:17)                                                                                                                             |
| 1                                                                                                                                    |
| 1                                                                                                                                    |
| 2 q                                                                                                                                  |
| − η3τ 2                                                                                                                              |
| Now recall A = η                                                                                                                     |
| . Since q ≤ O                                                                                                                        |
| and η ≤                                                                                                                              |
| we have A = Θ(η). Thus plugging into inequality                                                                                      |
| 2 − L1η2                                                                                                                             |
| 2                                                                                                                                    |
| τ 2                                                                                                                                  |
| 2L1                                                                                                                                  |
| 2 η2                                                                                                                                 |
| (cid:16)                                                                                                                             |
| (cid:17)                                                                                                                             |
| 1                                                                                                                                    |
| (5) and again using the fact that q ≤ O                                                                                              |
| we have                                                                                                                              |
| τ 2                                                                                                                                  |
| 2 η2                                                                                                                                 |
| (cid:104)                                                                                                                            |
| τ 2                                                                                                                                  |
| (cid:18) F0                                                                                                                          |
| (cid:18) F0                                                                                                                          |
| 2 η3q                                                                                                                                |
| E                                                                                                                                    |
| = O                                                                                                                                  |
| (1 + τ 2                                                                                                                             |
| 3 +                                                                                                                                  |
| τ 2                                                                                                                                  |
| + τ 2                                                                                                                                |
| = O                                                                                                                                  |
| .                                                                                                                                    |
| ∥∇F (wi∗ ; S)∥2(cid:105)                                                                                                             |
| 2 η2q) +                                                                                                                             |
| 1                                                                                                                                    |
| 1                                                                                                                                    |
| T η                                                                                                                                  |
| A                                                                                                                                    |
| T η                                                                                                                                  |
| The claim then follows from the Jensen inequality.                                                                                   |
| For privacy, we will rely on the moments accountant analysis of (Abadi et al., 2016). This roughly gives the same analysis           |
| as using privacy amplification via subsampling and the advanced composition theorem, but allows for improvements in                  |
| log factors. We provide the following theorem implicit in (Abadi et al., 2016) Theorem 1 below. The same result can be               |
| obtained using the analysis for (Kulkarni et al., 2021) Theorem 3.1 which uses the truncated central differential privacy            |
| guarantees of the Gaussian mechanism (Bun et al., 2018).                                                                             |
| Theorem B.2 ((Abadi et al., 2016; Kulkarni et al., 2021)). Let ε, δ ∈ (0, 1] and c be a universal constant. Let D ∈ Y n be a         |
| dataset over some domain Y, and let h1, ..., hT : Y (cid:55)→ Rd be a series of (possibly adaptive) queries such that for any y ∈ Y, |
| √                                                                                                                                    |
| √                                                                                                                                    |
| (cid:111)                                                                                                                            |
| log(1/δ)                                                                                                                             |
| cλt                                                                                                                                  |
| T                                                                                                                                    |
| (cid:110) 1                                                                                                                          |
| max                                                                                                                                  |
| t ∈ [T ], ∥ht(y)∥2 ≤ λt. Let σt =                                                                                                    |
| ε                                                                                                                                    |
| b ,                                                                                                                                  |
| n                                                                                                                                    |
| (cid:80)                                                                                                                             |
| 1n                                                                                                                                   |
| of size b uniformly at random and outputs                                                                                            |
| ht(y) + gt for all t ∈ [T ] where gt ∼ N (0, Iσ2                                                                                     |
| t ), is (ε, δ)-DP.                                                                                                                   |
| y∈Bt                                                                                                                                 |
| √                                                                                                                                    |
| T log(1/δ)                                                                                                                           |
| cλt                                                                                                                                  |
| and T ≥ n2ε                                                                                                                          |
| (or                                                                                                                                  |
| We note that the original statement of the Theorem in (Abadi et al., 2016) requires σt ≥                                             |
| nε                                                                                                                                   |
| b2                                                                                                                                   |
| T ≥ n2                                                                                                                               |
| so long as ε ≤ 1). However, in the case where T ≤ n2                                                                                 |
| b2                                                                                                                                   |
| b2 , one can simply consider the meta algorithm that does run                                                                        |
| T ′ = n2                                                                                                                             |
| steps and only outputs the first T results. This algorithm is at least as private as the algorithm which outputs every               |
| b2                                                                                                                                   |
| √                                                                                                                                    |
| log(1/δ)                                                                                                                             |
| 8λt                                                                                                                                  |
| result, and under the setting T ′                                                                                                    |
| the scale of noise is                                                                                                                |
| .                                                                                                                                    |
| bε                                                                                                                                   |
| We can now prove the main result for Private Spiderboost, restated below. We note that the setting of b2 given below will            |
| always be less than n under required conditions. More details are provided in the proof below.                                       |
| √                                                                                                                                    |
| √                                                                                                                                    |
| (cid:26)                                                                                                                             |
| (cid:27)                                                                                                                             |
| d max{1,                                                                                                                             |
| L1F0/L0}                                                                                                                             |
| (L0ε)2                                                                                                                               |
| .                                                                                                                                    |
| Private                                                                                                                              |
| Spiderboost                                                                                                                          |
| Let n                                                                                                                                |
| ≥ max                                                                                                                                |
| Theorem B.3                                                                                                                          |
| (Private Spiderboost).                                                                                                               |
| ε                                                                                                                                    |
| F0L1d log(1/δ) ,                                                                                                                     |
| (cid:40)(cid:18)                                                                                                                     |
| (cid:19)2/3                                                                                                                          |
| 1                                                                                                                                    |
| L0nε                                                                                                                                 |
| √                                                                                                                                    |
| run with parameter                                                                                                                   |
| ,                                                                                                                                    |
| settings η =                                                                                                                         |
| , T                                                                                                                                  |
| , (L0nd log(1/δ))1/3                                                                                                                 |
| max                                                                                                                                  |
| =                                                                                                                                    |
| b1 = n,                                                                                                                              |
| b2 =                                                                                                                                 |
| 2L1                                                                                                                                  |
| (L1F0)1/6ε2/3                                                                                                                        |
| F0L1d log(1/δ)                                                                                                                       |
| (cid:40)(cid:18)                                                                                                                     |
| (cid:19)4/3                                                                                                                          |
| (cid:106)                                                                                                                            |
| (cid:107)                                                                                                                            |
| (F0L1)1/4nε                                                                                                                          |
| nε                                                                                                                                   |
| n2ε2                                                                                                                                 |
| √                                                                                                                                    |
| √                                                                                                                                    |
| satisfies                                                                                                                            |
| max                                                                                                                                  |
| ,                                                                                                                                    |
| , and q =                                                                                                                            |
| L2                                                                                                                                   |
| 1T d log(1/δ)                                                                                                                        |
| L0d log(1/δ)                                                                                                                         |
|                                                                                                                                    |
|                                                                                                                                    |
| (cid:32) (cid:112)F0L1L0d log (1/δ)                                                                                                  |
| (cid:112)d log (1/δ)L0                                                                                                               |
| +                                                                                                                                    |
| E [∥∇F ( ˜w)∥] = O                                                                                                                   |
| nε                                                                                                                                   |
| nε                                                                                                                                   |
| 16                                                                                                                                   |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

is  ( ε, δ ) -DP and has oracle complexity   ˜ O � max �� n 5 / 3 ε 2 / 3

d 1 / 3 � , � nε √

d

� 2 �� .

Proof.   For privacy, we rely on the moment accountant analysis of the Gaussian mechanism as per Theorem  B.2 .   Note that each gradient estimate computed in line  9  has elements with  ℓ 2 -norm at most  L 0 , and this estimate is computed at most   T

q times.   Similarly, for a gradient variation at step  t  in line  13  we have norm bound  L 1  ∥ w t  − w t − 1 ∥ , and have that at most  T such estimates are computed.   As such, the scale of noise in both cases ensures the overall algorithm is  ( ε, δ ) -DP by Theorem B.2 .

We now prove the convergence result.   To simplify notation in the following, we define   ¯ α   = √

d  log(1 /δ )

nϵ .   If  b 1   =   n  (full

batch gradient), the conditions of Lemma  4.1  are satisfied with  τ   2 1   =  O � L 2 0 T  ¯ α 2

q � and  τ   2 2   =  O � L 2 1 b 2   +  L 2 1 T  ¯ α 2 � and some

setting of  q  so long as  T   ≥ q   n 2

b 2 1   =  q  and  T   ≥ n 2

b 2 2   .   Further, if  b 2   ≥ 1 T  ¯ α 2   then  τ  2 2   =  O � L 2 1 T  ¯ α 2 � .   Thus the condition on  q  in

Lemma  B.1  is satisfied with  q   =   L 2 1 τ   2 2   = 1 T  ¯ α 2   since  η   = 1 2 L 1

Plugging into Eqn.   ( 1 ) we obtain

E  [ ∥∇ F ( ˜ w ) ∥ ] =  O

�� F 0 L 1

T +   L 0 √

T  ¯ α √ q

�

=  O

�� F 0 L 1

T +  L 0 T  ¯ α 2 �

. (6)

We   now   consider   the   setting   of   T .   Since   q   = 1 T  ¯ α 2  ,   it   suffices   to   set   T   ≥ 1 ¯ α   to   ensure   T   ≥ q .   We   now   set   T   =

max �� ( L 1 F 0 ) 1 / 4

√ L 0  ¯ α

� 4 / 3 ,   1

¯ α

� .   Using Eqn.   ( 6 ) above we have

E  [ ∥∇ F ( ˜ w ) ∥ ] =  O ��� F 0 L 1 L 0 ¯ α � 2 / 3 +  L 0 ¯ α � .

The claimed rate now follows if there exists a valid setting for  b 2  satisfying the previously stated conditions.   The restrictions

on   the   batch   size   implied   by   T   imply   we   need   b 2   ≥ n √

T   and   thus   it   suffices   to   have   b 2   ≥ L 1 / 3 0 n ¯ α 2 / 3

( L 1 F 0 ) 1 / 6   to   satisfy   this

condition   since   T   ≥ � ( L 1 F 0 ) 1 / 4

√ L 0  ¯ α

� 4 / 3 .   We   recall   that   for   the   setting   of   q   to   be   valid   we   also   require   b 2   ≥ 1 T  ¯ α 2   and

because  T   ≥ � ( L 1 F 0 ) 1 / 4

√ L 0  ¯ α

� 4 / 3 it suffices that  b 2   ≥ � L 0 √ F 0 L 1  ¯ α

� 2 / 3 .   Thus we need  b 2   =   max �� L 0 √ F 0 L 1  ¯ α

� 2 / 3 ,   L 1 / 3 0 n ¯ α 2 / 3

( L 1 F 0 ) 1 / 6

� .

Finally,   we   need   b 2   ≤ n   whenever   q   ≥ 1 .   Note   that   by   the   setting   of   q   and   T   we   have   q   ≤ � L 0 √ F 0 L 1  ¯ α

� 2 / 3 and   thus

q   ≥ 1   = ⇒ � √ L 1 F 0  ¯ α L 0

� ≤ 1 .   Under this same condition we have   L 1 / 3 0 n ¯ α 2 / 3

( L 1 F 0 ) 1 / 6   ≤ n .   We further have � L 0 √ F 0 L 1  ¯ α

� 2 / 3 ≤ n

under the assumption  n  ≥ ( L 0 ε ) 2

F 0 L 1 d  log(1 /δ )   given in the theorem statement.   It can also be verified that under the condition on n  given in the theorem statement that  q   ≥ 1 .   Thus the parameter settings obtain the claimed rate.

Note the number of gradient computations is bounded by

O � Tb 2  +   Tb 1

q

� =   ˜ O

�� nε √

d

� 4 / 3 max

�� nε √

d

� 2 / 3 ,   ( nd ) 1 / 3

ε 2 / 3

�

+  n � nε √

d

� 2 / 3 �

=   ˜ O

�

max

�� nε √

d

� 2 ,  n 5 / 3 ε 2 / 3

d 1 / 3

��

.

B.2. Additional Discussion of Rate Improvement Challenges

We here give a more detailed version of the informal discussion in Section  4.2 .   We want to emphasize that the goal of the following discussion is not to provide a universal lower bound, but rather to inform future research.

17



| 0                                                                                                                            |
|:-----------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                      |
| (cid:17)                                                                                                                     |
| (cid:26)(cid:16) n5/3ε2/3                                                                                                    |
| (cid:16) nε√                                                                                                                 |
| .                                                                                                                            |
| is (ε, δ)-DP and has oracle complexity ˜O                                                                                    |
| max                                                                                                                          |
| ,                                                                                                                            |
| d1/3                                                                                                                         |
| d                                                                                                                            |
| Proof. For privacy, we rely on the moment accountant analysis of the Gaussian mechanism as per Theorem B.2. Note that        |
| each gradient estimate computed in line 9 has elements with ℓ2-norm at most L0, and this estimate is computed at most T      |
| q                                                                                                                            |
| times. Similarly, for a gradient variation at step t in line 13 we have norm bound L1 ∥wt − wt−1∥, and have that at most T   |
| such estimates are computed. As such, the scale of noise in both cases ensures the overall algorithm is (ε, δ)-DP by Theorem |
| B.2.                                                                                                                         |
| √                                                                                                                            |
| d log(1/δ)                                                                                                                   |
| We now prove the convergence result. To simplify notation in the following, we define ¯α =                                   |
| .                                                                                                                            |
| If b1 = n (full                                                                                                              |
| nϵ                                                                                                                           |
| (cid:17)                                                                                                                     |
| (cid:16) L2                                                                                                                  |
| (cid:16) L2                                                                                                                  |
| 0T ¯α2                                                                                                                       |
| 1                                                                                                                            |
| + L2                                                                                                                         |
| batch gradient), the conditions of Lemma 4.1 are satisfied with τ 2                                                          |
| and τ 2                                                                                                                      |
| and some                                                                                                                     |
| 1T ¯α2(cid:17)                                                                                                               |
| 1 = O                                                                                                                        |
| 2 = O                                                                                                                        |
| q                                                                                                                            |
| b2                                                                                                                           |
| 1                                                                                                                            |
| setting of q so long as T ≥ q n2                                                                                             |
| = q and T ≥ n2                                                                                                               |
| then τ 2                                                                                                                     |
| . Further, if b2 ≥                                                                                                           |
| 2 = O (cid:0)L2                                                                                                              |
| 1T ¯α2(cid:1). Thus the condition on q in                                                                                    |
| T ¯α2                                                                                                                        |
| b2                                                                                                                           |
| b2                                                                                                                           |
| 1                                                                                                                            |
| 2                                                                                                                            |
| 1                                                                                                                            |
| 1                                                                                                                            |
| =                                                                                                                            |
| Lemma B.1 is satisfied with q = L2                                                                                           |
| since η =                                                                                                                    |
| T ¯α2                                                                                                                        |
| τ 2                                                                                                                          |
| 2L1                                                                                                                          |
| 2                                                                                                                            |
| Plugging into Eqn. (1) we obtain                                                                                             |
| √                                                                                                                            |
| (cid:33)                                                                                                                     |
| (cid:32)(cid:114)                                                                                                            |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Let  L   :   R d   �→ R  be a loss function.   We say the randomized mapping  O   :   R d   ×  ( R d   ∪⊥ )   �→ R d , is a  ( τ 1 , τ 2 ) -accurate oracle for  L  if  ∀ w, w ′   ∈ R d

E O   [ O ( w,  ⊥ )] =  ∇L ( w ) , E O   [ O ( w, w ′ )] =  ∇L ( w )  −∇L ( w ′ )

E O

� ∥O ( w,  ⊥ )  −∇L ( w ) ∥ 2 � ≤ τ   2 1   , E O

� ∥O ( w, w ′ ) ∥ 2 � ≤ τ   2 2   ∥ w  − w ′ ∥ 2  .

In short,  O  is an unbiased and accurate gradient/gradient variation oracle for  L .   Define

m ( G, L 1 ,  L 0 , τ 1 , τ 2 ) = inf A   sup O , L inf � α  :  E  [ ∥∇L ( A ( O , L 1 ,  L 0 , τ 1 , τ 2 ) ∥ ]  ≤ α � ,

where the supremum is taken over  L 1 -smooth functions  L  satisfying  L (0)  − arg min w ∈ R d {L ( w ) } ≤L 0 , and  ( τ 1 , τ 2 ) -accurate

oracles for  L .   The infimum is taken over algorithms which make at most  G  calls to  O .

We have the following lower bound on  m  (i.e.   a lower bound on the accuracy of optimization algorithms which make at most  G  queries to the oracle) following from ( Arjevani et al. ,  2019 , Theorem 3) and the fact that the oracle model described above is a special case of the multi-query oracles considered by ( Arjevani et al. ,  2019 ).

Theorem B.4  (( Arjevani et al. ,  2019 )) .   Let  G,  L 0 , L 1 , τ 1 , τ 2   ≥ 0  and define  α  = � L 0 τ 2 τ 1 G � 1 / 3  + τ 1 √

G .   If  d  =   ˜Ω �� L 0 L 1 α 2 � 2 � ,

then  m ( G, L 1 ,  L 0 , τ 1 , τ 2 ) = Ω( α ) .

Now   consider   L   such   that   L ( w )   =   1

n �

x ∈ S   ℓ ( w ;  x )   for   some   L 0 -Lipschitz   and   L 1 -smooth   loss   ℓ :   R d   × X   �→ R   and S   ∈X   n .   We are interested in designing some  ( � τ 1 ,  � τ 2 ) -accurate and differentially private oracle,   � O , which can then be used by an optimization algorithm,  A , to obtain an approximate stationary point   ¯ w   =  A (   � O , L 1 ,  L 0 ,  � τ 1 ,  � τ 2 ) .   Specifically, we want � O  to be capable of answering  G  queries under  ( ε, δ ) -DP. A common method for achieving this is to ensure each query to O  is at least  ( ε √

G , δ ) -DP and use advanced composition (or the more refined moment accountant) analysis.   Such a setup encapsulates numerous results in the convex setting ( Bassily et al. ,  2019 ;  Kulkarni et al. ,  2021 ), and is even more dominant in non-convex settings ( Wang et al. ,  2017 ;  Zhou et al. ,  2020 ;  Abadi et al. ,  2016 ).

Our key observation is that under such a setup, any increase in the number of oracle calls to  G  must be met with a proportional increase in the accuracy parameters  ( � τ 1 ,  � τ 2 ) .   Thus, if such an oracle,   � O  is applied in a black box fashion to a stochastic optimization algorithm  A , one can obtain a lower bound on the accuracy of the overall algorithm independent of  G .

Specifically, since estimating the gradient and gradient variation can be viewed as mean estimation problems on  n  vectors, we can use fingerprinting code arguments to lower bound  � τ 1  and  � τ 2  ( Steinke & Ullman ,  2015 ).   In Lemma  B.5  below, we

prove that any  ( � τ 1 ,  � τ 2 ) -accurate oracle which ensures that any query is  ( ε √

G , δ ) -DP must have  � τ 1   =   Ω � L 0 √ Gd  log(1 /δ )

nε �

and  � τ 2   = Ω � L 1 √ Gd  log(1 /δ )

nε � .   Now, observe that by Theorem  B.4 , we have

m ( G, L 1 ,  L 0 ,  � τ 1 ,  � τ 2 ) = Ω





� √ F 0 L 1 L 0 � d  log (1 /δ ) nε

� 2 / 3

+   L 0 � d  log (1 /δ )

nε



 ,

which matches our upper bound.

We   now   remark   on   several   ways   the   above   barrier   could   be   circumvented.   The   first   and   most   obvious   possibility   is   to employ a different privatization method than private oracles.   However, this is particularly difficult in the nonconvex setting as existing methods which avoid private gradients (see e.g.   ( Feldman et al. ,  2020 ) for several such methods) rely crucially on   stability   guarantees   arising   from   convexity.   Other   possible   ways   to   beat   the   above   rate   is   by   designing   a   stochastic optimization algorithm which leverages the structure of the noise used in private implementations of the oracle or makes use of additional assumptions to beat the  Ω �� L 0 τ 2 τ 1 G � 1 / 3  + τ 1 √

G

� non-private lower bound.

Additional   Details   on   Fingerprinting   Bound We   conclude   by   giving   a   concrete   construction   for   the   fingerprinting argument mentioned above.

18



| 0                                                                                                                                                    |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                                              |
| Let L : Rd (cid:55)→ R be a loss function. We say the randomized mapping O : Rd × (Rd ∪ ⊥) (cid:55)→ Rd,                                             |
| is a (τ1, τ2)-accurate                                                                                                                               |
| oracle for L if ∀w, w′ ∈ Rd                                                                                                                          |
| E O                                                                                                                                                  |
| E O                                                                                                                                                  |
| [O(w, ⊥)] = ∇L(w),                                                                                                                                   |
| [O(w, w′)] = ∇L(w) − ∇L(w′)                                                                                                                          |
| E O                                                                                                                                                  |
| E O                                                                                                                                                  |
| ∥O(w, ⊥) − ∇L(w)∥2(cid:105)                                                                                                                          |
| ≤ τ 2                                                                                                                                                |
| ∥O(w, w′)∥2(cid:105)                                                                                                                                 |
| ≤ τ 2                                                                                                                                                |
| 1 ,                                                                                                                                                  |
| 2 ∥w − w′∥2 .                                                                                                                                        |
| In short, O is an unbiased and accurate gradient/gradient variation oracle for L. Define                                                             |
| (cid:110)                                                                                                                                            |
| (cid:111)                                                                                                                                            |
| sup                                                                                                                                                  |
| inf                                                                                                                                                  |
| ,                                                                                                                                                    |
| m(G, L1, L0, τ1, τ2) = inf                                                                                                                           |
| α : E [∥∇L(A(O, L1, L0, τ1, τ2)∥] ≤ α                                                                                                                |
| A                                                                                                                                                    |
| O,L                                                                                                                                                  |
| where the supremum is taken over L1-smooth functions L satisfying L(0) − arg min                                                                     |
| {L(w)} ≤ L0, and (τ1, τ2)-accurate                                                                                                                   |
| w∈Rd                                                                                                                                                 |
| oracles for L. The infimum is taken over algorithms which make at most G calls to O.                                                                 |
| We have the following lower bound on m (i.e. a lower bound on the accuracy of optimization algorithms which make at                                  |
| most G queries to the oracle) following from (Arjevani et al., 2019, Theorem 3) and the fact that the oracle model described                         |
| above is a special case of the multi-query oracles considered by (Arjevani et al., 2019).                                                            |
| (cid:16)(cid:2) L0L1                                                                                                                                 |
| (cid:3)2(cid:17)                                                                                                                                     |
| ,                                                                                                                                                    |
| + τ1√                                                                                                                                                |
| . If d = ˜Ω                                                                                                                                          |
| Theorem B.4 ((Arjevani et al., 2019)). Let G, L0, L1, τ1, τ2 ≥ 0 and define α = (cid:0) L0τ2τ1                                                       |
| G                                                                                                                                                    |
| α2                                                                                                                                                   |
| G                                                                                                                                                    |
| then m(G, L1, L0, τ1, τ2) = Ω (α).                                                                                                                   |
| (cid:80)                                                                                                                                             |
| Now consider L such that L(w) = 1                                                                                                                    |
| x∈S ℓ(w; x) for some L0-Lipschitz and L1-smooth loss ℓ : Rd × X (cid:55)→ R and                                                                      |
| n                                                                                                                                                    |
| S ∈ X n. We are interested in designing some ((cid:98)τ1, (cid:98)τ2)-accurate and differentially private oracle, (cid:98)O, which can then be used  |
| by an optimization algorithm, A, to obtain an approximate stationary point ¯w = A( (cid:98)O, L1, L0, (cid:98)τ1, (cid:98)τ2). Specifically, we want |
| O to be capable of answering G queries under (ε, δ)-DP. A common method for achieving this is to ensure each query to                                |
| ε                                                                                                                                                    |
| √                                                                                                                                                    |
| O is at least (                                                                                                                                      |
| , δ)-DP and use advanced composition (or the more refined moment accountant) analysis. Such a setup                                                  |
| G                                                                                                                                                    |
| encapsulates numerous results in the convex setting (Bassily et al., 2019; Kulkarni et al., 2021), and is even more dominant                         |
| in non-convex settings (Wang et al., 2017; Zhou et al., 2020; Abadi et al., 2016).                                                                   |
| Our key observation is that under such a setup, any increase in the number of oracle calls to G must be met with a proportional                      |
| increase in the accuracy parameters ((cid:98)τ1, (cid:98)τ2). Thus, if such an oracle, (cid:98)O is applied in a black box fashion to a stochastic   |
| optimization algorithm A, one can obtain a lower bound on the accuracy of the overall algorithm independent of G.                                    |
| Specifically, since estimating the gradient and gradient variation can be viewed as mean estimation problems on n vectors,                           |
| we can use fingerprinting code arguments to lower bound (cid:98)τ1 and (cid:98)τ2 (Steinke & Ullman, 2015). In Lemma B.5 below, we                   |
| (cid:17)                                                                                                                                             |
| Gd log(1/δ)                                                                                                                                          |
| (cid:16) L0                                                                                                                                          |
| ε                                                                                                                                                    |
| √                                                                                                                                                    |
| nε                                                                                                                                                   |
| prove that any ((cid:98)τ1, (cid:98)τ2)-accurate oracle which ensures that any query is (                                                            |
| , δ)-DP must have (cid:98)τ1 = Ω                                                                                                                     |
| G                                                                                                                                                    |
| √                                                                                                                                                    |
| (cid:17)                                                                                                                                             |
| Gd log(1/δ)                                                                                                                                          |
| (cid:16) L1                                                                                                                                          |
| . Now, observe that by Theorem B.4, we have                                                                                                          |
| nε                                                                                                                                                   |
| and (cid:98)τ2 = Ω                                                                                                                                   |
| (cid:33)2/3                                                                                                                                          |
| (cid:32) √                                                                                                                                           |
|                                                                                                                                                    |
|                                                                                                                                                    |
| (cid:112)d log (1/δ)                                                                                                                                 |
| (cid:112)d log (1/δ)                                                                                                                                 |
| F0L1L0                                                                                                                                               |
| L0                                                                                                                                                   |
| +                                                                                                                                                    |
| ,                                                                                                                                                    |
| m(G, L1, L0, (cid:98)τ1, (cid:98)τ2) = Ω                                                                                                             |
| nε                                                                                                                                                   |
| nε                                                                                                                                                   |
| which matches our upper bound.                                                                                                                       |
| We now remark on several ways the above barrier could be circumvented. The first and most obvious possibility is to                                  |
| employ a different privatization method than private oracles. However, this is particularly difficult in the nonconvex setting                       |
| as existing methods which avoid private gradients (see e.g. (Feldman et al., 2020) for several such methods) rely crucially                          |
| on stability guarantees arising from convexity. Other possible ways to beat                                                                          |
| the above rate is by designing a stochastic                                                                                                          |
| optimization algorithm which leverages the structure of the noise used in private implementations of the oracle or makes use                         |
| (cid:16)(cid:0) L0τ2τ1                                                                                                                               |
| (cid:1)1/3                                                                                                                                           |
| + τ1√                                                                                                                                                |
| of additional assumptions to beat the Ω                                                                                                              |
| non-private lower bound.                                                                                                                             |
| G                                                                                                                                                    |
| G                                                                                                                                                    |
| Additional Details on Fingerprinting Bound                                                                                                           |
| We conclude by giving a concrete construction for the fingerprinting                                                                                 |
| argument mentioned above.                                                                                                                            |
| 18                                                                                                                                                   |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Lemma B.5.   Let  L 0 , L 1   ≥ 0 ,  ε   =   O (1) ,  2 − Ω( n )   ≤ δ   ≤ 1 n 1+Ω(1)   and � d  log (1 /δ ) / ( nε )   =   O (1) .   Let  ℓ,  L , S   satisfy the assumptions above.   Then there exists  ℓ, S   such that for any oracle,  O , which is  ( τ 1 , τ 2 ) -accurate for  L  it holds that

τ 1   = Ω

� L 0 � d  log (1 /δ )

nε

�

and τ 2   = Ω

� L 1 � d  log (1 /δ )

nε

�

.

Proof.   In the following, we use  u j   to denote the  j ’th component of some vector  u .   Let  B   = L 0 L 1 √

d   and define  h  :  R  �→ R  as

h ( z ) =

� L 1 2   w 2 if | w | ≤ B

L 0 √

d | w | − L 2 0 2 dL 1 otherwise

Define   d ′   = d 2   (assume   d   is   even   for   simplicity)   and   for   any   vector   u   ∈ R d   let   u (1)   =   [ u 1 , ..., u d ′ ] ⊤ and   u (2)   = [ u d ′ +1 , ..., u d ] ⊤ .   Define  ℓ ( w ;  x ) =  ℓ 1 ( w ;  x ) +  ℓ 2 ( w ;  x )  where

ℓ 1 ( w ;  x ) =   L 0 √

d

� w (1) , x (1) � , ℓ 2 ( w ;  x ) =   1

2

d �

j = d ′ +1 h ( w j ) x j .

Let  W   =  { w   :  ∥ w ∥ ∞ ≤ B }  and note for any  w   ∈W   we have

∇ ℓ ( w ;  x ) = [   x 1 √

d , ...,   x d ′ √

d , w d ′ +1 x d ′ +1 , ..., w d x d ] ⊤ , ∇ 2 ℓ 2 ( w ;  x ) =  L 1  ·  Diag (0 , ...,  0 , x d ′ +1 , ..., x d )

That is, the Hessian of  ℓ 2 ( w ;  x )  is a diagonal matrix with entries from  x .   Thus one can observe that for any  x  ∈{± 1 } d   we have that  ℓ ( · ;  x )  is  L 0 -Lipschitz and  L 1 -smooth over  R d .

To prove a lower bound on  τ 1  and  τ 2 , it suffices to show that for any  ( ε, δ ) -DP implementation of  O  there exists  w   ∈ R d

such that  E O

� ∥O ( w ;  ⊥ )  −∇L ( w ) ∥ 2 � ≥ τ   2 1   and there exist  w, w ′   ∈ R d   such that  E O

� ∥O ( w, w ′ ) ∥ 2 � ≥ τ   2 2   ∥ w  − w ′ ∥ 2 .   For

sake of generality, we will show that these properties hold for a set of  w, w ′ .

Note that to lower bound the gradient error, it suffices to lower bound the error with respect to the first  d ′   components.   We thus argue using  ℓ 1 , and will in fact show a lower bound for any  w   ∈ R d .   Let  w   ∈ R d .   We have for any  ( ε, δ ) -DP oracle  O there exists a dataset  S   ⊆{± 1 } d , where  | S |  =  n , of fingerprinting codes such that

E O   [ ∥O ( w ;  ⊥ )  −∇L ( w ) ∥ ]  ≥ E O

������ O ( w ;  ⊥ ) (1)  − 1 n

�

x ∈ S x (1) �����

�

= Ω

� L 0 � d  log (1 /δ )

nε

�

.

The bound follows from standard fingerprinting code arguments.   See ( Bassily et al. ,  2014 , Lemma 5.1) for a lower bound and ( Steinke & Ullman ,  2015 , Theorem 1.1) for a group privacy reduction that obtains the additional � log (1 /δ )  factor.   This

fingerprinting result also induces the parameter constraints in the theorem statement.   We thus have  τ 1   = Ω � L 0 √

d  log(1 /δ )

nε

� .

Similarly, we will argue a bound on the gradient variation using  ℓ 2 .   Let  w, w ′   ∈W   and  u  = ( w  − w ′ ) (2) .   In what follows, we   only   use   the   second   half   of   the   components   for   each   vector,   and   thus   omit   the   superscript   (2)   from   all   vectors   for readability.   We have  ∇ ℓ 2 ( w ;  x )  −∇ ℓ 2 ( w ′ ;  x )   =   L 1 [ u 1 x 1 , ..., u d ′ x d ′ ] ⊤ .   Then for any  c   ∈ (0 , 2 L 0 L 1 √

d ]  and  u   ∈{± c } 2   we

19



| 0                                                                                                                                   |
|:------------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                             |
| Lemma B.5. Let L0, L1 ≥ 0, ε = O(1), 2−Ω(n) ≤ δ ≤                                                                                   |
| n1+Ω(1) and (cid:112)d log (1/δ)/(nε) = O(1). Let ℓ, L, S satisfy the                                                               |
| assumptions above. Then there exists ℓ, S such that for any oracle, O, which is (τ1, τ2)-accurate for L it holds that               |
| (cid:32)                                                                                                                            |
| (cid:33)                                                                                                                            |
| (cid:33)                                                                                                                            |
| (cid:32)                                                                                                                            |
| (cid:112)d log (1/δ)                                                                                                                |
| (cid:112)d log (1/δ)                                                                                                                |
| L1                                                                                                                                  |
| L0                                                                                                                                  |
| and                                                                                                                                 |
| .                                                                                                                                   |
| τ2 = Ω                                                                                                                              |
| τ1 = Ω                                                                                                                              |
| nε                                                                                                                                  |
| nε                                                                                                                                  |
| L0                                                                                                                                  |
| √                                                                                                                                   |
| Proof.                                                                                                                              |
| to denote the j’th component of some vector u. Let B =                                                                              |
| and define h : R (cid:55)→ R as                                                                                                     |
| In the following, we use uj                                                                                                         |
| d                                                                                                                                   |
| L1                                                                                                                                  |
| (cid:40) L1                                                                                                                         |
| if|w| ≤ B                                                                                                                           |
| 2 w2                                                                                                                                |
| h(z) =                                                                                                                              |
| L0√                                                                                                                                 |
| |w| − L2                                                                                                                            |
| otherwise                                                                                                                           |
| 2dL1                                                                                                                                |
| d                                                                                                                                   |
| Define d′ = d                                                                                                                       |
| (assume d is even for simplicity) and for any vector u ∈ Rd                                                                         |
| let u(1) = [u1, ..., ud′]⊤ and u(2) =                                                                                               |
| 2                                                                                                                                   |
| [ud′+1, ..., ud]⊤. Define ℓ(w; x) = ℓ1(w; x) + ℓ2(w; x) where                                                                       |
| d(cid:88)                                                                                                                           |
| (cid:68)                                                                                                                            |
| 1 2                                                                                                                                 |
| L0√                                                                                                                                 |
| w(1), x(1)(cid:69)                                                                                                                  |
| ,                                                                                                                                   |
| h(wj)xj.                                                                                                                            |
| ℓ1(w; x) =                                                                                                                          |
| ℓ2(w; x) =                                                                                                                          |
| d                                                                                                                                   |
| j=d′+1                                                                                                                              |
| Let W = {w : ∥w∥∞ ≤ B} and note for any w ∈ W we have                                                                               |
| xd′                                                                                                                                 |
| x1√                                                                                                                                 |
| √                                                                                                                                   |
| ∇ℓ(w; x) = [                                                                                                                        |
| , ...,                                                                                                                              |
| , wd′+1xd′+1, ..., wdxd]⊤,                                                                                                          |
| ∇2ℓ2(w; x) = L1 · Diag(0, ..., 0, xd′+1, ..., xd)                                                                                   |
| d                                                                                                                                   |
| d                                                                                                                                   |
| That is, the Hessian of ℓ2(w; x) is a diagonal matrix with entries from x. Thus one can observe that for any x ∈ {±1}d we           |
| have that ℓ(·; x) is L0-Lipschitz and L1-smooth over Rd.                                                                            |
| To prove a lower bound on τ1 and τ2, it suffices to show that for any (ε, δ)-DP implementation of O there exists w ∈ Rd             |
| ∥O(w; ⊥) − ∇L(w)∥2(cid:105)                                                                                                         |
| ≥ τ 2                                                                                                                               |
| ∥O(w, w′)∥2(cid:105)                                                                                                                |
| ≥ τ 2                                                                                                                               |
| such that E                                                                                                                         |
| 1 and there exist w, w′ ∈ Rd such that E                                                                                            |
| 2 ∥w − w′∥2. For                                                                                                                    |
| O                                                                                                                                   |
| O                                                                                                                                   |
| sake of generality, we will show that these properties hold for a set of w, w′.                                                     |
| Note that to lower bound the gradient error, it suffices to lower bound the error with respect to the first d′ components. We       |
| thus argue using ℓ1, and will in fact show a lower bound for any w ∈ Rd. Let w ∈ Rd. We have for any (ε, δ)-DP oracle O             |
| there exists a dataset S ⊆ {±1}d, where |S| = n, of fingerprinting codes such that                                                  |
| (cid:33)                                                                                                                            |
| (cid:35)                                                                                                                            |
| (cid:32)                                                                                                                            |
| (cid:34)(cid:13)                                                                                                                    |
| (cid:112)d log (1/δ)                                                                                                                |
| L0                                                                                                                                  |
| E O                                                                                                                                 |
| (cid:13)(cid:13)(cid:13)(cid:13)                                                                                                    |
| 1 n                                                                                                                                 |
| (cid:88) x                                                                                                                          |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                                                                                            |
| [∥O(w; ⊥) − ∇L(w)∥] ≥ E                                                                                                             |
| x(1)                                                                                                                                |
| .                                                                                                                                   |
| O(w; ⊥)(1) −                                                                                                                        |
| = Ω                                                                                                                                 |
| O                                                                                                                                   |
| nε                                                                                                                                  |
| ∈S                                                                                                                                  |
| The bound follows from standard fingerprinting code arguments. See (Bassily et al., 2014, Lemma 5.1) for a lower bound              |
| and (Steinke & Ullman, 2015, Theorem 1.1) for a group privacy reduction that obtains the additional (cid:112)log (1/δ) factor. This |
| √                                                                                                                                   |
| (cid:18)                                                                                                                            |
| (cid:19)                                                                                                                            |
| d log(1/δ)                                                                                                                          |
| L0                                                                                                                                  |
| .                                                                                                                                   |
| fingerprinting result also induces the parameter constraints in the theorem statement. We thus have τ1 = Ω                          |
| nε                                                                                                                                  |
| Similarly, we will argue a bound on the gradient variation using ℓ2. Let w, w′ ∈ W and u = (w − w′)(2). In what follows,            |
| (2)                                                                                                                                 |
| we only use the second half of the components for each vector, and thus omit                                                        |
| the superscript                                                                                                                     |
| from all vectors for                                                                                                                |
| 2L0                                                                                                                                 |
| √                                                                                                                                   |
| ] and u ∈ {±c}2 we                                                                                                                  |
| readability. We have ∇ℓ2(w; x) − ∇ℓ2(w′; x) = L1[u1x1, ..., ud′xd′]⊤. Then for any c ∈ (0,                                          |
| d                                                                                                                                   |
| L1                                                                                                                                  |
| 19                                                                                                                                  |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

have

E O

� ∥O ( w, w ′ )  − ( ∇L ( w )  −∇L ( w ′ )) ∥ 2 � =  L 2 1   ·  E O



 d ′ �

j =1

�

O ( w, w ′ ) j   − u j

n

�

x ∈ S x j

� 2 



=  L 2 1   ·  E O



 d ′ �

j =1

�

u j � O ( w, w ′ ) j u j − 1

n

�

x ∈ S x j � � 2 



=  L 2 1   ·  E O



 c 2 d ′ �

j =1

� O ( w, w ′ ) j

u j − 1

n

�

x ∈ S x j

� 2 



= Ω � L 2 1 c 2  d 2  log (1 /δ )

n 2 ε 2

� ,

where the last step again comes from fingerprinting results.   Note that the extra factor of  d  as compared to the previous bound comes from the fact that we are considering fingerprinting codes with norm larger by a factor of √

d .   We also use the

fact that the vector  O ( w, w ′ )  transformed using  u  is  ( ε, δ ) -DP by post processing.   Now since  c  =   ∥ w − w ′ ∥ √

d we have

E O   [ ∥O ( w, w ′ )  − ( ∇L ( w )  −∇L ( w ′ )) ∥ ] =

�

L 1  ∥ w  − w ′ ∥

� d  log (1 /δ )

nε

�

.

Finally, noting that  E O

� ∥O ( w, w ′ )  − ( ∇L ( w )  −∇L ( w ′ )) ∥ 2 � ≤ E O

� ∥O ( w, w ′ ) ∥ 2 � we obtain  τ 2   = Ω � L 1 √ d  log(1 /δ )

nε � .   This

completes the proof.

We remark that the accuracy lower bound for the gradient variation can hold for a much more general set of vectors than that given in the proof.   Specifically, the same result can be obtained for any  u  =  w  − w ′   such that  u  has  Θ( d )  components which are  Ω � ∥ u ∥ √

d � (i.e.   any sufficiently spread out vector).   This uses the fact that it suffices to bound the number of components which disagree in sign with the fingerprinting mean and that fingerprinting codes are sampled using a product distribution, and thus the tracing attack used by fingerprinting constructions holds over any sufficiently large subset of dimensions.

C. Missing Results for Population Stationary Points

Here we present the proof of privacy and accuracy for Algorithm  1 .   We start by proving the privacy guarantee.

Proof of Theorem  3.1 .   By parallel composition of differential privacy, and since the used batches are disjoint, it suffices to prove that each step in lines  6  and  15  of the algorithm is  ( ε, δ ) -DP. Note that the gradient estimator in step  6  has  ℓ 2 -sensitivity 2 L 0 /b , so by the Gaussian mechanism this step is  ( ε, δ ) -DP.

For step  15 , suppose  S t,s  and  S ′ t,s   are neighboring datasets that differ in at most one element:   x i ∗ ̸ =  x ′ i ∗ , and let  η t,s i   and η ′ t,s i   the respective stepsizes used in step  23 .   Then

∥ ∆ t,s  − ∆ ′ t,s ∥ =   2 | s |

b   ∥∇ f  ( w t,s ;  x i ∗ )  −∇ f  ( w t, � s ;  x i ∗ )  − ( ∇ f  ( w t,s ;  x ′ i ∗ )  −∇ f  ( w t, � s ;  x ′ i ∗ ))  ∥ ,

and note between the parent node  u t, � s   and  u t,s   there are  2 D −| s |   iterates generated by the algorithm, which we denote as

20



| 0                                                                                                                                   |
|:------------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                             |
| have                                                                                                                                |
| d′                                                                                                                                  |
| (cid:32)                                                                                                                            |
| (cid:33)2                                                                                                                          |
|                                                                                                                                   |
| (cid:104)                                                                                                                           |
| E O                                                                                                                                 |
| (cid:88) j                                                                                                                          |
| (cid:88) x                                                                                                                          |
| uj                                                                                                                                  |
| ∥O(w, w′) − (∇L(w) − ∇L(w′))∥2(cid:105)                                                                                             |
| = L2                                                                                                                                |
| O(w, w′)j −                                                                                                                         |
| xj                                                                                                                                  |
| 1 · E                                                                                                                               |
|                                                                                                                                    |
| O                                                                                                                                   |
| n                                                                                                                                   |
| =1                                                                                                                                  |
| ∈S                                                                                                                                  |
| d′                                                                                                                                  |
| (cid:32)                                                                                                                            |
| (cid:33)2                                                                                                                          |
|                                                                                                                                   |
| (cid:17)                                                                                                                            |
| (cid:16) O(w, w′)j                                                                                                                  |
| (cid:88) j                                                                                                                          |
| 1 n                                                                                                                                 |
| (cid:88) x                                                                                                                          |
| = L2                                                                                                                                |
| −                                                                                                                                   |
| xj                                                                                                                                  |
| uj                                                                                                                                  |
| 1 · E                                                                                                                               |
|                                                                                                                                    |
| O                                                                                                                                   |
| uj                                                                                                                                  |
| =1                                                                                                                                  |
| ∈S                                                                                                                                  |
| d′                                                                                                                                  |
| (cid:32)                                                                                                                            |
| (cid:33)2                                                                                                                          |
|                                                                                                                                   |
| O(w, w′)j                                                                                                                           |
| (cid:88) j                                                                                                                          |
| 1 n                                                                                                                                 |
| (cid:88) x                                                                                                                          |
| = L2                                                                                                                                |
| c2                                                                                                                                  |
| −                                                                                                                                   |
| xj                                                                                                                                  |
| 1 · E                                                                                                                               |
|                                                                                                                                    |
| O                                                                                                                                   |
| uj                                                                                                                                  |
| =1                                                                                                                                  |
| ∈S                                                                                                                                  |
| (cid:19)                                                                                                                            |
| (cid:18)                                                                                                                            |
| ,                                                                                                                                   |
| = Ω                                                                                                                                 |
| L2                                                                                                                                  |
| 1c2 d2 log (1/δ)                                                                                                                    |
| n2ε2                                                                                                                                |
| where the last step again comes from fingerprinting results. Note that the extra factor of d as compared to the previous            |
| √                                                                                                                                   |
| bound comes from the fact that we are considering fingerprinting codes with norm larger by a factor of                              |
| d. We also use the                                                                                                                  |
| ∥w−w′∥                                                                                                                              |
| √                                                                                                                                   |
| fact that the vector O(w, w′) transformed using u is (ε, δ)-DP by post processing. Now since c =                                    |
| we have                                                                                                                             |
| d                                                                                                                                   |
| (cid:32)                                                                                                                            |
| (cid:33)                                                                                                                            |
| (cid:112)d log (1/δ)                                                                                                                |
| E O                                                                                                                                 |
| [∥O(w, w′) − (∇L(w) − ∇L(w′))∥] =                                                                                                   |
| .                                                                                                                                   |
| L1 ∥w − w′∥                                                                                                                         |
| nε                                                                                                                                  |
| √                                                                                                                                   |
| (cid:104)                                                                                                                           |
| (cid:17)                                                                                                                            |
| (cid:104)                                                                                                                           |
| (cid:16) L1                                                                                                                         |
| d log(1/δ)                                                                                                                          |
| ∥O(w, w′) − (∇L(w) − ∇L(w′))∥2(cid:105)                                                                                             |
| ≤ E                                                                                                                                 |
| ∥O(w, w′)∥2(cid:105)                                                                                                                |
| Finally, noting that E                                                                                                              |
| . This                                                                                                                              |
| we obtain τ2 = Ω                                                                                                                    |
| nε                                                                                                                                  |
| O                                                                                                                                   |
| O                                                                                                                                   |
| completes the proof.                                                                                                                |
| We remark that the accuracy lower bound for the gradient variation can hold for a much more general set of vectors than that        |
| given in the proof. Specifically, the same result can be obtained for any u = w − w′ such that u has Θ(d) components which          |
| are Ω(cid:0) ∥u∥                                                                                                                    |
| (cid:1) (i.e. any sufficiently spread out vector). This uses the fact that it suffices to bound the number of components            |
| d                                                                                                                                   |
| which disagree in sign with the fingerprinting mean and that fingerprinting codes are sampled using a product distribution,         |
| and thus the tracing attack used by fingerprinting constructions holds over any sufficiently large subset of dimensions.            |
| C. Missing Results for Population Stationary Points                                                                                 |
| Here we present the proof of privacy and accuracy for Algorithm 1. We start by proving the privacy guarantee.                       |
| Proof of Theorem 3.1. By parallel composition of differential privacy, and since the used batches are disjoint, it suffices to      |
| prove that each step in lines 6 and 15 of the algorithm is (ε, δ)-DP. Note that the gradient estimator in step 6 has ℓ2-sensitivity |
| 2L0/b, so by the Gaussian mechanism this step is (ε, δ)-DP.                                                                         |
| ̸= x′                                                                                                                               |
| For step 15, suppose St,s and S′                                                                                                    |
| i∗ , and let ηt,si and                                                                                                              |
| η′                                                                                                                                  |
| the respective stepsizes used in step 23. Then                                                                                      |
| t,si                                                                                                                                |
| 2|s|                                                                                                                                |
| ∥∆t,s − ∆′                                                                                                                          |
| i∗ )) ∥ ,                                                                                                                           |
| t,s∥ =                                                                                                                              |
| ∥∇f (wt,s; xi∗ ) − ∇f (wt,(cid:98)s; xi∗ ) − (∇f (wt,s; x′                                                                          |
| i∗) − ∇f (wt,(cid:98)s; x′                                                                                                          |
| b                                                                                                                                   |
| iterates generated by the algorithm, which we denote as                                                                             |
| and note between the parent node ut,(cid:98)s and ut,s there are 2D−|s|                                                             |
| 20                                                                                                                                  |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

w t, � s   =  w t,s 0 , w t,s 1 , ..., w t,s 2 | D |− s   =  w t,s .   Then, by smoothness of  f   and the triangle inequality

∥ ∆ t,s  − ∆ ′ t,s ∥

=   2 | s |

b   ∥∇ f  ( w t,s ;  z i ∗ )  −∇ f  ( w t, � s ;  z i ∗ )  − ( ∇ f  ( w t,s ;  z ′ i ∗ )  −∇ f  ( w t, � s ;  z ′ i ∗ ))  ∥

≤

2 D −| s | �

i =1

2 | s |

b � ∥∇ f  ( w t,s i ;  z i ∗ )  −∇ f � w t,s i − 1 ;  z i ∗ � ∥ +  ∥ � ∇ f  ( w t,s i ;  z ′ i ∗ )  −∇ f � w t,s i − 1 ;  z ′ i ∗ �� ∥ �

≤

2 D −| s | �

i =1

2 | s |

b   L 1 η t,s i − 1 ∥∇ t,s i − 1 ∥ +

2 D −| s | �

i =1

2 | s |

b   L 1 η ′ t,s i − 1 ∥∇ ′ t,s i − 1 ∥

= 2

2 D −| s | �

i =1

2 | s |

b β 2 D/ 2   =   2 β 2 D/ 2

b .

The Gaussian mechanism combined with our choice of  σ t,s   certifies privacy of this step.

To prove Theorem  3.2  we will need some technical lemmas.   Define  ( T  ,  S )  as a random stopping time that indicates when Algorithm  1  ends.   Also, we say  ( t 1 , s 1 )  ⪯ 2   ( t 2 , s 2 )  whenever  w t 1 ,s 1   comes before  w t 2 ,s 2   in the algorithm iterates.

Lemma C.1  (Gradient estimation error, extension of Lemma  6  in ( Fang et al. ,  2018 )) .   Let  p  ∈ (0 ,  1) .   Then, with probability 1  − p  the event E   =  {∥∇ t,s  −∇ F ( w t,s ;  D ) ∥ 2   ≤ α  ·   ˜ α ∀ ( t, s )  ⪯ 2   ( T  ,  S ) }

holds, under the parameter setting of  σ t, ∅ , σ t,s  and  η t,s  in Algorithm  1 , for

α 2   ≥ � L 2 0 b   +   β 2 D 2 D

b

� max � 1 ,   ( d  + 1)

bε 2

� and ˜ α  ≥ 256 log � 1 . 25 δ

� log � 2 T 2 D +1

p

� α.

Proof.   Recall the gradient estimate associated to a left child node is the same as that of the parent node.   Hence, the gradient estimate of a non-leaf node is the same as that of the left-most leaf of its left sub-tree.   In addition, we only need to control the   gradient   estimation   error   when   we   perform   a   gradient   step,   which   occurs   at   the   leaves.   Then,   to   prove   the   claim, it suffices to prove that we can control the gradient estimation error at the leaves.   Since,   the number of iterations (and leaves) is at most  T 2 D − 1 , to prove event  E   happens with probability  1  − p , by the union bound it suffices to prove that P [ ∥∇ t,s  −∇ F ( w t,s ;  D ) ∥ 2   > α  ·   ˜ α ]  ≤ p T  2 D − 1   for every  ( t, s )  ⪯ 2   ( T  ,  S )  where  u t,s  is a leaf.

Denote by  F t   the sigma algebra generated by randomness in the algorithm until the end of round  t .   Fix  ( t, s )  ⪯ 2   ( T  ,  S ) such that  u t,s  is leaf, and let  u t,s ∅ =  u t,s 0 , u t,s 1 , ..., u t,s k   =  u t,s  be the path from the root to  s .   Next, extract a sub-sequence of it including only the root and the nodes that are right children, obtaining  u t,s ∅ =  u t,s a 0 , u t,s a 1 , ..., u t,s am   =  u t,s .   Now we can write

∇ t,s  −∇ F ( w t,s ;  D ) =

m �

i =0 g t,s ai   + �

x ∈ S t, ∅

1 b   ( ∇ f ( w t, ∅ ;  x )  −∇ F ( w t, ∅ ;  D )) � �� � γ 1 ,x

+

m �

i =1

�

x ∈ S t,sai

2 | s ai |

b

�� ∇ f ( w t,s ai ;  x ) −∇ f ( w t,s ai − 1 ;  x ) � − � ∇ F ( w t,s ai ;  D ) −∇ F ( w t,s ai − 1 ;  D ) ��

� �� � γ 2 ,x,i

.

To bound the estimation error, we note that

P [ ∥∇ t,s  −∇ F ( w t,s ;  D ) ∥ 2   > α  ·   ˜ α |F t − 1 ]

≤ P ����

m �

i =0 g t,s ai

��� 2 >   α  ·   ˜ α

4

��� F t − 1 � +  P ���� �

x ∈ S t, ∅ γ 1 ,x  +

m �

i =1

�

x ∈ S t,sai

γ 2 ,x,i ��� 2 >   α  ·   ˜ α

4

��� F t − 1 � .

21



| 0                                                                                                                                     |
|:--------------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                               |
| 2|D|−s = wt,s. Then, by smoothness of f and the triangle inequality                                                                   |
| wt,(cid:98)s = wt,s0 , wt,s1, ..., wt,s                                                                                               |
| ∥∆t,s − ∆′                                                                                                                            |
| t,s∥                                                                                                                                  |
| 2|s|                                                                                                                                  |
| =                                                                                                                                     |
| i∗ )) ∥                                                                                                                               |
| ∥∇f (wt,s; zi∗ ) − ∇f (wt,(cid:98)s; zi∗ ) − (∇f (wt,s; z′                                                                            |
| i∗) − ∇f (wt,(cid:98)s; z′                                                                                                            |
| b                                                                                                                                     |
| 2D−|s|                                                                                                                                |
| 2|s|                                                                                                                                  |
| (cid:88) i                                                                                                                            |
| ≤                                                                                                                                     |
| (cid:1)(cid:1) ∥(cid:3)                                                                                                               |
| (cid:2)∥∇f (wt,si; zi∗ ) − ∇f (cid:0)wt,si−1; zi∗                                                                                     |
| (cid:1) ∥ + ∥ (cid:0)∇f (wt,si ; z′                                                                                                   |
| i∗) − ∇f (cid:0)wt,si−1; z′                                                                                                           |
| i∗                                                                                                                                    |
| b                                                                                                                                     |
| =1                                                                                                                                    |
| 2D−|s|                                                                                                                                |
| 2D−|s|                                                                                                                                |
| 2|s|                                                                                                                                  |
| 2|s|                                                                                                                                  |
| (cid:88) i                                                                                                                            |
| (cid:88) i                                                                                                                            |
| ≤                                                                                                                                     |
| ∥∇′                                                                                                                                   |
| ∥                                                                                                                                     |
| L1η′                                                                                                                                  |
| L1ηt,si−1∥∇t,si−1∥ +                                                                                                                  |
| t,si−1                                                                                                                                |
| t,si−1                                                                                                                                |
| b                                                                                                                                     |
| b                                                                                                                                     |
| =1                                                                                                                                    |
| =1                                                                                                                                    |
| 2D−|s|                                                                                                                                |
| 2β2D/2                                                                                                                                |
| β                                                                                                                                     |
| 2|s|                                                                                                                                  |
| (cid:88) i                                                                                                                            |
| =                                                                                                                                     |
| = 2                                                                                                                                   |
| .                                                                                                                                     |
| b                                                                                                                                     |
| b                                                                                                                                     |
| 2D/2                                                                                                                                  |
| =1                                                                                                                                    |
| The Gaussian mechanism combined with our choice of σt,s certifies privacy of this step.                                               |
| To prove Theorem 3.2 we will need some technical lemmas. Define (T , S) as a random stopping time that indicates when                 |
| in the algorithm iterates.                                                                                                            |
| Algorithm 1 ends. Also, we say (t1, s1) ⪯2 (t2, s2) whenever wt1,s1 comes before wt2,s2                                               |
| Lemma C.1 (Gradient estimation error, extension of Lemma 6 in (Fang et al., 2018)). Let p ∈ (0, 1). Then, with probability            |
| 1 − p the event                                                                                                                       |
| E = {∥∇t,s − ∇F (wt,s; D)∥2 ≤ α · ˜α                                                                                                  |
| ∀(t, s) ⪯2 (T , S)}                                                                                                                   |
| holds, under the parameter setting of σt,∅, σt,s and ηt,s in Algorithm 1, for                                                         |
| (cid:19)                                                                                                                              |
| (cid:26)                                                                                                                              |
| (cid:27)                                                                                                                              |
| (cid:19)                                                                                                                              |
| (cid:19)                                                                                                                              |
| (cid:18) L2                                                                                                                           |
| β2D2D                                                                                                                                 |
| (d + 1)                                                                                                                               |
| (cid:18) 1.25                                                                                                                         |
| (cid:18) 2T 2D+1                                                                                                                      |
| 0                                                                                                                                     |
| and                                                                                                                                   |
| α2 ≥                                                                                                                                  |
| +                                                                                                                                     |
| max                                                                                                                                   |
| 1,                                                                                                                                    |
| α ≥ 256 log                                                                                                                           |
| log                                                                                                                                   |
| α.                                                                                                                                    |
| b                                                                                                                                     |
| b                                                                                                                                     |
| bε2                                                                                                                                   |
| δ                                                                                                                                     |
| p                                                                                                                                     |
| Proof. Recall the gradient estimate associated to a left child node is the same as that of the parent node. Hence, the gradient       |
| estimate of a non-leaf node is the same as that of the left-most leaf of its left sub-tree. In addition, we only need to control      |
| the gradient estimation error when we perform a gradient step, which occurs at                                                        |
| the leaves. Then,                                                                                                                     |
| to prove the claim,                                                                                                                   |
| it suffices to prove that we can control                                                                                              |
| the gradient estimation error at                                                                                                      |
| the leaves. Since,                                                                                                                    |
| the number of iterations (and                                                                                                         |
| leaves) is at most T 2D−1,                                                                                                            |
| to prove event E happens with probability 1 − p, by the union bound it suffices to prove that                                         |
| p                                                                                                                                     |
| P[∥∇t,s − ∇F (wt,s; D)∥2 > α · ˜α] ≤                                                                                                  |
| for every (t, s) ⪯2 (T , S) where ut,s is a leaf.                                                                                     |
| T 2D−1                                                                                                                                |
| Denote by Ft                                                                                                                          |
| the sigma algebra generated by randomness in the algorithm until the end of round t. Fix (t, s) ⪯2 (T , S)                            |
| such that ut,s is leaf, and let ut,s∅ = ut,s0 , ut,s1, ..., ut,sk = ut,s be the path from the root to s. Next, extract a sub-sequence |
| = ut,s. Now                                                                                                                           |
| of it including only the root and the nodes that are right children, obtaining ut,s∅ = ut,sa0                                         |
| , ut,sa1                                                                                                                              |
| we can write                                                                                                                          |
| (cid:88)                                                                                                                              |
| 1 b                                                                                                                                   |
| +                                                                                                                                     |
| m(cid:88) i                                                                                                                           |
| gt,sai                                                                                                                                |
| =0                                                                                                                                    |
| x∈St,∅                                                                                                                                |
| (cid:124)                                                                                                                             |
| (cid:123)(cid:122)                                                                                                                    |
| (cid:125)                                                                                                                             |
| γ1,x                                                                                                                                  |
| (cid:104)(cid:16)                                                                                                                     |
| (cid:17)                                                                                                                              |
| (cid:16)                                                                                                                              |
| (cid:17)(cid:105)                                                                                                                     |
| 2|sai |                                                                                                                               |
| (cid:88)                                                                                                                              |
| m(cid:88) i                                                                                                                           |
| ∇f (wt,sai                                                                                                                            |
| ∇F (wt,sai                                                                                                                            |
| ; x)−∇f (wt,sai−1                                                                                                                     |
| ; D)−∇F (wt,sai−1                                                                                                                     |
| b                                                                                                                                     |
| =1                                                                                                                                    |
| x∈St,sai                                                                                                                              |
| (cid:124)                                                                                                                             |
| (cid:123)(cid:122)                                                                                                                    |
| (cid:125)                                                                                                                             |
| γ2,x,i                                                                                                                                |
| To bound the estimation error, we note that                                                                                           |
| P[∥∇t,s − ∇F (wt,s; D)∥2 > α · ˜α|Ft−1]                                                                                               |
| 2                                                                                                                                     |
| 2                                                                                                                                     |
| (cid:104)(cid:13)                                                                                                                     |
| (cid:105)                                                                                                                             |
| (cid:105)                                                                                                                             |
| (cid:104)(cid:13)                                                                                                                     |
| α · ˜α                                                                                                                                |
| α · ˜α                                                                                                                                |
| (cid:88)                                                                                                                              |
| (cid:88)                                                                                                                              |
| (cid:13)(cid:13)(cid:13)                                                                                                              |
| (cid:12)(cid:12)(cid:12)                                                                                                              |
| (cid:13)(cid:13)(cid:13)                                                                                                              |
| (cid:12)(cid:12)(cid:12)                                                                                                              |
| ≤ P                                                                                                                                   |
| >                                                                                                                                     |
| + P                                                                                                                                   |
| >                                                                                                                                     |
| .                                                                                                                                     |
| (cid:13)(cid:13)                                                                                                                      |
| m(cid:88) i                                                                                                                           |
| (cid:13)(cid:13)                                                                                                                      |
| m(cid:88) i                                                                                                                           |
| γ1,x +                                                                                                                                |
| γ2,x,i                                                                                                                                |
| gt,sai                                                                                                                                |
| 4                                                                                                                                     |
| 4                                                                                                                                     |
| =0                                                                                                                                    |
| =1                                                                                                                                    |
| x∈St,sai                                                                                                                              |
| 21                                                                                                                                    |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

and proceed to bound each term on the right hand side separately.   By vector subgaussian concentration (see Lemma 1 in ( Jin et al. ,  2019 )) and noting that the gaussians are independent of  F t − 1 , we know that

P





�����

m �

i =0 g t,s ai

�����

2

>   α  ·   ˜ α

4



 ≤ 4 d   exp

�

− α  ·   ˜ α 32( σ 2 t, ∅ +  � m i =1   σ 2 t,s ai )

�

,

and in order to bound this probability by p 2 T  2 D − 1  , since  m  ≤ D , it suffices that

α  ·   ˜ α >  32 log � 4 d T 2 D

p

�� 8 L 2 0   log (1 . 25 /δ )

b 2 ε 2 +   8 D 2 D β 2  log (1 . 25 /δ )

b 2 ε 2

�

= 256 log � 1 . 25 δ

�� d  log (4) + log � T 2 D

p

��� L 2 0 b 2 ε 2   +   D 2 D β 2

b 2 ε 2

� .

Now, noting that surely

∥ γ 1 ,x ∥≤ 2 L 0

b and ∥ γ 2 ,x,i ∥≤ 2 β 2 D/ 2

b ,

where   the   second   bound   comes   from   following   similar   steps   as   in   the   privacy   analysis   in   Theorem   3.1 ,   we   have   that �

x ∈ S t, ∅ γ 1 ,x  +  � m i =1 � x ∈ S t,sai   γ 2 ,x,i   is a sum of bounded martingale differences when conditioned on  F t − 1 , thus by concentration of martingale-difference sequences in  ℓ 2   (see Proposition 2 in ( Fang et al. ,  2018 )), and using the fact that | S t, ∅ |  =  b  and  | S t,s ai |  =  b/ 2 | s ai |   it follows that

P





������

�

x ∈ S t, ∅ γ 1 ,x  +

m �

i =1

�

x ∈ S t,sai

γ 2 ,x,i

������

2

>   α  ·   ˜ α

4 | F t − 1



 ≤ 4 exp



 − α  ·   ˜ α

16 � 4 L 2 0 b +   � m i =1 4 β 2 2 D

2 | sai  | b

�



 .

Repeating a similar argument as before, to bound this term by p 2 T  2 D − 1  , it suffices that

α  ·   ˜ α  ≥ 64 log � 2 T 2 D +1

p

�� L 2 0 b   +   β 2 D 2 D

b

� .

Finally, both conditions hold simultaneously for

α 2   ≥ � L 2 0 b   +   β 2 D 2 D

b

� max � 1 ,   ( d  + 1)

bε 2

�

and

˜ α  ≥ 256 log � 1 . 25 δ

� log � 2 T 2 D +1

p

� α.

Lemma C.2  (Descent lemma; Lemma  7  in ( Fang et al. ,  2018 )) .   Under the assumption that the event  E   from Lemma  C.1 occurs and  β   ≤ 2 D/ 2 ˜ α , we have that if Algorithm  1  reaches the last line, then

F ( w T,ℓ (2 D ) ;  D )  − F (0;  D )  ≤− ( T 2 D − 1 ) β  ·   ˜ α 4  ·  2 D/ 2 L 1 .

where  w T,ℓ (2 D )   is the last iterate in the  T -th tree of Algorithm  1 .

We provide the proof of Lemma  C.2  adapted to our case for completeness.

22



| 0                                                                                       | 1                                                                                                                          | 2        | 3     | 4         | 5   | 6       |
|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:---------|:------|:----------|:----|:--------|
|                                                                                         | Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                    |          |       |           |     |         |
|                                                                                         | and proceed to bound each term on the right hand side separately. By vector subgaussian concentration (see Lemma 1 in      |          |       |           |     |         |
| (Jin et al., 2019)) and noting that the gaussians are independent of Ft−1, we know that |                                                                                                                            |          |       |           |     |         |
| 2                                                                                       | (cid:33)                                                                                                                   |          |       |           |     |         |
| (cid:32)                                                                                |                                                                                                                            |          |       |           |     |         |
|                                                                                       |                                                                                                                            |          |       |           |     |         |
|                                                                                       |                                                                                                                            |          |       |           |     |         |
| α · ˜α                                                                                  |                                                                                                                            |          |       |           |     |         |
| α · ˜α                                                                                  |                                                                                                                            |          |       |           |     |         |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                                                |                                                                                                                            |          |       |           |     |         |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                                                |                                                                                                                            |          |       |           |     |         |
| P                                                                                       |                                                                                                                            |          |       |           |     |         |
| >                                                                                       |                                                                                                                            |          |       |           |     |         |
| ≤ 4d exp                                                                                |                                                                                                                            |          |       |           |     |         |
| m(cid:88) i                                                                             |                                                                                                                            |          |       |           |     |         |
| gt,sai                                                                                  |                                                                                                                            |          |       |           |     |         |
| 4                                                                                       | )                                                                                                                          |          |       |           |     |         |
| 32(σ2                                                                                   |                                                                                                                            |          |       |           |     |         |
| t,∅ + (cid:80)m                                                                         |                                                                                                                            |          |       |           |     |         |
| i=1 σ2                                                                                  |                                                                                                                            |          |       |           |     |         |
| t,sai                                                                                   |                                                                                                                            |          |       |           |     |         |
| =0                                                                                      |                                                                                                                            |          |       |           |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
| and in order to bound this probability by                                               |                                                                                                                            |          |       |           |     |         |
| 2T 2D−1 , since m ≤ D, it suffices that                                                 |                                                                                                                            |          |       |           |     |         |
| (cid:18) 4dT 2D                                                                         | 8D2Dβ2 log (1.25/δ)                                                                                                        |          |       |           |     |         |
| (cid:19) (cid:20) 8L2                                                                   |                                                                                                                            |          |       |           |     |         |
| 0 log (1.25/δ)                                                                          |                                                                                                                            |          |       |           |     |         |
| +                                                                                       |                                                                                                                            |          |       |           |     |         |
| α · ˜α > 32 log                                                                         |                                                                                                                            |          |       |           |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
| b2ε2                                                                                    |                                                                                                                            |          |       |           |     |         |
| b2ε2                                                                                    |                                                                                                                            |          |       |           |     |         |
|                                                                                         |                                                                                                                            | (cid:21) |       |           |     |         |
| (cid:18) 1.25                                                                           | D2Dβ2                                                                                                                      |          |       |           |     |         |
| (cid:18) T 2D                                                                           |                                                                                                                            |          |       |           |     |         |
| (cid:19)(cid:21) (cid:20) L2                                                            |                                                                                                                            |          |       |           |     |         |
| 0                                                                                       |                                                                                                                            |          |       |           |     |         |
| = 256 log                                                                               |                                                                                                                            | .        |       |           |     |         |
| δ                                                                                       | b2ε2                                                                                                                       |          |       |           |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
| b2ε2 +                                                                                  |                                                                                                                            |          |       |           |     |         |
| Now, noting that surely                                                                 |                                                                                                                            |          |       |           |     |         |
| 2β2D/2                                                                                  |                                                                                                                            |          |       |           |     |         |
| 2L0                                                                                     |                                                                                                                            |          |       |           |     |         |
| ,                                                                                       |                                                                                                                            |          |       |           |     |         |
| and                                                                                     |                                                                                                                            |          |       |           |     |         |
| ∥γ1,x∥ ≤                                                                                |                                                                                                                            |          |       |           |     |         |
| ∥γ2,x,i∥ ≤                                                                              |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
|                                                                                         | where the second bound comes from following similar steps as in the privacy analysis in Theorem 3.1, we have that          |          |       |           |     |         |
| (cid:80)                                                                                |                                                                                                                            |          |       |           |     |         |
| (cid:80)                                                                                |                                                                                                                            |          |       |           |     |         |
| γ2,x,i                                                                                  | is a sum of bounded martingale differences when conditioned on Ft−1,                                                       |          |       |           |     | thus by |
| i=1                                                                                     |                                                                                                                            |          |       |           |     |         |
| x∈St,∅ γ1,x + (cid:80)m                                                                 |                                                                                                                            |          |       |           |     |         |
| x∈St,sai                                                                                |                                                                                                                            |          |       |           |     |         |
|                                                                                         | concentration of martingale-difference sequences in ℓ2 (see Proposition 2 in (Fang et al., 2018)), and using the fact that |          |       |           |     |         |
| | = b/2|sai |                                                                           |                                                                                                                            |          |       |           |     |         |
| it follows that                                                                         |                                                                                                                            |          |       |           |     |         |
| |St,∅| = b and |St,sai                                                                  |                                                                                                                            |          |       |           |     |         |
| 2                                                                                       |                                                                                                                            |          |       |           |     |         |
|                                                                                      | α · ˜α                                                                                                                     |          |       |           |   |         |
|                                                                                      |                                                                                                                            |          |       |           |     |         |
|                                                                                       |                                                                                                                            |          |       |           |     |         |
| α · ˜α                                                                                  |                                                                                                                            |          |       |           |     |         |
| (cid:88)                                                                                |                                                                                                                            |          |       |           |     |         |
| (cid:88)                                                                                |                                                                                                                            |          |       |           |     |         |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                                        |                                                                                                                            |          |       |           |     |         |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                                        |                                                                                                                            |          |       |           |     |         |
| P                                                                                       |                                                                                                                            |          |       |           |     |         |
| m(cid:88) i                                                                             |                                                                                                                            |          |       |           |     | .       |
| γ2,x,i                                                                                  |                                                                                                                            |          |       |           |     |         |
| γ1,x +                                                                                  |                                                                                                                            |          |       |           |     |         |
| | Ft−1                                                                                  |                                                                                                                            |          |       |           |     |         |
| (cid:104) 4L2                                                                           |                                                                                                                            |          |       | (cid:105) |     |         |
| 4                                                                                       |                                                                                                                            |          | 4β22D |           |     |         |
| 0                                                                                       | + (cid:80)m                                                                                                                |          |       |           |     |         |
| 16                                                                                      |                                                                                                                            |          |       |           |     |         |
| =1                                                                                      |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            | i=1      |       |           |     |         |
| x∈St,sai                                                                                |                                                                                                                            |          |       |           |     |         |
|                                                                                         |                                                                                                                            |          | 2|sai | |b        |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
| Repeating a similar argument as before, to bound this term by                           |                                                                                                                            |          |       |           |     |         |
| 2T 2D−1 , it suffices that                                                              |                                                                                                                            |          |       |           |     |         |
| (cid:18) 2T 2D+1                                                                        |                                                                                                                            |          |       |           |     |         |
| (cid:19) (cid:20) L2                                                                    |                                                                                                                            |          |       |           |     |         |
| β2D2D                                                                                   |                                                                                                                            |          |       |           |     |         |
| 0                                                                                       |                                                                                                                            |          |       |           |     |         |
| α · ˜α ≥ 64 log                                                                         |                                                                                                                            |          |       |           |     |         |
| +                                                                                       |                                                                                                                            |          |       |           |     |         |
| .                                                                                       |                                                                                                                            |          |       |           |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
| Finally, both conditions hold simultaneously for                                        |                                                                                                                            |          |       |           |     |         |
| β2D2D                                                                                   |                                                                                                                            |          |       |           |     |         |
| (cid:18) L2                                                                             |                                                                                                                            |          |       |           |     |         |
| (d + 1)                                                                                 |                                                                                                                            |          |       |           |     |         |
| 0                                                                                       |                                                                                                                            |          |       |           |     |         |
| max                                                                                     |                                                                                                                            |          |       |           |     |         |
| 1,                                                                                      |                                                                                                                            |          |       |           |     |         |
| +                                                                                       |                                                                                                                            |          |       |           |     |         |
| α2 ≥                                                                                    |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
| b                                                                                       |                                                                                                                            |          |       |           |     |         |
| bε2                                                                                     |                                                                                                                            |          |       |           |     |         |
| and                                                                                     |                                                                                                                            |          |       |           |     |         |
| (cid:18) 2T 2D+1                                                                        |                                                                                                                            |          |       |           |     |         |
| (cid:18) 1.25                                                                           |                                                                                                                            |          |       |           |     |         |
| log                                                                                     |                                                                                                                            |          |       |           |     |         |
| α.                                                                                      |                                                                                                                            |          |       |           |     |         |
| α ≥ 256 log                                                                             |                                                                                                                            |          |       |           |     |         |
| δ                                                                                       |                                                                                                                            |          |       |           |     |         |
| p                                                                                       |                                                                                                                            |          |       |           |     |         |
|                                                                                         | Lemma C.2 (Descent lemma; Lemma 7 in (Fang et al., 2018)). Under the assumption that the event E from Lemma C.1            |          |       |           |     |         |
| occurs and β ≤ 2D/2 ˜α, we have that if Algorithm 1 reaches the last line, then         |                                                                                                                            |          |       |           |     |         |
| β · ˜α                                                                                  |                                                                                                                            |          |       |           |     |         |
| .                                                                                       |                                                                                                                            |          |       |           |     |         |
| F (wT,ℓ(2D); D) − F (0; D) ≤ −(T 2D−1)                                                  |                                                                                                                            |          |       |           |     |         |
| 4 · 2D/2L1                                                                              |                                                                                                                            |          |       |           |     |         |
| where wT,ℓ(2D) is the last iterate in the T -th tree of Algorithm 1.                    |                                                                                                                            |          |       |           |     |         |
| We provide the proof of Lemma C.2 adapted to our case for completeness.                 |                                                                                                                            |          |       |           |     |         |
| 22                                                                                      |                                                                                                                            |          |       |           |     |         |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Proof.   By standard analysis for smooth functions we have

F ( w t,s + ;  D )  ≤ F ( w t,s ;  D )  − η t,s

2   (1  − η t,s L 1 ) ∥∇ t,s ∥ 2  +   η t,s

2   ∥∇ t,s  −∇ F ( w t,s ;  D ) ∥ 2 ,

where  η t,s   = β 2 D/ 2 L 1 ∥∇ t,s ∥ and  u t,s +   is the node after  u t,s   in the tree.   Since  β   ≤ 2 D/ 2 ˜ α  and  ∥∇ t,s ∥ >   2˜ α , we have that (1  − η t,s L 1 )  ≥ 1 / 2 .   Using this inequality, the definition of  η t,s  and the fact that we are assuming  E   occurs, we obtain

F ( w t,s + ;  D )  − F ( w t,s ;  D )  ≤− β 4  ·  2 D/ 2 L 1 ∥∇ t,s ∥ ∥∇ t,s ∥ 2  + β 2  ·  2 D/ 2 L 1 ∥∇ t,s ∥ α  ·   ˜ α

≤− β 4  ·  2 D/ 2 L 1 ·   ˜ α,

where the second inequality comes from  ∥∇ t,s ∥ >  2˜ α  and  α  ≤ ˜ α .   Then telescoping over all  T 2 D − 1   iterations provides the claimed bound.

We are now ready to prove the convergence guarantee of Algorithm  1 .

Proof of Theorem  3.2 .   From Lemma  C.1 , we know that  ∥∇ t,s  −∇ F ( w t,s ;  D ) ∥ 2   ≤ α  ·   ˜ α  with probability  1  − p  when

α  = √

2 L 0  max � 1 n 1 / 3  , � √ d nε � 1 / 2 � ,  ˜ α  = � 256 log � 1 . 25 δ � log � 2 T  2 D +1

p � +   8 L 1 F 0 √

2 D ( D/ 2+1) 2 L 2 0

� α.

Indeed, using our parameter setting, and noting that  d > bε 2   if and only if,  d > n 2 / 3 ε 2 , yields

α 2   ≥ L 2 0 b   max � 1 ,   ( d  + 1)

bε 2

� +   β 2

2   max � 1 ,   ( d  + 1)

bε 2

�

=  L 2 0

� 1 n 2 / 3  1 { d +1 ≤ n 2 / 3 ε 2 }  +

√

d nε   1 { d +1 >n 2 / 3 ε 2 }

�

+   α 2

2   min � 1 ,  bε 2

d

� max � 1 ,   ( d  + 1)

bε 2

�

≥ L 2 0   max

� 1 n 2 / 3  ,

√

d nε

�

+   α 2

2   ,

which shows our values of  α  and   ˜ α  are valid for controlling the gradient estimation error with high probability, as claimed in Lemma  C.1 .

Now, suppose for the sake of contradiction that Algorithm  1  does not end in line  20  under  E .   This means it performs  T 2 D − 1

gradient updates.   We’ll show this implies  ( T 2 D − 1 ) β · ˜ α 4 · 2 D/ 2 L 1   >   F 0   and thus contradicts Lemma  C.2 , which claims that

F 0   ≥− [ F ( w T,ℓ (2 D ) ;  D )  − F ( w 0 ,ℓ (2 D ) ;  D )]  ≥ ( T 2 D − 1 ) β · ˜ α 4 · 2 D/ 2 L 1   .   Indeed, note that by our parameter setting:

( T 2 D − 1 ) β  ·   ˜ α 4  ·  2 D/ 2 L 1 > F 0   ⇐⇒ β  ·   ˜ α >   8 L 1 F 0

T 2 D/ 2

⇐⇒ α  min

�

1 ,

√

bε √

d

�

·   ˜ α >   8 L 1 F 0 √

2 D T √

b

⇐⇒ α  ·   ˜ α >   8 L 1 F 0 √

2 D ( D/ 2 + 1) √

b n max

�

1 ,

√

d √

bε

�

⇐⇒ α  ·   ˜ α >  8 L 1 F 0 √

2 D ( D/ 2 + 1) max

� √ b n   ,

√

d nε

�

,

23



| 0                                                                                                                               |
|:--------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                         |
| Proof. By standard analysis for smooth functions we have                                                                        |
| ηt,s                                                                                                                            |
| ηt,s                                                                                                                            |
| (1 − ηt,sL1)∥∇t,s∥2 +                                                                                                           |
| ∥∇t,s − ∇F (wt,s; D)∥2,                                                                                                         |
| F (wt,s+; D) ≤ F (wt,s; D) −                                                                                                    |
| 2                                                                                                                               |
| 2                                                                                                                               |
| where ηt,s =                                                                                                                    |
| 2D/2L1∥∇t,s∥ and ut,s+ is the node after ut,s in the tree. Since β ≤ 2D/2 ˜α and ∥∇t,s∥ > 2˜α, we have that                     |
| (1 − ηt,sL1) ≥ 1/2. Using this inequality, the definition of ηt,s and the fact that we are assuming E occurs, we obtain         |
| β                                                                                                                               |
| β                                                                                                                               |
| α · ˜α                                                                                                                          |
| ∥∇t,s∥2 +                                                                                                                       |
| F (wt,s+; D) − F (wt,s; D) ≤ −                                                                                                  |
| 4 · 2D/2L1∥∇t,s∥                                                                                                                |
| 2 · 2D/2L1∥∇t,s∥                                                                                                                |
| β                                                                                                                               |
| ≤ −                                                                                                                             |
| · ˜α,                                                                                                                           |
| 4 · 2D/2L1                                                                                                                      |
| where the second inequality comes from ∥∇t,s∥ > 2˜α and α ≤ ˜α. Then telescoping over all T 2D−1 iterations provides the        |
| claimed bound.                                                                                                                  |
| We are now ready to prove the convergence guarantee of Algorithm 1.                                                             |
| Proof of Theorem 3.2. From Lemma C.1, we know that ∥∇t,s − ∇F (wt,s; D)∥2 ≤ α · ˜α with probability 1 − p when                  |
| √                                                                                                                               |
| √                                                                                                                               |
| (cid:16) √                                                                                                                      |
| (cid:17)1/2(cid:27)                                                                                                             |
| (cid:16) 2T 2D+1                                                                                                                |
| 2D(D/2+1)                                                                                                                       |
| 1                                                                                                                               |
| d                                                                                                                               |
| α =                                                                                                                             |
| (cid:1) log                                                                                                                     |
| + 8L1F0                                                                                                                         |
| α.                                                                                                                              |
| , ˜α =                                                                                                                          |
| 256 log (cid:0) 1.25                                                                                                            |
| nε                                                                                                                              |
| δ                                                                                                                               |
| p                                                                                                                               |
| 2L2                                                                                                                             |
| n1/3 ,                                                                                                                          |
| 0                                                                                                                               |
| Indeed, using our parameter setting, and noting that d > bε2 if and only if, d > n2/3ε2, yields                                 |
| (cid:26)                                                                                                                        |
| (cid:27)                                                                                                                        |
| (cid:26)                                                                                                                        |
| (cid:27)                                                                                                                        |
| L2                                                                                                                              |
| (d + 1)                                                                                                                         |
| β2                                                                                                                              |
| (d + 1)                                                                                                                         |
| 0                                                                                                                               |
| α2 ≥                                                                                                                            |
| max                                                                                                                             |
| 1,                                                                                                                              |
| +                                                                                                                               |
| max                                                                                                                             |
| 1,                                                                                                                              |
| b                                                                                                                               |
| bε2                                                                                                                             |
| 2                                                                                                                               |
| bε2                                                                                                                             |
| √                                                                                                                               |
| (cid:32)                                                                                                                        |
| (cid:33)                                                                                                                        |
| (cid:26)                                                                                                                        |
| (cid:27)                                                                                                                        |
| (cid:26)                                                                                                                        |
| (cid:27)                                                                                                                        |
| 1                                                                                                                               |
| d                                                                                                                               |
| α2                                                                                                                              |
| bε2                                                                                                                             |
| (d + 1)                                                                                                                         |
| 1                                                                                                                               |
| 1                                                                                                                               |
| = L2                                                                                                                            |
| +                                                                                                                               |
| min                                                                                                                             |
| 1,                                                                                                                              |
| max                                                                                                                             |
| 1,                                                                                                                              |
| {d+1≤n2/3ε2} +                                                                                                                  |
| {d+1>n2/3ε2}                                                                                                                    |
| 0                                                                                                                               |
| nε                                                                                                                              |
| 2                                                                                                                               |
| d                                                                                                                               |
| bε2                                                                                                                             |
| n2/3                                                                                                                            |
| √                                                                                                                               |
| (cid:41)                                                                                                                        |
| (cid:40)                                                                                                                        |
| d                                                                                                                               |
| α2                                                                                                                              |
| 1                                                                                                                               |
| ,                                                                                                                               |
| +                                                                                                                               |
| ,                                                                                                                               |
| ≥ L2                                                                                                                            |
| 0 max                                                                                                                           |
| nε                                                                                                                              |
| 2                                                                                                                               |
| n2/3                                                                                                                            |
| which shows our values of α and ˜α are valid for controlling the gradient estimation error with high probability, as claimed in |
| Lemma C.1.                                                                                                                      |
| Now, suppose for the sake of contradiction that Algorithm 1 does not end in line 20 under E. This means it performs T 2D−1      |
| β· ˜α                                                                                                                           |
| gradient updates. We’ll show this implies (T 2D−1)                                                                              |
| > F0 and thus contradicts Lemma C.2, which claims that                                                                          |
| 4·2D/2L1                                                                                                                        |
| β· ˜α                                                                                                                           |
| . Indeed, note that by our parameter setting:                                                                                   |
| F0 ≥ −[F (wT,ℓ(2D); D) − F (w0,ℓ(2D); D)] ≥ (T 2D−1)                                                                            |
| 4·2D/2L1                                                                                                                        |
| β · ˜α                                                                                                                          |
| 8L1F0                                                                                                                           |
| (T 2D−1)                                                                                                                        |
| > F0 ⇐⇒ β · ˜α >                                                                                                                |
| T 2D/2                                                                                                                          |
| 4 · 2D/2L1                                                                                                                      |
| √                                                                                                                               |
| √                                                                                                                               |
| (cid:41)                                                                                                                        |
| (cid:40)                                                                                                                        |
| bε                                                                                                                              |
| 2D                                                                                                                              |
| 8L1F0                                                                                                                           |
| √                                                                                                                               |
| √                                                                                                                               |
| · ˜α >                                                                                                                          |
| ⇐⇒ α min                                                                                                                        |
| 1,                                                                                                                              |
| d                                                                                                                               |
| T                                                                                                                               |
| b                                                                                                                               |
| √                                                                                                                               |
| √                                                                                                                               |
| √                                                                                                                               |
| (cid:40)                                                                                                                        |
| (cid:41)                                                                                                                        |
| 2D(D/2 + 1)                                                                                                                     |
| b                                                                                                                               |
| d                                                                                                                               |
| 8L1F0                                                                                                                           |
| √                                                                                                                               |
| 1,                                                                                                                              |
| ⇐⇒ α · ˜α >                                                                                                                     |
| max                                                                                                                             |
| n                                                                                                                               |
| bε                                                                                                                              |
| (cid:40) √                                                                                                                      |
| (cid:41)                                                                                                                        |
| √                                                                                                                               |
| b                                                                                                                               |
| d                                                                                                                               |
| 2D(D/2 + 1) max                                                                                                                 |
| ,                                                                                                                               |
| ,                                                                                                                               |
| ⇐⇒ α · ˜α > 8L1F0                                                                                                               |
| n                                                                                                                               |
| nε                                                                                                                              |
| 23                                                                                                                              |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

and noting that by the setting of  b  we have  max � √ b n   , √

d nε � = max � 1 n 2 / 3  , √

d nε � , we conclude the following

( T 2 D − 1 ) β  ·   ˜ α 4  ·  2 D/ 2 L 1 > F 0   ⇐⇒ α  ·   ˜ α >  8 L 1 F 0 √

2 D ( D/ 2 + 1) max

� 1 n 2 / 3  ,

√

d nε

�

⇐⇒ α  ·   ˜ α >   8 L 1 F 0 √

2 D ( D/ 2 + 1)

2 L 2 0 α 2 .

Finally, note  α  ·   ˜ α  = � 256 log (1 . 25 /δ ) log � 2 T 2 D +1 /p � +   8 L 1 F 0 √

2 D ( D/ 2+1) 2 L 2 0

� α 2   and thus the last inequality holds under

our parameter setting.   Since this is equivalent to  ( T 2 D − 1 ) β · ˜ α 4 · 2 D/ 2 L 1   > F 0 , we are done with the contradiction.   It follows that with high probability, Algorithm  1  ends in line  20  returning  w t,s   such that  ∥∇ t,s ∥≤ 2˜ α .   Also, by Lemma  C.1  we have ∥∇ F ( w t,s ;  D )  −∇ t,s ∥ <   ˜ α , so the returned iterate satisfies by the triangle inequality

∥∇ F ( w t,s ;  D ) ∥ <  3˜ α.

In addition, the linear time oracle complexity follows from the fact that at each binary tree we use  b  samples at the root, and then  b/ 2  in levels  1  to  D .   This gives a total of  b ( D/ 2 + 1)  samples used at every round.   Since we run the algorithm for  T   = n b ( D/ 2+1)   rounds, we compute exactly  n  gradients. To conclude, note the condition  n  ≥ max { √

d ( D/ 2+1) 2 /ε,  ( D/ 2+1) 3 } implies the number of rounds  T   is at least  1 .   Besides, since the definition of  D  implies  2 D   < b , the size of the mini-batches are well-defined (meaning Algorithm  1  uses batches with at least  1  sample).   This concludes the proof.

D. Missing Results for Stationary Points in the Convex Setting

We first give pseudo-codes of algorithms used in the section.

Algorithm 5  Phased SGD ( S,  ( w, x )  �→ f ( w ;  x )) , R, η,  S ( · ) , σ ) Input:   Dataset  S , loss function  f ( · ;  x )) , radius  R  of the constraint set  W , steps  T ,  η , Selection function  S , Noise variance σ 1:   w 1   = 0 2:   K   =  ⌈ log ( | S | ) ⌉ and  T 0   = 1 3:   for  k   = 1  to  K  − 1  do 4: T k   = 2 − k   | S |  , η k   = 4 − k η, σ k   =  η k σ 5: w k +1   =  OutputPerturbedSGD ( w k , S T k − 1 +1: T k , R, η k , σ k ,  S ( · )) 6:   end for Output:   ¯ w   =  w K

Algorithm 6  OutputPerturbedSGD ( w 1 , S,  ( w, x )  �→ f ( w ;  x ) ,  ∆( · ) , R, η,  S ( · ) Input:   Dataset  S , loss function  f ( · ;  x )) , regularizer  ∆( · ) , radius  R  of the constraint set  W , steps  T ,  η , Selection function S , Noise variance  σ 1:   for  t  = 1  to  | S | − 1  do 2: w t +1   = Π W  ( w t  − η  ( ∇ f ( w t ;  x t ))) 3:   end for 4:   ξ   ∼N (0 , σ 2 I )

5:   ˜ w   =  S � { w t } | S | t =1 �

Output:   ¯ w   =   ˜ w  +  ξ

Proof of Theorem  5.1 .   The   privacy   guarantee,   in   both   cases,   follows   from   the   privacy   guarantees   of   Algorithm   7   and Algorithm  5 , in Lemmas  D.3  and  D.6  respectively, together with parallel composition.

24



| 0                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------|
| Algorithm 5 Phased SGD(S, (w, x) (cid:55)→ f (w; x)), R, η, S(·), σ)                                                           |
| Input: Dataset S, loss function f (·; x)), radius R of the constraint set W, steps T , η, Selection function S, Noise variance |
| σ                                                                                                                              |
| 1: w1 = 0                                                                                                                      |
| 2: K = ⌈log (|S|)⌉ and T0 = 1                                                                                                  |
| 3:                                                                                                                             |
| for k = 1 to K − 1 do                                                                                                          |
| 4:                                                                                                                             |
| Tk = 2−k |S| , ηk = 4−kη, σk = ηkσ                                                                                             |
| 5:                                                                                                                             |
| wk+1 = OutputPerturbedSGD(wk, STk−1+1:Tk , R, ηk, σk, S(·))                                                                    |
| 6:                                                                                                                             |
| end for                                                                                                                        |
| Output:                                                                                                                        |
| w = wK                                                                                                                         |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Algorithm 7  Noisy GD ( S,  ( w, x )  �→ f ( w ;  x )) , R, T, η,  S ( · ) , σ ) Input:   Dataset  S , loss function  ( w, x )  �→ f ( w ;  x ) , radius  R  of the constraint set  W , steps  T ,  η , Selection function  S , Noise variance  σ 1:   w 1   = 0 2:   for  t  = 1  to  T   − 1  do 3: ξ t   ∼N (0 , σ 2 I ) 4: w t +1   = Π W  ( w t  − η  ( ∇ F ( w t ;  S ) +  ξ t )) 5:   end for Output:   ¯ w   =  S � { w t } T t =1 �

We now proceed to the utility part.   For simplicity of notation, let  R  =  ∥ w ∗ ∥ .   Recall the definition of the regularized losses f   ( t ) ( w, x )  in Algorithm  3 .   Let  { α t } t   be such that  E [ F   ( t − 1) ( ¯ w t ;  D )]  − F   ( t − 1) ( w ∗ t − 1 ;  D )   ≤ α t   where   ¯ w t   are the iterates produced in the algorithm and  w ∗ t − 1   = arg min w ∈ R d  F  ( t − 1) ( w ;  D ) .   Following ( Allen-Zhu ,  2018 ;  Foster et al. ,  2019 ), we first establish a general result which will be useful for both parts of the result.

E  ∥∇ F ( ¯ w T  ;  D ) ∥ =  E

����� ∇ F  ( T  − 1) ( ¯ w T  ;  D ) +  λ

T �

t =0 2 t   ( ¯ w t  − ¯ w T  )

�����

≤ E ��� ∇ F  ( T  − 1) ( ¯ w T  ;  D ) ��� +  λ

T  − 1 �

t =0 2 t E ��� ¯ w t  − w ∗ T  − 1 �� + �� ¯ w T   − w ∗ T  − 1 ���

≤ 2 E ��� ∇ F  ( T  − 1) ( ¯ w T  ;  D ) ��� +  λ

T  − 1 �

t =1 2 t E �� ¯ w t  − w ∗ T  − 1 �� +  λ E �� w 0  − w ∗ T  − 1 ��

≤ 2 E ��� ∇ F  ( T  − 1) ( ¯ w T  ;  D ) ��� + 4

T  − 1 �

t =1

� λ 2 t α t  +  λR T  − 1

≤ 4 � L 1 α T   + 4

T  − 1 �

t =1

� λ 2 t +1 α t  +  λ 2 T/ 2 R

≤ 4

T �

t =1

� λ 2 t +1 α t  + � λL 1 R

where the third and fourth inequality follows from strong convexity of  F   ( T  − 1) ( · ;  D )  and Lemma  D.2  respectively.   The last inequality follows from the setting of  T   since we have that  F   ( T  − 1)   is  L 1  +   � T  − 1 t =1   2 t λ   ≤ L 1  +  λ 2 T   ≤ 2 L 1   smooth. Note that the definition of  R t   and Lemma  D.1 , �� w ∗ T  − 1 �� ≤ R T  − 1 , so the unconstrained minimizer lies in the constraint set. Therefore  E �� ∇ F  ( T  − 1) ( ¯ w T  ;  D ) �� =  E �� ∇ F  ( T  − 1) ( ¯ w T  ;  D )  −∇ F  ( T  − 1) ( w ∗ T  − 1 ;  D ) �� ≤ 2 √ L 1 α T  .

Observe that from the setting of  T ,  F   ( T  )   is  4 L 1   smooth for all  t .   Furthermore, the radius of the constraint set in the  t -th round is  R t   =   2 T/ 2 R .   Hence,   the Lipschitz constant  G t   ≤ L 0   + 8 L 1 R t   ≤ O � L 0  +  L 1 2 T/ 2 � .   Now we instantiate  α t , which is the excess population risk bound of the DP-SCO sub-routine.

Optimal rate: The excess population risk guarantee of Algorithm  7  is in Lemma  D.3 , with (in context of the notation in the Lemma) Lipschitz parameter  L 0   being the same and  G ∆ =  O � L 1 2 T/ 2 � .   Therefore, we have  α t   =   ˜ O � G 2 λ t n   + dG 2 λ t n 2 ε 2 � . Plugging in the above estimate, we get,

E  ∥∇ F ( ¯ w ;  D ) ∥ =   ˜ O

� G √ n   +

√

dG nε +

� λ L 1 R

�

=   ˜ O

� G √ n   +

√

dG nε

�

where the last step follows by setting of  λ .

The optimality claim follows by combining the non-private lower bound in Theorem  5.1 , and the DP empirical stationarity lower bound in Theorem  4.3  together with a reduction to population stationarity as in ( Bassily et al. ,  2019 , Appendix C).

25



| 0                                                                     | 1                                    |
|:----------------------------------------------------------------------|:-------------------------------------|
| Algorithm 7 Noisy GD(S, (w, x) (cid:55)→ f (w; x)), R, T, η, S(·), σ) |                                      |
|                                                                       | variance σ                           |
| 1: w1 = 0                                                             |                                      |
| 2:                                                                    | for t = 1 to T − 1 do                |
| 3:                                                                    | ξt ∼ N (0, σ2I)                      |
| 4:                                                                    | wt+1 = ΠW (wt − η (∇F (wt; S) + ξt)) |
| 5:                                                                    | end for                              |
|                                                                       | (cid:17)                             |
|                                                                       | (cid:16)                             |
| Output:                                                               | w = S                                |
|                                                                       | {wt}T                                |
|                                                                       | t=1                                  |




| 0                                        | 1                                        | 2                 | 3                | 4                 | 5                |
|:-----------------------------------------|:-----------------------------------------|:------------------|:-----------------|:------------------|:-----------------|
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | T(cid:88) t                              |                   |                  |                   |                  |
| ∇F (T −1)( ¯wT ; D) + λ                  | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                   |                  |                   |                  |
|                                          | 2t ( ¯wt − ¯wT )                         |                   |                  |                   |                  |
|                                          | =0                                       |                   |                  |                   |                  |
|                                          | T −1                                     |                   |                  |                   |                  |
| (cid:13)(cid:13)(cid:13)                 | (cid:88) t                               | T −1              | (cid:13)(cid:13) | (cid:13)(cid:13)  |                  |
| (cid:13)(cid:13)(cid:13)                 | 2tE (cid:0)(cid:13)                      |                   |                  | (cid:1)           |                  |
| ≤ E                                      | (cid:13) ¯wt − w∗                        |                   |                  | + (cid:13)        |                  |
| + λ                                      |                                          |                   |                  | (cid:13) ¯wT − w∗ |                  |
| ∇F (T −1)( ¯wT ; D)                      |                                          |                   |                  | T −1              |                  |
|                                          | =0                                       |                   |                  |                   |                  |
|                                          | T −1                                     |                   |                  |                   |                  |
| (cid:13)(cid:13)(cid:13)                 | (cid:88) t                               | (cid:13) ¯wt − w∗ | (cid:13)(cid:13) | + λE (cid:13)     | (cid:13)(cid:13) |
| (cid:13)(cid:13)(cid:13)                 | 2tE (cid:13)                             | T −1              |                  | (cid:13)w0 − w∗   |                  |
| ≤ 2E                                     |                                          |                   |                  | T −1              |                  |
| + λ                                      |                                          |                   |                  |                   |                  |
| ∇F (T −1)( ¯wT ; D)                      |                                          |                   |                  |                   |                  |
|                                          | =1                                       |                   |                  |                   |                  |
|                                          | T −1                                     |                   |                  |                   |                  |
|                                          | (cid:112)                                |                   |                  |                   |                  |
| (cid:13)(cid:13)(cid:13)                 | (cid:88) t                               | λ2tαt + λRT −1    |                  |                   |                  |
| (cid:13)(cid:13)(cid:13)                 |                                          |                   |                  |                   |                  |
| ≤ 2E                                     |                                          |                   |                  |                   |                  |
| + 4                                      |                                          |                   |                  |                   |                  |
| ∇F (T −1)( ¯wT ; D)                      |                                          |                   |                  |                   |                  |
|                                          | =1                                       |                   |                  |                   |                  |
| T −1                                     |                                          |                   |                  |                   |                  |
| (cid:112)                                |                                          |                   |                  |                   |                  |
| (cid:112)                                |                                          |                   |                  |                   |                  |
| (cid:88) t                               | λ2t+1αt + λ2T /2R                        |                   |                  |                   |                  |
| ≤ 4                                      |                                          |                   |                  |                   |                  |
| L1αT + 4                                 |                                          |                   |                  |                   |                  |
| =1                                       |                                          |                   |                  |                   |                  |
| (cid:112)                                |                                          |                   |                  |                   |                  |
| (cid:112)                                |                                          |                   |                  |                   |                  |
| T(cid:88) t                              |                                          |                   |                  |                   |                  |
| ≤ 4                                      |                                          |                   |                  |                   |                  |
| λ2t+1αt +                                |                                          |                   |                  |                   |                  |
| λL1R                                     |                                          |                   |                  |                   |                  |
| =1                                       |                                          |                   |                  |                   |                  |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Linear time rate: The excess population risk guarantee of Algorithm  5  is in Lemma  D.6 , with Lipschitz parameter  L 0 being the same and  G ∆ =  O � L 1 2 T/ 2 � .   This gives us  α t   =   ˜ O � L 2 0 λ t n   + dL 2 0 λ t n 2 ε 2 � , and thus

E  ∥∇ F ( ¯ w ;  D ) ∥ =   ˜ O

� L 0 √ n   +

√

dL 0 nε + � λL 1 R

�

=   ˜ O

� L 0 √ n   +

√

dL 0 nε +   L 1 R √ n

�

where the last step follows by setting of  λ .   Finally, note that the Lemma  D.6  requires that  n  =   ˜ Ω � L 1 + λ t

λ t

� for all  t .   This can be checked to be satisfied by substituting the value of  λ t .

D.1. Utility Lemmas

We first present some key results which will be useful in the proofs. Lemma D.1.   Let  f   :   R d   → R  be an  L 1 -smooth convex function and let  w ∗ =   arg min w ∈ R d  f ( w ) .   Let  R   =   ∥ w ∗ ∥ and w 0   ∈ R d   such that  ∥ w 0 ∥≤ R .   Define   ˜ f ( w )   =   f ( w ) +   λ

2   ∥ w  − w 0 ∥ 2   and let   ˜ w   =   arg min   ˜ f ( w ) .   Then for any  λ   ≥ 0 , ∥ ˜ w ∥≤ √

2 R .

Proof.   From   optimality   criterion,   0   =   ∇ ˜ f ( ˜ w )   =   ∇ f ( ˜ w ) +  λ  ( ˜ w  − w 0 ) .   Therefore,   ∇ f ( ˜ w )   =   λ  ( w 0  − ˜ w )   and   thus ⟨∇ f ( ˜ w ) , w 0  − ˜ w ⟩ >  0 .   Furthermore, since  f   is convex, from monotonicity,  ⟨∇ f ( ˜ w ) , w ∗ − ˜ w ⟩≤ 0 .   Since both  w 0   and  w ∗

lie in the ball of radius  R  (say  W R ), the above two implies that the hyperplane  H   =  { w   :  ⟨∇ f ( ˜ w ) , w  − ˜ w ⟩ = 0 }  intersects with  W R .   Furthermore, since  ∇ f ( ˜ w ) =  λ  ( w 0  − ˜ w ) , we have that   ˜ w  is the projection of  w 0  on  H   i.e.   Π H ( w 0 ) .

Let  w ′   = Π H (0) .   We have that  w ′   ∈W R ; this is because the hyperplane cuts the hypersphere  W R   creating a spherical cap and  w ′   is the center of the cap.   From properties of convex projections  ∥ Π H ( w 0 )  − Π H (0) ∥≤∥ w 0  − 0 ∥≤ R .   Furthermore, Π H (0)   and   Π H ( w 0 )  − Π H (0)   are   orthogonal.   Hence   ∥ ˜ w ∥ 2   =   ∥ Π H ( w 0 ) ∥ 2   =   ∥ Π H (0) ∥ 2   +  ∥ Π H ( w 0 )  − Π H (0) ∥ 2   ≤ 2 R 2 .

We state the following result from ( Allen-Zhu ,  2018 ;  Foster et al. ,  2019 ). Lemma   D.2.   Suppose   for   every   t   =   1 ,  2 , . . . T ,   E [ F   ( t − 1) ( ¯ w t ;  D )]  − F   ( t − 1) ( w ∗ t − 1 ;  D )   ≤ α t   where   ¯ w t   are   the   iterates produced in the algorithm,  w ∗ t − 1   = arg min w ∈ R d  F  ( t − 1) ( w ;  D )  and  λ t   = 2 t λ , we have,

1.   For every  t  ≥ 1 ,  E [ �� ¯ w t  − w ∗ t − 1 �� 2 ]  ≤ 2 α t λ t − 1

2.   For every  t  ≥ 1 ,  E [ ∥ ¯ w t  − w ∗ t   ∥ 2 ]  ≤ α t

λ t

3.   E [ � T t =1   λ t  ∥ ¯ w t  − w ∗ T   ∥ ]  ≤ 4  � T t =1 √ α t λ t

D.2. Lemmas for NoisyGD (Algorithm  7 )

Lemma   D.3.   Consider   a   function   f ( w ;  x )   =   ℓ ( w ;  x )   +   ∆( w ) ,   where   w   �→ ℓ ( w ;  x )   is   convex   and   L 0   Lipschitz   for all   x ,   and   ∆( w )   is   λ   strongly   convex,   G ∆ Lipschitz   and   H ∆ smooth   over   a   bounded   convex   set   W .   Algorithm   6   run

with   parameters   η   =   log( T  )

λT ,   σ 2   =   64 L 2 0 T   log(1 /δ )

n 2 ε 2 ,   T   =   max � L 1 + H ∆

λ log � L 1 + H ∆ λ � , n 2 ε 2 ( L 2 0 + G 2 ∆ ) dL 2 0   log(1 /δ )

� and   S ( { w t } t )   =

1 � T t =1 (1 − ηλ ) − t � T t =1   (1  − ηλ ) − t  w t  satisfies  ( ε, δ ) -DP and given a dataset  S  of  n  i.i.d.   points from  D , the excess population risk of its output   ¯ w  is bounded by,

E � F ( ¯ w ;  D )  − min w ∈W R   F ( w ;  D ) � =  O � L 2 0 λn   +   dL 2 0   log (1 /δ )

λn 2 ε 2

� .

Proof.   For the privacy analysis, as in ( Bassily et al. ,  2014 ), for fixed  w , the sensitivity of the gradient update is bounded by 2 L 0

n   .   Applying advanced composition, we have that  σ 2   =   64 L 2 0 T   log(1 /δ )

n 2 ε 2 suffices for  ( ε, δ ) -DP.

For utility, we first compute a bound on uniform argument stability of the algorithm; let  { w t }  and  { w ′ t }  be sequence of iterates on neighbouring datasets.   Note that the function  w   �→ f ( w ;  x )  is  L 1  +  H ∆ -smooth and  λ -strongly convex for all  x . From the setting of  T , we have that the step size  η   ≤ 1 L 1 + H ∆ , hence from the standard stability analysis,

26



| 0                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                           |
| Linear time rate:                                                                                                                 |
| The excess population risk guarantee of Algorithm 5 is in Lemma D.6, with Lipschitz parameter L0                                  |
| (cid:17)                                                                                                                          |
| dL2                                                                                                                               |
| (cid:16) L2                                                                                                                       |
| 0                                                                                                                                 |
| , and thus                                                                                                                        |
| being the same and G∆ = O (cid:0)L12T /2(cid:1). This gives us αt = ˜O                                                            |
| λtn +                                                                                                                             |
| λtn2ε2                                                                                                                            |
| √                                                                                                                                 |
| √                                                                                                                                 |
| (cid:32)                                                                                                                          |
| (cid:33)                                                                                                                          |
| (cid:32)                                                                                                                          |
| (cid:33)                                                                                                                          |
| dL0                                                                                                                               |
| dL0                                                                                                                               |
| L1R                                                                                                                               |
| (cid:112)                                                                                                                         |
| L0√                                                                                                                               |
| L0√                                                                                                                               |
| √                                                                                                                                 |
| E ∥∇F ( ¯w; D)∥ = ˜O                                                                                                              |
| = ˜O                                                                                                                              |
| +                                                                                                                                 |
| +                                                                                                                                 |
| +                                                                                                                                 |
| +                                                                                                                                 |
| λL1R                                                                                                                              |
| nε                                                                                                                                |
| nε                                                                                                                                |
| n                                                                                                                                 |
| n                                                                                                                                 |
| n                                                                                                                                 |
| (cid:17)                                                                                                                          |
| (cid:16) L1+λt                                                                                                                    |
| where the last step follows by setting of λ. Finally, note that the Lemma D.6 requires that n = ˜Ω                                |
| for all t. This                                                                                                                   |
| λt                                                                                                                                |
| can be checked to be satisfied by substituting the value of λt.                                                                   |
| D.1. Utility Lemmas                                                                                                               |
| We first present some key results which will be useful in the proofs.                                                             |
| Lemma D.1. Let f : Rd → R be an L1-smooth convex function and let w∗ = arg minw∈Rd f (w). Let R = ∥w∗∥ and                        |
| f (w) = f (w) + λ                                                                                                                 |
| w0 ∈ Rd such that ∥w0∥ ≤ R. Define                                                                                                |
| 2 ∥w − w0∥2 and let ˜w = arg min ˜f (w). Then for any λ ≥ 0,                                                                      |
| √                                                                                                                                 |
| ∥ ˜w∥ ≤                                                                                                                           |
| 2R.                                                                                                                               |
| Proof. From optimality criterion, 0 = ∇ ˜f ( ˜w) = ∇f ( ˜w) + λ ( ˜w − w0). Therefore, ∇f ( ˜w) = λ (w0 − ˜w) and thus            |
| ⟨∇f ( ˜w), w0 − ˜w⟩ > 0. Furthermore, since f is convex, from monotonicity, ⟨∇f ( ˜w), w∗ − ˜w⟩ ≤ 0. Since both w0 and w∗         |
| lie in the ball of radius R (say WR), the above two implies that the hyperplane H = {w : ⟨∇f ( ˜w), w − ˜w⟩ = 0} intersects       |
| with WR. Furthermore, since ∇f ( ˜w) = λ (w0 − ˜w), we have that ˜w is the projection of w0 on H i.e. ΠH (w0).                    |
| Let w′ = ΠH (0). We have that w′ ∈ WR; this is because the hyperplane cuts the hypersphere WR creating a spherical cap            |
| and w′                                                                                                                            |
| is the center of the cap. From properties of convex projections ∥ΠH (w0) − ΠH (0)∥ ≤ ∥w0 − 0∥ ≤ R. Furthermore,                   |
| ΠH (0) and ΠH (w0) − ΠH (0) are orthogonal. Hence ∥ ˜w∥2 = ∥ΠH (w0)∥2 = ∥ΠH (0)∥2 + ∥ΠH (w0) − ΠH (0)∥2 ≤                         |
| 2R2.                                                                                                                              |
| We state the following result from (Allen-Zhu, 2018; Foster et al., 2019).                                                        |
| Lemma D.2.                                                                                                                        |
| Suppose for every t = 1, 2, . . . T , E[F (t−1)( ¯wt; D)] − F (t−1)(w∗                                                            |
| wt are the iterates                                                                                                               |
| t−1; D) ≤ αt where                                                                                                                |
| produced in the algorithm, w∗                                                                                                     |
| t−1 = arg minw∈Rd F (t−1)(w; D) and λt = 2tλ, we have,                                                                            |
| (cid:13)(cid:13)                                                                                                                  |
| 2                                                                                                                                 |
| ] ≤ 2αt                                                                                                                           |
| 1. For every t ≥ 1, E[(cid:13)                                                                                                    |
| (cid:13) ¯wt − w∗                                                                                                                 |
| t−1                                                                                                                               |
| λt−1                                                                                                                              |
| 2. For every t ≥ 1, E[∥ ¯wt − w∗                                                                                                  |
| t ∥2] ≤ αt                                                                                                                        |
| λt                                                                                                                                |
| √                                                                                                                                 |
| 3. E[(cid:80)T                                                                                                                    |
| αtλt                                                                                                                              |
| T ∥] ≤ 4 (cid:80)T                                                                                                                |
| t=1 λt ∥ ¯wt − w∗                                                                                                                 |
| t=1                                                                                                                               |
| D.2. Lemmas for NoisyGD (Algorithm 7)                                                                                             |
| Lemma D.3. Consider a function f (w; x) = ℓ(w; x) + ∆(w), where w (cid:55)→ ℓ(w; x) is convex and L0 Lipschitz for                |
| all x, and ∆(w) is λ strongly convex, G∆ Lipschitz and H∆ smooth over a bounded convex set W. Algorithm 6 run                     |
| (cid:18)                                                                                                                          |
| (cid:19)                                                                                                                          |
| n2ε2(L2                                                                                                                           |
| 0+G2                                                                                                                              |
| ∆)                                                                                                                                |
| 0T log(1/δ)                                                                                                                       |
| L1+H∆                                                                                                                             |
| with parameters η = log(T )                                                                                                       |
| , σ2 = 64L2                                                                                                                       |
| log (cid:0) L1+H∆                                                                                                                 |
| (cid:1) ,                                                                                                                         |
| and S({wt}t) =                                                                                                                    |
| λT                                                                                                                                |
| n2ε2                                                                                                                              |
| λ                                                                                                                                 |
| λ                                                                                                                                 |
| dL2                                                                                                                               |
| 0 log(1/δ)                                                                                                                        |
| (cid:80)T                                                                                                                         |
| 1                                                                                                                                 |
| (cid:80)T                                                                                                                         |
| t=1 (1 − ηλ)−t wt satisfies (ε, δ)-DP and given a dataset S of n i.i.d. points from D, the excess population                      |
| t=1(1−ηλ)−t                                                                                                                       |
| risk of its output ¯w is bounded by,                                                                                              |
| (cid:20)                                                                                                                          |
| (cid:21)                                                                                                                          |
| (cid:19)                                                                                                                          |
| (cid:18) L2                                                                                                                       |
| dL2                                                                                                                               |
| 0 log (1/δ)                                                                                                                       |
| 0                                                                                                                                 |
| E                                                                                                                                 |
| F ( ¯w; D) − min                                                                                                                  |
| F (w; D)                                                                                                                          |
| = O                                                                                                                               |
| +                                                                                                                                 |
| .                                                                                                                                 |
| w∈WR                                                                                                                              |
| λn                                                                                                                                |
| λn2ε2                                                                                                                             |
| Proof. For the privacy analysis, as in (Bassily et al., 2014), for fixed w, the sensitivity of the gradient update is bounded by  |
| 2L0                                                                                                                               |
| 0T log(1/δ)                                                                                                                       |
| . Applying advanced composition, we have that σ2 = 64L2                                                                           |
| suffices for (ε, δ)-DP.                                                                                                           |
| n                                                                                                                                 |
| n2ε2                                                                                                                              |
| For utility, we first compute a bound on uniform argument stability of the algorithm;                                             |
| let {wt} and {w′                                                                                                                  |
| t} be sequence of                                                                                                                 |
| iterates on neighbouring datasets. Note that the function w (cid:55)→ f (w; x) is L1 + H∆-smooth and λ-strongly convex for all x. |
| 1                                                                                                                                 |
| , hence from the standard stability analysis,                                                                                     |
| From the setting of T , we have that the step size η ≤                                                                            |
| L1+H∆                                                                                                                             |
| 26                                                                                                                                |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

w t +1  − w ′ t +1   =  w t   − η ∇ L ( w t ;  S )  − η ∇ ∆( w t )  − w ′ t   +  η ∇ L ( w ′ t ;  S ′ ) +  η ∇ ∆( w ′ t )

=  w t  − w ′ t   − η  ( ∇ L ( w t ;  S ) +  ∇ ∆( w t )  −∇ L ( w ′ t ;  S )  − η ∇ ∆( w ′ t ))

+  η  ( ∇ L ( w ′ t ;  S ′ )  −∇ L ( w ′ t ;  S ))

= � I  − η � ∇ 2 L ( ˜ w t ;  S ) +  ∇ 2 ∆( ˜ w t ) �� ( w t  − w ′ t )

+  η  ( ∇ L ( w ′ t ;  S ′ )  −∇ L ( w ′ t ;  S ))

where the last equality follows from Taylor remainder theorem where   ˜ w t  is some intermediate point on the line joining  w t and  w ′ t .   Using the fact that  η   ≤ 1 L 1 + H ∆ , we have

�� w t +1  − w ′ t +1 �� ≤ (1  − ηλ )  ∥ w t  − w ′ t ∥ +   2 ηL 0

n ≤ 2 L 0

λn

The above gives the same bound for the iterate using the selector  S ,

∥S ( { w t } )  −S ( { w ′ t } ) ∥≤ 2 L 0

λn

Note that the overall Lipschitz constant for the empirical loss is   ˜ L 0   =  L 0  +  G ∆ .   For the excess empirical risk guarantee, we use Lemma 5.2 in ( Feldman et al. ,  2020 ) to get,

E  [ L  ( ¯ w ;  S ) + ∆( ¯ w )  − L ( w ∗ ;  S )  − ∆( w ∗ )] =  E  [ F   ( ¯ w ;  S )  − F ( w ∗ ;  S )]

=   ˜ O

� ˜ L 0 2

λT

�

=   ˜ O

� ˜ L 0 2  +  σ 2 d

λT

�

=   ˜ O

� ˜ L 0 2

λT   +   dL 2 0   log (1 /δ )

λn 2 ε 2

�

=  O � dL 2 0   log (1 /δ )

λn 2 ε 2

�

where the last step follows from the setting of  T .   For the population risk guarantee, we have,

E  [ F ( ¯ w ;  D )  − F ( w ∗ ;  D )] =  E  [ F ( ¯ w ;  D )  − F ( ¯ w ;  S )] +  E  [ F ( ¯ w ;  D )  − F ( w ∗ )]

=  E [ L ( ¯ w ;  D )  − L ( ¯ w ;  S )] +  O � dL 2 0   log (1 /δ )

λn 2 ε 2

�

≤ L 0 E  ∥ ¯ w  − ¯ w ′ ∥ +  O � dL 2 0   log (1 /δ )

λn 2 ε 2

�

=   ˜ O � L 2 0 λn   +   dL 2 0   log (1 /δ )

λn 2 ε 2

�

where the inequality follows from Lipschitzness and standard generalization gap to stability argument.

D.3. Lemmas for PhasedSGD (Algorithm  5 )

The following lemma gives population risk guarantees for strongly convex functions under privacy, in terms of variance of stochastic gradients, as opposed to standard Lipschitzness bounds.

Lemma   D.4   (Variance   based   bound   for   constant   step-size   SGD   for   strongly-convex   functions) .   Consider   a   func- tion   f ( w ;  x )   such   that   w   �→ f ( w ;  x )   is   λ   strongly   convex,   L 1   smooth   over   a   convex   set   W   for   all   x   and   let

27



| 0                                                                                                                         |
|:--------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                   |
| wt+1 − w′                                                                                                                 |
| t+1 = wt − η∇L(wt; S) − η∇∆(wt) − w′                                                                                      |
| t + η∇L(w′                                                                                                                |
| t; S′) + η∇∆(w′                                                                                                           |
| t)                                                                                                                        |
| = wt − w′                                                                                                                 |
| t − η (∇L(wt; S) + ∇∆(wt) − ∇L(w′                                                                                         |
| t; S) − η∇∆(w′                                                                                                            |
| t))                                                                                                                       |
| + η (∇L(w′                                                                                                                |
| t; S′) − ∇L(w′                                                                                                            |
| t; S))                                                                                                                    |
| = (cid:0)I − η (cid:0)∇2L( ˜wt; S) + ∇2∆( ˜wt)(cid:1)(cid:1) (wt − w′                                                     |
| t)                                                                                                                        |
| + η (∇L(w′                                                                                                                |
| t; S′) − ∇L(w′                                                                                                            |
| t; S))                                                                                                                    |
| where the last equality follows from Taylor remainder theorem where ˜wt is some intermediate point on the line joining wt |
| 1                                                                                                                         |
| and w′                                                                                                                    |
| , we have                                                                                                                 |
| t. Using the fact that η ≤                                                                                                |
| L1+H∆                                                                                                                     |
| 2ηL0                                                                                                                      |
| 2L0                                                                                                                       |
| (cid:13)(cid:13)                                                                                                          |
| (cid:13)(cid:13)                                                                                                          |
| ≤                                                                                                                         |
| ≤ (1 − ηλ) ∥wt − w′                                                                                                       |
| wt+1 − w′                                                                                                                 |
| t∥ +                                                                                                                      |
| t+1                                                                                                                       |
| n                                                                                                                         |
| λn                                                                                                                        |
| The above gives the same bound for the iterate using the selector S,                                                      |
| 2L0                                                                                                                       |
| ∥S({wt}) − S({w′                                                                                                          |
| t})∥ ≤                                                                                                                    |
| λn                                                                                                                        |
| ˜                                                                                                                         |
| Note that the overall Lipschitz constant for the empirical loss is                                                        |
| L0 = L0 + G∆. For the excess empirical risk guarantee, we                                                                 |
| use Lemma 5.2 in (Feldman et al., 2020) to get,                                                                           |
| E [L ( ¯w; S) + ∆( ¯w) − L(w∗; S) − ∆(w∗)] = E [F ( ¯w; S) − F (w∗; S)]                                                   |
| 2                                                                                                                         |
| (cid:33)                                                                                                                  |
| (cid:32) ˜L0                                                                                                              |
| = ˜O                                                                                                                      |
| λT                                                                                                                        |
| 2                                                                                                                         |
| + σ2d                                                                                                                     |
| (cid:32) ˜L0                                                                                                              |
| = ˜O                                                                                                                      |
| λT                                                                                                                        |
| 2                                                                                                                         |
| dL2                                                                                                                       |
| (cid:32) ˜L0                                                                                                              |
| 0 log (1/δ)                                                                                                               |
| = ˜O                                                                                                                      |
| +                                                                                                                         |
| λT                                                                                                                        |
| λn2ε2                                                                                                                     |
| (cid:18) dL2                                                                                                              |
| 0 log (1/δ)                                                                                                               |
| = O                                                                                                                       |
| λn2ε2                                                                                                                     |
| where the last step follows from the setting of T . For the population risk guarantee, we have,                           |
| E [F ( ¯w; D) − F (w∗; D)] = E [F ( ¯w; D) − F ( ¯w; S)] + E [F ( ¯w; D) − F (w∗)]                                        |
| (cid:18) dL2                                                                                                              |
| 0 log (1/δ)                                                                                                               |
| = E[L( ¯w; D) − L( ¯w; S)] + O                                                                                            |
| λn2ε2                                                                                                                     |
| (cid:18) dL2                                                                                                              |
| 0 log (1/δ)                                                                                                               |
| ≤ L0E ∥ ¯w − ¯w′∥ + O                                                                                                     |
| λn2ε2                                                                                                                     |
| dL2                                                                                                                       |
| (cid:18) L2                                                                                                               |
| 0 log (1/δ)                                                                                                               |
| 0                                                                                                                         |
| +                                                                                                                         |
| = ˜O                                                                                                                      |
| λn                                                                                                                        |
| λn2ε2                                                                                                                     |
| where the inequality follows from Lipschitzness and standard generalization gap to stability argument.                    |
| D.3. Lemmas for PhasedSGD (Algorithm 5)                                                                                   |
| The following lemma gives population risk guarantees for strongly convex functions under privacy, in terms of variance of |
| stochastic gradients, as opposed to standard Lipschitzness bounds.                                                        |
| Lemma D.4 (Variance based bound for                                                                                       |
| constant                                                                                                                  |
| step-size SGD for                                                                                                         |
| strongly-convex functions). Consider a func-                                                                              |
| smooth over a convex                                                                                                      |
| tion f (w; x)                                                                                                             |
| such that w                                                                                                               |
| (cid:55)→ f (w; x)                                                                                                        |
| is λ strongly                                                                                                             |
| set W for all x and let                                                                                                   |
| convex, L1                                                                                                                |
| 27                                                                                                                        |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

E x  ∥∇ f ( w ;  x )  − E x ∇ f ( w ;  x ) ∥ 2   ≤V 2   for   all   w   ∈W .   Let   γ t   =   (1  − ηλ ) − t .   Given   a   dataset   S   =   { x 1 , x 2 , . . . , x n } sampled i.i.d from  D  and  η   ≤ 1 2 β   as input, for any  w   ∈W , the iterates of Algorithm  6  satisfy

E

� 1 � n t =1   γ t

n �

t =1 γ t F ( w t ;  D )

�

− F ( w )  ≤ λ e ηλn   − 1   ∥ w 0  − w ∥ 2  +  η V 2

Furthermore, for  n  = Ω � L 1 λ   log � L 1 λ �� , with  η   =   log( n )

λn and  S ( { w t } t ) = 1 � n t =1   γ t � n t =1   γ t w t , the excess population risk of ˜ w   =  S ( { w t } t )  satisfies

E � F ( ˜ w ;  D )  − min w ∈W   F ( w ;  D ) � =  O � V 2  log ( n ) λn

�

Proof.   An equivalent way to write the update in Algorithm  6  is

w t +1   = arg min w ∈W

� ⟨∇ f ( w t , x t ) , w ⟩ +   1

η   ∥ w t  − w ∥ 2  +  ψ ( w ) �

where  ψ ( w ) = 0  if  w   ∈W , otherwise  ∞ .

Following standard arguments in convex optimization, for any  w   ∈W , we have

F ( w t +1 ;  D )  − F ( w )

=  F ( w t +1 ;  D ) +  ψ ( w t +1 )  − F ( w ;  D )  − ψ ( w )

≤ F ( w t ) +  ⟨∇ F ( w t ) , w t +1  − w t ⟩ +   L 1

2   ∥ w t +1  − w t ∥ 2  +  ψ ( w t +1 )

+  F ( w ;  D )  − ψ ( w )

≤⟨∇ F ( w t ) , w t +1  − w t ⟩ +  ⟨∇ F ( w t ) , w t  − w ⟩− λ

2   ∥ w t  − w ∥ 2  +   L 1

2   ∥ w t +1  − w t ∥ 2

+  ψ ( w t +1 ) +  F ( w ;  D )  − ψ ( w )

=  E z t

� ⟨∇ p ( w t ;  z t )  −∇ F ( w ;  D ) , w t  − w t +1 ⟩ +   L 1

2   ∥ w t +1  − w t ∥ 2  +  ⟨∇ p ( w t ;  z t ) , w t  − w ⟩ �

− λ

2   ∥ w t  − w ∥ 2  +  ψ ( w t +1 ) +  F ( w ;  D )  − ψ ( w )

≤ E z t � ⟨∇ p ( w t ;  z t )  −∇ F ( w ;  D ) , w t  − w t +1 ⟩− � 1 2 η   − L 1

2

� ∥ w t +1  − w t ∥ 2

+ � 1 2 η   − λ

2

� ∥ w t  − w ∥ 2   − 1

2 η   ∥ w t +1  − w ∥ 2  �

≤ E z t � η 2 (1  − ηL 1 )   ∥∇ p ( w t ;  z t )  −∇ F ( w ;  D ) ∥ 2  + � 1 2 η   − λ

2

� ∥ w t  − w ∥ 2   − 1

2 η   ∥ w t +1  − w ∥ 2  �

≤ η V 2   +  E z t

�� 1 2 η   − λ

2

� ∥ w t  − w ∥ 2   − 1

2 η   ∥ w t +1  − w ∥ 2 �

where the first inequality follows from smoothness, the second from strong convexity, the third from Fact D.1 in ( Allen-Zhu , 2018 ), fourth from AM-GM inequality and the last from the assumption about variance bound on the oracle.

Now, the above is exactly the bound obtained in the proof of Lemma 5.2 in ( Feldman et al. ,  2020 ) with the second moment on gradient norm replaced by variance.   Repeating the rest of the arguments in that Lemma gives us the claimed result.

Lemma D.5  (Privacy of Algorithm  6 ) .   Consider a function  f ( w ;  x ) =  ℓ ( w ;  x ) + ∆( w )  such that  w   �→ ℓ ( w ;  x )  is convex, L 0   Lipschitz,  L 1 -smooth for all  z , and  ∆( · )  is  λ  strongly convex,  G ∆ Lipschitz and  H ∆ smooth over a bounded set  W . For   n   =   Ω � L 1 + H ∆ λ log � L 1 + H ∆ λ �� ,   Algorithm   6   with   input   as   function   ( w, x )   �→ f ( w ;  x ) ,   σ 2   =   64 G 2 (log( n )) 2  log(1 /δ )

λ 2 n 2 ε 2 , η   =   log( n )

λn and  S  ( { w t } n t =1 ) = 1 � n t =1   γ t � n t =1   γ t w t  for any weights  γ t  satisfies  ( ε, δ ) -DP.

28



| 0                                                                           | 1                                                                                                                            | 2                                               | 3      | 4   |
|:----------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------|:-------|:----|
|                                                                             | Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                      |                                                 |        |     |
|                                                                             | Ex ∥∇f (w; x) − Ex∇f (w; x)∥2 ≤ V 2 for all w ∈ W. Let γt = (1 − ηλ)−t. Given a dataset S = {x1, x2, . . . , xn}             |                                                 |        |     |
| sampled i.i.d from D and η ≤ 1                                              |                                                                                                                              |                                                 |        |     |
|                                                                             | 2β as input, for any w ∈ W, the iterates of Algorithm 6 satisfy                                                              |                                                 |        |     |
|                                                                             | (cid:34)                                                                                                                     |                                                 |        |     |
|                                                                             | (cid:35)                                                                                                                     |                                                 |        |     |
|                                                                             | 1                                                                                                                            |                                                 |        |     |
|                                                                             | λ                                                                                                                            |                                                 |        |     |
|                                                                             | E                                                                                                                            |                                                 |        |     |
|                                                                             | n(cid:88) t                                                                                                                  |                                                 |        |     |
|                                                                             | − F (w) ≤                                                                                                                    |                                                 |        |     |
|                                                                             | γtF (wt; D)                                                                                                                  |                                                 |        |     |
|                                                                             | ∥w0 − w∥2 + ηV 2                                                                                                             |                                                 |        |     |
|                                                                             | (cid:80)n                                                                                                                    |                                                 |        |     |
|                                                                             | eηλn − 1                                                                                                                     |                                                 |        |     |
|                                                                             | t=1 γt                                                                                                                       |                                                 |        |     |
|                                                                             | =1                                                                                                                           |                                                 |        |     |
| Furthermore, for n = Ω (cid:0) L1                                           | 1                                                                                                                            |                                                 |        |     |
|                                                                             | (cid:80)n                                                                                                                    |                                                 |        |     |
|                                                                             | log (cid:0) L1                                                                                                               |                                                 |        |     |
|                                                                             | (cid:1)(cid:1), with η = log(n)                                                                                              |                                                 |        |     |
|                                                                             | (cid:80)n                                                                                                                    | t=1 γtwt, the excess population risk of         |        |     |
|                                                                             | and S({wt}t) =                                                                                                               |                                                 |        |     |
|                                                                             | λ                                                                                                                            |                                                 |        |     |
|                                                                             | λ                                                                                                                            |                                                 |        |     |
|                                                                             | λn                                                                                                                           |                                                 |        |     |
|                                                                             | t=1 γt                                                                                                                       |                                                 |        |     |
| w = S({wt}t) satisfies                                                      |                                                                                                                              |                                                 |        |     |
|                                                                             | (cid:18) V 2 log (n)                                                                                                         |                                                 |        |     |
|                                                                             | E                                                                                                                            |                                                 |        |     |
|                                                                             | F ( ˜w; D) − min                                                                                                             |                                                 |        |     |
|                                                                             | F (w; D)                                                                                                                     |                                                 |        |     |
|                                                                             | = O                                                                                                                          |                                                 |        |     |
|                                                                             | w∈W                                                                                                                          |                                                 |        |     |
|                                                                             | λn                                                                                                                           |                                                 |        |     |
| Proof. An equivalent way to write the update in Algorithm 6 is              |                                                                                                                              |                                                 |        |     |
|                                                                             | (cid:18)                                                                                                                     |                                                 |        |     |
|                                                                             | (cid:19)                                                                                                                     |                                                 |        |     |
|                                                                             | 1 η                                                                                                                          |                                                 |        |     |
|                                                                             | ⟨∇f (wt, xt), w⟩ +                                                                                                           |                                                 |        |     |
|                                                                             | wt+1 = arg min                                                                                                               |                                                 |        |     |
|                                                                             | ∥wt − w∥2 + ψ(w)                                                                                                             |                                                 |        |     |
|                                                                             | w∈W                                                                                                                          |                                                 |        |     |
| where ψ(w) = 0 if w ∈ W, otherwise ∞.                                       |                                                                                                                              |                                                 |        |     |
| Following standard arguments in convex optimization, for any w ∈ W, we have |                                                                                                                              |                                                 |        |     |
|                                                                             | F (wt+1; D) − F (w)                                                                                                          |                                                 |        |     |
|                                                                             | = F (wt+1; D) + ψ(wt+1) − F (w; D) − ψ(w)                                                                                    |                                                 |        |     |
|                                                                             | L1                                                                                                                           |                                                 |        |     |
|                                                                             | ≤ F (wt) + ⟨∇F (wt), wt+1 − wt⟩ +                                                                                            |                                                 |        |     |
|                                                                             | ∥wt+1 − wt∥2 + ψ(wt+1)                                                                                                       |                                                 |        |     |
|                                                                             | 2                                                                                                                            |                                                 |        |     |
|                                                                             | + F (w; D) − ψ(w)                                                                                                            |                                                 |        |     |
|                                                                             | L1                                                                                                                           |                                                 |        |     |
|                                                                             | λ 2                                                                                                                          |                                                 |        |     |
|                                                                             | ≤ ⟨∇F (wt), wt+1 − wt⟩ + ⟨∇F (wt), wt − w⟩ −                                                                                 |                                                 |        |     |
|                                                                             | ∥wt − w∥2 +                                                                                                                  |                                                 |        |     |
|                                                                             | ∥wt+1 − wt∥2                                                                                                                 |                                                 |        |     |
|                                                                             | 2                                                                                                                            |                                                 |        |     |
|                                                                             | + ψ(wt+1) + F (w; D) − ψ(w)                                                                                                  |                                                 |        |     |
|                                                                             | (cid:20)                                                                                                                     | (cid:21)                                        |        |     |
|                                                                             | L1                                                                                                                           |                                                 |        |     |
|                                                                             | ⟨∇p(wt; zt) − ∇F (w; D), wt − wt+1⟩ +                                                                                        | ∥wt+1 − wt∥2 + ⟨∇p(wt; zt), wt − w⟩             |        |     |
|                                                                             | = Ezt                                                                                                                        |                                                 |        |     |
|                                                                             | 2                                                                                                                            |                                                 |        |     |
|                                                                             | λ 2                                                                                                                          |                                                 |        |     |
|                                                                             | −                                                                                                                            |                                                 |        |     |
|                                                                             | ∥wt − w∥2 + ψ(wt+1) + F (w; D) − ψ(w)                                                                                        |                                                 |        |     |
|                                                                             | (cid:104)                                                                                                                    |                                                 |        |     |
|                                                                             | (cid:18) 1                                                                                                                   |                                                 |        |     |
|                                                                             | L1                                                                                                                           |                                                 |        |     |
|                                                                             | −                                                                                                                            |                                                 |        |     |
|                                                                             | ⟨∇p(wt; zt) − ∇F (w; D), wt − wt+1⟩ −                                                                                        |                                                 |        |     |
|                                                                             | ∥wt+1 − wt∥2                                                                                                                 |                                                 |        |     |
|                                                                             | ≤ Ezt                                                                                                                        |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | 2                                                                                                                            |                                                 |        |     |
|                                                                             | (cid:18) 1                                                                                                                   |                                                 |        |     |
|                                                                             | λ 2                                                                                                                          |                                                 |        |     |
|                                                                             | 1                                                                                                                            |                                                 |        |     |
|                                                                             | +                                                                                                                            |                                                 |        |     |
|                                                                             | −                                                                                                                            |                                                 |        |     |
|                                                                             | ∥wt − w∥2 −                                                                                                                  |                                                 |        |     |
|                                                                             | ∥wt+1 − w∥2 (cid:105)                                                                                                        |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | (cid:19)                                                                                                                     |                                                 |        |     |
|                                                                             | (cid:104)                                                                                                                    |                                                 |        |     |
|                                                                             | (cid:18) 1                                                                                                                   |                                                 |        |     |
|                                                                             | λ 2                                                                                                                          | ∥wt+1 − w∥2 (cid:105)                           |        |     |
|                                                                             | η                                                                                                                            |                                                 |        |     |
|                                                                             | 1                                                                                                                            |                                                 |        |     |
|                                                                             | −                                                                                                                            |                                                 |        |     |
|                                                                             | ∥∇p(wt; zt) − ∇F (w; D)∥2 +                                                                                                  |                                                 |        |     |
|                                                                             | ∥wt − w∥2 −                                                                                                                  |                                                 |        |     |
|                                                                             | ≤ Ezt                                                                                                                        |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | 2 (1 − ηL1)                                                                                                                  |                                                 |        |     |
|                                                                             | (cid:19)                                                                                                                     |                                                 |        |     |
|                                                                             | (cid:20)(cid:18) 1                                                                                                           |                                                 |        |     |
|                                                                             | 1                                                                                                                            |                                                 |        |     |
|                                                                             | λ 2                                                                                                                          |                                                 |        |     |
|                                                                             | −                                                                                                                            |                                                 |        |     |
|                                                                             | ∥wt − w∥2 −                                                                                                                  |                                                 |        |     |
|                                                                             | ∥wt+1 − w∥2                                                                                                                  |                                                 |        |     |
|                                                                             | ≤ ηV 2 + Ezt                                                                                                                 |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | 2η                                                                                                                           |                                                 |        |     |
|                                                                             | where the first inequality follows from smoothness, the second from strong convexity, the third from Fact D.1 in (Allen-Zhu, |                                                 |        |     |
|                                                                             | 2018), fourth from AM-GM inequality and the last from the assumption about variance bound on the oracle.                     |                                                 |        |     |
|                                                                             | Now, the above is exactly the bound obtained in the proof of Lemma 5.2 in (Feldman et al., 2020) with the second moment      |                                                 |        |     |
|                                                                             | on gradient norm replaced by variance. Repeating the rest of the arguments in that Lemma gives us the claimed result.        |                                                 |        |     |
|                                                                             | Lemma D.5 (Privacy of Algorithm 6). Consider a function f (w; x) = ℓ(w; x) + ∆(w) such that w (cid:55)→ ℓ(w; x) is convex,   |                                                 |        |     |
|                                                                             | L0 Lipschitz, L1-smooth for all z, and ∆(·) is λ strongly convex, G∆ Lipschitz and H∆ smooth over a bounded set W.           |                                                 |        |     |
| For n = Ω (cid:0) L1+H∆                                                     | log (cid:0) L1+H∆                                                                                                            | (cid:55)→ f (w; x), σ2 = 64G2(log(n))2 log(1/δ) |        | ,   |
|                                                                             | (cid:1)(cid:1), Algorithm 6 with input as function (w, x)                                                                    |                                                 |        |     |
|                                                                             | λ                                                                                                                            |                                                 | λ2n2ε2 |     |
|                                                                             | λ                                                                                                                            |                                                 |        |     |
|                                                                             | (cid:80)n                                                                                                                    |                                                 |        |     |
| η = log(n)                                                                  | 1                                                                                                                            |                                                 |        |     |
|                                                                             | and S ({wt}n                                                                                                                 |                                                 |        |     |
|                                                                             | (cid:80)n                                                                                                                    |                                                 |        |     |
|                                                                             | t=1 γtwt for any weights γt satisfies (ε, δ)-DP.                                                                             |                                                 |        |     |
|                                                                             | t=1) =                                                                                                                       |                                                 |        |     |
| λn                                                                          | t=1 γt                                                                                                                       |                                                 |        |     |
|                                                                             | 28                                                                                                                           |                                                 |        |     |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Proof.   We start with computing the sensitivity of the algorithm’s output: let  { w t }  and  { w ′ t }  be sequence of iterates produced by Algorithm  6  on neighbouring datasets.   Note that the function  w   �→ f ( w ;  x )  is  L ′ 1   =  L 1   +  H ∆ -smooth and  λ -strongly convex for all  x .   From the assumption on  n , we have that the step size  η   ≤ 1 H + H ∆ .   Suppose the differing sample between neighbouring datasets is  x j , then  w t   =  w ′ t   for all  t  ≤ j .   Also,

�� w j +1  − w ′ j +1 �� =  η �� ∇ ℓ ( w j ;  x j )  −∇ ℓ ( w j ;  x ′ j ) �� ≤ 2 ηL 0   =   2 L 0  log ( n ) λn

Now, for any  t > j , as in the standard stability analysis we have,

w t +1  − w ′ t +1   =  w t   − η ∇ ℓ ( w t ;  x t )  − η ∇ ∆( w t )  − w t   +  η ∇ ℓ ( w ′ t ;  x t ) +  η ∇ ∆( w ′ t )

= � I  − η � ∇ 2 ℓ ( ˜ w t ;  x t ) +  ∇ 2 ∆( ˜ w t ) �� ( w t  − w ′ t )

where the last equality follows from Taylor remainder theorem where   ˜ w t  is some intermediate point in the line joining  w t and  w ′ t .   Using the fact that  η   ≤ 1 L 1 + H ∆ and  λ  strong convexity, we have

�� w t +1  − w ′ t +1 �� ≤ (1  − ηλ )  ∥ w t  − w ′ t ∥≤ �� w j +1  − w ′ j +1 �� ≤ 2 L 0  log ( n ) λn

Applying convexity to the weights in the definition of the selector function  S , we get,

∥S ( { w t } )  −S ( { w ′ t } ) ∥≤ 2 L 0  log ( n )

λn

The privacy proof now follows from the Gaussian mechanism guarantee.

Lemma D.6  (Phased SGD composite guarantee) .   Consider a function  f ( w ;  x )   =   ℓ ( w ;  x ) + ∆( w )  where  w   �→ ℓ ( w ;  x ) is   convex,   L 0   Lipschitz,   L 1   smooth   for   all   x ,   and   ∆( w )   is   λ   strongly   convex,   G ∆ Lipschitz   and   H ∆ smooth   over   a bounded set  W .   For  n   =   Ω � K ( L 1 + H ∆ )

λ log � L 1 + H ∆ λ � � , Algorithm  6  with  σ 2   =   64 L 2 0 K 2 (log( n )) 2  log(1 /δ )

λ 2 n 2 ε 2 , satisfies  ( ε, δ ) -

DP. Furthermore,   with   input as function   ( w, x )   �→ f ( w ;  x ) ,   a   dataset   S   of   n   samples drawn i.i.d.   from   D ,   η   =   log( n )

λn   , K   = ln ln  n ,  γ t   = (1  − ηλ ) − t   and  S  ( { w t } n t =1 ) = 1 � n t =1   γ t � n t =1   γ t w t , the excess population risk of output  w K   is bounded as

E  [ F ( w K ;  D )]  − min w ∈W   F ( w ;  D ) =   ˜ O � L 2 0 λn   +   dL 2 0 λn 2 ε 2

�

Proof.   The   privacy   proof   simply   follows   from   parallel   composition.   For   the   utility   proof,   we   repeat   the   arguments   in Theorem 5.3 in ( Feldman et al. ,  2020 ) substituting the variance-based bound from Lemma  D.4 .   Note that the variance of the stochastic gradients used,  V 2   ≤ L 2 0 , this gives us,

E  [ F ( w K ;  D )]  − min w ∈W   F ( w ;  D ) =   ˜ O � L 2 0 λn   +   dL 2 0 λn 2 ε 2

�

E. Missing Results for Generalized Linear Models

We first give the definition of oblivious subspace embedding.

Definition E.1  ( ( r, τ, β ) -oblivious subspace embedding) .   A random matrix  Φ  ∈ R k × d   is an  ( r, τ, β ) -oblivious subspace embedding if for any  r  dimensional linear subspace in  R d , say  V  , we have that with probability at least  1  − β , for all  x  ∈ V  ,

(1  − τ )  ∥ x ∥ 2   ≤∥ Φ x ∥ 2   ≤ (1 +  τ )  ∥ x ∥ 2

29



| 0                                                                                                                               |
|:--------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                         |
| Proof. We start with computing the sensitivity of the algorithm’s output:                                                       |
| let {wt} and {w′                                                                                                                |
| t} be sequence of iterates produced                                                                                             |
| by Algorithm 6 on neighbouring datasets. Note that the function w (cid:55)→ f (w; x) is L′                                      |
| 1 = L1 + H∆-smooth and λ-strongly                                                                                               |
| 1                                                                                                                               |
| convex for all x. From the assumption on n, we have that the step size η ≤                                                      |
| . Suppose the differing sample between                                                                                          |
| H+H∆                                                                                                                            |
| neighbouring datasets is xj, then wt = w′                                                                                       |
| t for all t ≤ j. Also,                                                                                                          |
| 2L0 log (n)                                                                                                                     |
| (cid:13)(cid:13)                                                                                                                |
| (cid:13)(cid:13)                                                                                                                |
| = η (cid:13)                                                                                                                    |
| wj+1 − w′                                                                                                                       |
| (cid:13)∇ℓ(wj; xj) − ∇ℓ(wj; x′                                                                                                  |
| (cid:13) ≤ 2ηL0 =                                                                                                               |
| j)(cid:13)                                                                                                                      |
| j+1                                                                                                                             |
| λn                                                                                                                              |
| Now, for any t > j, as in the standard stability analysis we have,                                                              |
| wt+1 − w′                                                                                                                       |
| t+1 = wt − η∇ℓ(wt; xt) − η∇∆(wt) − wt + η∇ℓ(w′                                                                                  |
| t; xt) + η∇∆(w′                                                                                                                 |
| t)                                                                                                                              |
| = (cid:0)I − η (cid:0)∇2ℓ( ˜wt; xt) + ∇2∆( ˜wt)(cid:1)(cid:1) (wt − w′                                                          |
| t)                                                                                                                              |
| where the last equality follows from Taylor remainder theorem where ˜wt is some intermediate point in the line joining wt       |
| 1                                                                                                                               |
| and λ strong convexity, we have                                                                                                 |
| and w′                                                                                                                          |
| t. Using the fact that η ≤                                                                                                      |
| L1+H∆                                                                                                                           |
| 2L0 log (n)                                                                                                                     |
| (cid:13)(cid:13)                                                                                                                |
| (cid:13)(cid:13)                                                                                                                |
| (cid:13)(cid:13)                                                                                                                |
| ≤                                                                                                                               |
| wt+1 − w′                                                                                                                       |
| ≤ (1 − ηλ) ∥wt − w′                                                                                                             |
| (cid:13)wj+1 − w′                                                                                                               |
| t+1                                                                                                                             |
| t∥ ≤ (cid:13)                                                                                                                   |
| j+1                                                                                                                             |
| λn                                                                                                                              |
| Applying convexity to the weights in the definition of the selector function S, we get,                                         |
| 2L0 log (n)                                                                                                                     |
| ∥S({wt}) − S({w′                                                                                                                |
| t})∥ ≤                                                                                                                          |
| λn                                                                                                                              |
| The privacy proof now follows from the Gaussian mechanism guarantee.                                                            |
| Lemma D.6 (Phased SGD composite guarantee). Consider a function f (w; x) = ℓ(w; x) + ∆(w) where w (cid:55)→ ℓ(w; x)             |
| is convex, L0 Lipschitz, L1                                                                                                     |
| smooth for all x, and ∆(w) is λ strongly convex, G∆ Lipschitz and H∆ smooth over a                                              |
| (cid:16) K(L1+H∆)                                                                                                               |
| 0K2(log(n))2 log(1/δ)                                                                                                           |
| (cid:1)(cid:17)                                                                                                                 |
| bounded set W. For n = Ω                                                                                                        |
| , Algorithm 6 with σ2 = 64L2                                                                                                    |
| , satisfies (ε, δ)-                                                                                                             |
| log (cid:0) L1+H∆                                                                                                               |
| λ                                                                                                                               |
| λ                                                                                                                               |
| λ2n2ε2                                                                                                                          |
| ,                                                                                                                               |
| DP. Furthermore, with input as function (w, x) (cid:55)→ f (w; x), a dataset S of n samples drawn i.i.d.                        |
| from D, η = log(n)                                                                                                              |
| λn                                                                                                                              |
| (cid:80)n                                                                                                                       |
| 1                                                                                                                               |
| (cid:80)n                                                                                                                       |
| K = ln ln n, γt = (1 − ηλ)−t and S ({wt}n                                                                                       |
| t=1) =                                                                                                                          |
| t=1 γtwt, the excess population risk of output wK is bounded                                                                    |
| t=1 γt                                                                                                                          |
| as                                                                                                                              |
| (cid:18) L2                                                                                                                     |
| dL2                                                                                                                             |
| 0                                                                                                                               |
| 0                                                                                                                               |
| F (w; D) = ˜O                                                                                                                   |
| +                                                                                                                               |
| E [F (wK; D)] − min                                                                                                             |
| w∈W                                                                                                                             |
| λn                                                                                                                              |
| λn2ε2                                                                                                                           |
| Proof. The privacy proof simply follows from parallel composition. For the utility proof, we repeat                             |
| the arguments in                                                                                                                |
| Theorem 5.3 in (Feldman et al., 2020) substituting the variance-based bound from Lemma D.4. Note that the variance of the       |
| stochastic gradients used, V 2 ≤ L2                                                                                             |
| 0, this gives us,                                                                                                               |
| (cid:19)                                                                                                                        |
| (cid:18) L2                                                                                                                     |
| dL2                                                                                                                             |
| 0                                                                                                                               |
| 0                                                                                                                               |
| F (w; D) = ˜O                                                                                                                   |
| +                                                                                                                               |
| E [F (wK; D)] − min                                                                                                             |
| w∈W                                                                                                                             |
| λn                                                                                                                              |
| λn2ε2                                                                                                                           |
| E. Missing Results for Generalized Linear Models                                                                                |
| We first give the definition of oblivious subspace embedding.                                                                   |
| Definition E.1 ((r, τ, β)-oblivious subspace embedding). A random matrix Φ ∈ Rk×d is an (r, τ, β)-oblivious subspace            |
| embedding if for any r dimensional linear subspace in Rd, say V , we have that with probability at least 1 − β, for all x ∈ V , |
| (1 − τ ) ∥x∥2 ≤ ∥Φx∥2 ≤ (1 + τ ) ∥x∥2                                                                                           |
| 29                                                                                                                              |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

It is well-known that JL matrices with embedding dimension  k   =  O � r  log(2 /β )

τ   2 � are  ( r, τ, β ) -oblivious subspace embeddings

and can be constructed efficiently ( Cohen ,  2016 ).   A simple example is a scaled Gaussian random matrix,  Φ = 1 √

k G  where entries of  G  are independent and distributed as  N (0 ,  1) .

Proof of Theorem  6.1 .   We first prove privacy.   Let  G ( S )  and  H ( S )  be the bounds on the Lipschitz and smoothness constants of the family of loss functions  { w   �→ f ( w ; Φ x ) } x ∈ S .   With  k   = Ω(log (2 n/δ )) , from the JL-property, it follows that with probability at least  1  − δ/ 2 ,  G ( S )   ≤ 2 L 0  ∥X∥ and  H ( S )   ≤ 2 L 1  ∥X∥ 2 .   Hence, using the fact that  A  is  ( ε, δ/ 2) -DP, we have that Algorithm  4  is  ( ε, δ ) -DP.

We   now   proceed   to   the   utility   part.   Let   ˜ w   ∈ R k   be   the   output   of   the   base   algorithm   in   low   dimensions.   Note   that   the final   output   is   ¯ w   =   Φ ⊤ ˜ w .   The   transpose   of   the   JL   matrix   can   only   increase   the   norm   by   the   polynomial   factor   of   d and   n ,   hence   ∥ ¯ w ∥≤ poly ( n, d )  ∥ ˜ w ∥ .   By   assumption,   P  ( ∥ ˜ w ∥ >  poly ( n, d, L 0 , L 1 ))   ≤ 1 √ n .   Hence   we   also   have   that P  ( ∥ ¯ w ∥ >  poly ( n, d, L 0 , L 1 ))  ≤ 1 √ n .   Let  W   ⊆ R d   denote the above set with radius poly ( n, d, L 0 , L 1 ) .

We now decompose the population stationarity as,

E  ∥∇ F ( ¯ w ;  D ) ∥≤ E  ∥∇ F ( ¯ w ;  D )  −∇ F ( ¯ w ;  S ) ∥ +  ∥∇ F ( ¯ w ;  S ) ∥

≤ E   sup w ∈W ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥ +   L 0  ∥X∥ √ n +  E  ∥∇ F ( ¯ w ;  S ) ∥ , (7)

where the last inequality follows from the above reasoning that that  P   ( ¯ w   ∈W )  ≥ 1  − 1 √ n .   The first term is bounded from uniform convergence guarantee in Lemma  E.2  noting that the dependence on  ∥W∥ in the Lemma is only poly-logarithmic.

E   sup w ∈W ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥ =   ˜ O � L 0  ∥X∥ √ n

� (8)

We now prove a bound on the empirical stationarity.   Note that it suffices to prove a high-probability (over the random JL matrix) bound because the norm of gradient is bounded in worst case by  L 0  ∥X∥ .   Thus the expected norm of gradient of the output is bounded by the high probability bound by considering a small enough failure probability.

From the assumption on  A , with probability at least  1  − δ/ 2 ,

∥∇ F ( ˜ w ; Φ S ) ∥ =  E

����� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ )Φ x i

����� ≤ g ( k, n,  2 L 0  ∥X∥ ,  2 L 0  ∥X∥ , ε, δ/ 2)

We   now   use   the   fact   that   if   k   =   O  ( rank  log (2 n/δ )) ,   then   the   JL   transform   is   an   ( rank ,  1 / 2 , δ/ 2)   oblivious   subspace embedding (see Definition  E.1 ).   Thus, it approximates the norm of any vector in  span ( { x i } n i =1 ) , and hence any gradient. Therefore,

E  ∥∇ F ( ˜ w ; Φ S ) ∥ =  E

����� Φ

� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) x i

������ ≥

�

1  −

� rank

k

�

E

����� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) x i

�����

≥ 1

2 E

����� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) x i

����� =   1 2 E

����� 1 n

n �

i =1 ϕ ′ y i ( � Φ ⊤ ˜ w, x i � ) x i

����� =   1 2 E  ∥∇ F ( ¯ w ;  S ) ∥

Thus with  k   =  O  ( rank  log (2 n/δ )) , we get

E  ∥∇ F ( ¯ w ;  S ) ∥≤ g ( k, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ ) =  g ( rank , n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ )

For the other bound, let  I d − k   ∈ R d × k   denote the matrix with first  k  diagonal entries,  ( I d − k ) j,j   with  j   ∈ [ k ] , are  1  and the

30



| 0                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------|
| Faster Rates of Convergence to Stationary Points in Differentially Private Optimization                                        |
| (cid:17)                                                                                                                       |
| (cid:16) r log(2/β)                                                                                                            |
| It is well-known that JL matrices with embedding dimension k = O                                                               |
| are (r, τ, β)-oblivious subspace embeddings                                                                                    |
| τ 2                                                                                                                            |
| and can be constructed efficiently (Cohen, 2016). A simple example is a scaled Gaussian random matrix, Φ = 1√                  |
| G where                                                                                                                        |
| k                                                                                                                              |
| entries of G are independent and distributed as N (0, 1).                                                                      |
| Proof of Theorem 6.1. We first prove privacy. Let G(S) and H(S) be the bounds on the Lipschitz and smoothness constants        |
| of the family of loss functions {w (cid:55)→ f (w; Φx)}x∈S. With k = Ω(log (2n/δ)), from the JL-property, it follows that with |
| probability at least 1 − δ/2, G(S) ≤ 2L0 ∥X ∥ and H(S) ≤ 2L1 ∥X ∥2. Hence, using the fact that A is (ε, δ/2)-DP, we            |
| have that Algorithm 4 is (ε, δ)-DP.                                                                                            |
| We now proceed to the utility part. Let                                                                                        |
| w ∈ Rk be the output of the base algorithm in low dimensions. Note that                                                        |
| the                                                                                                                            |
| final output                                                                                                                   |
| is                                                                                                                             |
| w = Φ⊤ ˜w. The transpose of the JL matrix can only increase the norm by the polynomial                                         |
| factor of d                                                                                                                    |
| 1                                                                                                                              |
| √                                                                                                                              |
| and n, hence ∥ ¯w∥ ≤ poly(n, d) ∥ ˜w∥. By assumption, P (∥ ˜w∥ > poly(n, d, L0, L1)) ≤                                         |
| n . Hence we also have that                                                                                                    |
| n . Let W ⊆ Rd denote the above set with radius poly(n, d, L0, L1).                                                            |
| We now decompose the population stationarity as,                                                                               |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

rest of the matrix is zero.   We have,

E  ∥∇ F ( ¯ w ;  S ) ∥

=  E

����� 1 n

n �

i =1 ϕ ′ y i ( � Φ ⊤ ˜ w, x i � ) x i

�����

≤ E

����� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) I d − k Φ x i

����� +  E

������ 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) x i   − 1

n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) I d − k Φ x i

�����

�

≤ E  ∥ I d − k ∥

����� 1 n

n �

i =1 ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ )Φ x i

����� +   1 n E

n �

i =1

�� ϕ ′ y i ( ⟨ ˜ w,  Φ x i ⟩ ) �� |∥ x i  − I d − k Φ x i ∥|

≤ E  ∥∇ F ( ˜ w ; Φ S ) ∥ +   1

n E

n �

i =1 L 0  ∥ I   − I d − k Φ ∥∥ x i ∥

≤ g ( k, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2) +  L 0  ∥X∥ E  ∥ I   − H ∥

where the second inequality follows from triangle inequality, the third inequality follows from  L 0 -Lipschitzness of the GLM, the third inequality follows from the accuracy guarantee of the base algorithm and substituting  H   =   I d − k Φ .   To bound E  ∥ I   − H ∥ , we use concentration properties of distribution used in the construction of JL matrices.   Specifically, using the scaled Gaussian matrix construction, from concentration of extreme eignevalues of square Gaussian matrices, we have that E  ∥ I   − H ∥ =   ˜ O � 1 √

k

� ( Rudelson & Vershynin ,  2010 ).   This gives us,

E  ∥∇ F ( ¯ w ;  S ) ∥≤ g ( k, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2) +   ˜ O � L 0  ∥X∥ √

k

�

Choosing   k   to   minimize   the   above   yields   the   bound   of   ˜ O � L 0 ∥X∥ √

k

� .   Combining   the   two   cases,   yields   the   bound   of

g ( k, n,  2 L 0  ∥X∥ ,  2 L 1  ∥X∥ 2   , ε, δ/ 2)  on gradient norm.   Plugging this and the bound in Eqn.   ( 8 ) in Inequality ( 7 ) gives the claimed bound.

Lemma E.2.   Let  D  be a probability distribution over  X   such that  ∥ x ∥≤∥X∥ for all  x   ∈ supp ( D ) .   Let  f ( w ; ( x, y ))   = ϕ y  ( ⟨ w, x ⟩ )  be an  L 1 -smooth  L 0 -Lipschitz GLM. Then, with probability at least  1  − β , over a draw of  n  i.i.d.   samples  S from  D , we have

sup w ∈W ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥≤ 4 L 0  ∥X∥ log � 2 n 3 / 2   ∥W∥ L 1  ∥X∥ /L 0 �

√ n +   4 L 0  ∥X∥ � log (1 /β ) √ n

Proof.   We first give a bound on the expected uniform deviation,  E S ∼D n  sup w ∈W  ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥ .   The gradient of the loss function is  ∇ f ( w ;  x ) =  ϕ ′ x   ( ⟨ w, x ⟩ )  x .   We start with the standard symmetrization trick,

E S ∼D n   sup w ∈W ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥

=  E S ∼D n   sup w ∈W

����� E ϕ ′ y   ( ⟨ w, x ⟩ )  x  − 1

n

n �

i =1 ϕ ′ x i   ( ⟨ w, x i ⟩ )  x i

�����

=  E S ∼D n   sup w ∈W

����� E { x ′ i } ∼D n 1 n

n �

i =1 ϕ ′ y ′ i   ( ⟨ w, x ′ i ⟩ )  x ′ i   − 1

n

n �

i =1 ϕ ′ x i   ( ⟨ w, x i ⟩ )  x i

�����

≤ E S,S ′ ∼D n   sup w ∈W

����� 1 n

n �

i =1 ϕ ′ y ′ i   ( ⟨ w, x ′ i ⟩ )  x ′ i   − 1

n

n �

i =1 ϕ ′ x i   ( ⟨ w, x i ⟩ )  x i

�����

=  E S,S ′ ∼D n E { σ i }   sup w ∈W

����� 1 n

n �

i =1 σ i � ϕ ′ y ′ i   ( ⟨ w, x ′ i ⟩ )  x ′ i   − ϕ ′ x i   ( ⟨ w, x i ⟩ )  x i � � ����

≤ 2 E S ∼D n E { σ i }   sup w ∈W

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w, x i ⟩ )  x i

����� (9)

31



| 0                                        | 1                                                            | 2                                        | 3                                | 4           | 5                | 6                    | 7                         | 8                   | 9                                        |
|:-----------------------------------------|:-------------------------------------------------------------|:-----------------------------------------|:---------------------------------|:------------|:-----------------|:---------------------|:--------------------------|:--------------------|:-----------------------------------------|
| rest of the matrix is zero. We have,     |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| E ∥∇F ( ¯w; S)∥                          |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                     |                                          |                                  |             |                  |                      |                           |                     |                                          |
| 1 n                                      | ((cid:10)Φ⊤ ˜w, xi                                           |                                          |                                  |             |                  |                      |                           |                     |                                          |
| n(cid:88) i                              | (cid:11))xi                                                  |                                          |                                  |             |                  |                      |                           |                     |                                          |
| = E                                      |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| ϕ′                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| yi                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| =1                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
|                                          |                                                              |                                          | (cid:34)(cid:13)                 |             |                  |                      |                           |                     | (cid:35)                                 |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (⟨ ˜w, Φxi⟩)Id−kΦxi                                          | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (cid:13)(cid:13)(cid:13)(cid:13) | n(cid:88) i | ϕ′               | 1 n                  | n(cid:88) i               | (⟨ ˜w, Φxi⟩)Id−kΦxi | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
| 1 n                                      |                                                              | + E                                      | 1 n                              |             |                  | (⟨ ˜w, Φxi⟩)xi −     | ϕ′                        |                     |                                          |
| n(cid:88) i                              |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| ≤ E                                      |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| ϕ′                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| yi                                       |                                                              |                                          |                                  |             | yi               |                      | yi                        |                     |                                          |
| =1                                       |                                                              |                                          |                                  | =1          |                  |                      | =1                        |                     |                                          |
| (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (⟨ ˜w, Φxi⟩)Φxi                                              | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | 1 n                              | n(cid:88) i | (cid:12)(cid:12) | (⟨ ˜w, Φxi⟩)(cid:12) | (cid:12) |∥xi − Id−kΦxi∥| |                     |                                          |
| 1 n                                      |                                                              | +                                        | E                                |             | ϕ′               |                      |                           |                     |                                          |
| n(cid:88) i                              |                                                              |                                          |                                  |             | yi               |                      |                           |                     |                                          |
| ϕ′                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| ≤ E ∥Id−k∥                               |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| yi                                       |                                                              |                                          |                                  |             |                  |                      |                           |                     |                                          |
| =1                                       |                                                              |                                          |                                  | =1          |                  |                      |                           |                     |                                          |
| ≤ E ∥∇F ( ˜w; ΦS)∥ +                     | E                                                            |                                          |                                  |             |                  |                      |                           |                     |                                          |
|                                          | 1 n                                                          | L0 ∥I − Id−kΦ∥ ∥xi∥                      |                                  |             |                  |                      |                           |                     |                                          |
|                                          | n(cid:88) i                                                  |                                          |                                  |             |                  |                      |                           |                     |                                          |
|                                          | =1                                                           |                                          |                                  |             |                  |                      |                           |                     |                                          |
|                                          | ≤ g(k, n, 2L0 ∥X ∥ , 2L1 ∥X ∥2 , ε, δ/2) + L0 ∥X ∥ E ∥I − H∥ |                                          |                                  |             |                  |                      |                           |                     |                                          |




| 0       | 1                | 2                                        | 3                                                              | 4            | 5                                        | 6            | 7                                        |
|:--------|:-----------------|:-----------------------------------------|:---------------------------------------------------------------|:-------------|:-----------------------------------------|:-------------|:-----------------------------------------|
|         |                  |                                          | x (⟨w, x⟩) x. We start with the standard symmetrization trick, |              |                                          |              |                                          |
| ES∼Dn   | sup              | ∥∇F (w; D) − ∇F (w; S)∥                  |                                                                |              |                                          |              |                                          |
|         | w∈W              |                                          |                                                                |              |                                          |              |                                          |
| = ES∼Dn | sup              | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | ϕ′                                                             | (⟨w, xi⟩) xi | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |              |                                          |
|         |                  | 1 n                                      |                                                                |              |                                          |              |                                          |
|         |                  | n(cid:88) i                              |                                                                |              |                                          |              |                                          |
|         |                  | Eϕ′                                      |                                                                |              |                                          |              |                                          |
|         |                  | y (⟨w, x⟩) x −                           |                                                                |              |                                          |              |                                          |
|         |                  |                                          | xi                                                             |              |                                          |              |                                          |
|         | w∈W              |                                          |                                                                |              |                                          |              |                                          |
|         |                  | =1                                       |                                                                |              |                                          |              |                                          |
|         |                  | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                                                                |              |                                          |              | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
|         |                  | E                                        |                                                                |              |                                          |              |                                          |
| = ES∼Dn | sup              | 1 n                                      | (⟨w, x′                                                        | 1 n          | n(cid:88) i                              | ϕ′           | (⟨w, xi⟩) xi                             |
|         |                  | n(cid:88) i                              | i⟩) x′                                                         |              |                                          |              |                                          |
|         |                  | ϕ′                                       | i −                                                            |              |                                          |              |                                          |
|         |                  | y′                                       |                                                                |              |                                          |              |                                          |
|         |                  | i                                        |                                                                |              |                                          | xi           |                                          |
|         |                  | {x′                                      |                                                                |              |                                          |              |                                          |
|         |                  | i}∼Dn                                    |                                                                |              |                                          |              |                                          |
|         | w∈W              |                                          |                                                                |              |                                          |              |                                          |
|         |                  | =1                                       |                                                                |              | =1                                       |              |                                          |
|         | ≤ ES,S′∼Dn       | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | 1 n                                                            | ϕ′           | (⟨w, xi⟩) xi                             |              | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
|         |                  | 1 n                                      | n(cid:88) i                                                    |              |                                          |              |                                          |
|         |                  | n(cid:88) i                              | i −                                                            |              |                                          |              |                                          |
|         |                  | ϕ′                                       |                                                                |              |                                          |              |                                          |
|         |                  | sup                                      |                                                                |              |                                          |              |                                          |
|         |                  | (⟨w, x′                                  |                                                                |              |                                          |              |                                          |
|         |                  | i⟩) x′                                   |                                                                |              |                                          |              |                                          |
|         |                  | y′                                       |                                                                | xi           |                                          |              |                                          |
|         |                  | i                                        |                                                                |              |                                          |              |                                          |
|         |                  | w∈W                                      |                                                                |              |                                          |              |                                          |
|         |                  | =1                                       | =1                                                             |              |                                          |              |                                          |
|         |                  | (cid:16)                                 |                                                                |              |                                          |              | (cid:17)                                 |
|         | = ES,S′∼Dn E{σi} | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (⟨w, x′                                                        | i − ϕ′       |                                          | (⟨w, xi⟩) xi | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
|         |                  | 1 n                                      | i⟩) x′                                                         |              |                                          |              |                                          |
|         |                  | n(cid:88) i                              |                                                                |              |                                          |              |                                          |
|         |                  | ϕ′                                       |                                                                |              |                                          |              |                                          |
|         |                  | sup                                      |                                                                |              |                                          |              |                                          |
|         |                  | σi                                       |                                                                |              |                                          |              |                                          |
|         |                  | y′                                       |                                                                |              | xi                                       |              |                                          |
|         |                  | i                                        |                                                                |              |                                          |              |                                          |
|         |                  | w∈W                                      |                                                                |              |                                          |              |                                          |
|         |                  | =1                                       |                                                                |              |                                          |              |                                          |
|         | ≤ 2ES∼Dn E{σi}   | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13)                       |              |                                          |              | (9)                                      |
|         |                  | 1 n                                      | (⟨w, xi⟩) xi                                                   |              |                                          |              |                                          |
|         |                  | n(cid:88) i                              |                                                                |              |                                          |              |                                          |
|         |                  | sup                                      |                                                                |              |                                          |              |                                          |
|         |                  | σiϕ′                                     |                                                                |              |                                          |              |                                          |
|         |                  | yi                                       |                                                                |              |                                          |              |                                          |
|         |                  | w∈W                                      |                                                                |              |                                          |              |                                          |
|         |                  | =1                                       |                                                                |              |                                          |              |                                          |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

where  σ i   are i.i.d.   Rademacher random variables.   For fixed  { x i } n i =1 , consider a set  W 0   s.t.   for all  w   ∈W   and  i   ∈ [ n ] , there exists  w 0   ∈W 0   such that  |⟨ w, x i ⟩−⟨ w 0 , x i ⟩| ≤ τ .   Since  ∥ w ∥≤∥W∥ and  ∥ x i ∥≤∥X∥ , we require only   2 n ∥W∥∥X∥

τ points in  W 0  to satisfy the above covering condition.   Therefore,

E S ∼D n E { σ i }   sup w ∈W

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w, x i ⟩ )  x i

�����

=  E S ∼D n E { σ i } sup w ∈W ,w 0 ∈W 0

����� 1 n

n �

i =1 σ i � ϕ ′ y i   ( ⟨ w, x i ⟩ )  − ϕ ′ y i   ( ⟨ w 0 , x i ⟩ ) +  ϕ ′ y i   ( ⟨ w 0 , x i ⟩ ) � x i

�����

≤ E S ∼D n E { σ i } sup w ∈W ,w 0 ∈W 0

����� 1 n

n �

i =1 σ i � ϕ ′ y i   ( ⟨ w, x i ⟩ )  − ϕ ′ y i   ( ⟨ w 0 , x i ⟩ ) � x i

����� +

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w 0 , x i ⟩ )  x i

�����

≤ E S ∼D n E { σ i } sup w ∈W ,w 0 ∈W 0 L 1  |⟨ w, x i ⟩−⟨ w 0 , x i ⟩| ∥X∥ +  E S ∼D n E { σ i } sup w 0 ∈W 0

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w 0 , x i ⟩ )  x i

�����

≤ L 1 τ  ∥X∥ +  E S ∼D n E { σ i } sup w 0 ∈W 0

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w 0 , x i ⟩ )  x i

����� (10)

where the second last inequality follows from smoothness and the last from the definition of cover  W 0 .   For fixed  w 0 , from standard manipulations, we have,

E { σ i }

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w 0 , x i ⟩ )  x i

����� ≤

� � � � E { σ i }

����� 1 n

n �

i =1 σ i ϕ ′ y i  ( ⟨ w 0 , x i ⟩ )  x i

�����

2

=

� � � � 1 n 2  E { σ i }

n �

i =1

�� σ i ϕ ′ y i  ( ⟨ w 0 , x i ⟩ )  x i �� 2

≤ L 0  ∥X∥ √ n

Using Massart’s finite class lemma to handle all  w 0   ∈W 0 , and substituting the above in Eqn.   ( 10 ), we get,

E S ∼D n E { σ i }   sup w ∈W

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w, x i ⟩ )  x i

����� ≤ L 1 τ  ∥X∥ +   G  ∥X∥ log (2 n  ∥W∥∥X∥ /τ ) √ n

Choosing  τ   = L 0 L 1 √ n , we get,

E S ∼D n E { σ i }   sup w ∈W

����� 1 n

n �

i =1 σ i ϕ ′ y i   ( ⟨ w, x i ⟩ )  x i

����� ≤ 2 L 0  ∥X∥ log � 2 n 3 / 2   ∥W∥ L 1  ∥X∥ /L 0 �

√ n

Finally, substituting the above in Eqn.   ( 9 ) gives us the following in-expectation bound.

E S ∼D n   sup w ∈W ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥≤ 4 L 0  ∥X∥ log � 2 n 3 / 2   ∥W∥ L 1  ∥X∥ /L 0 �

√ n

For the high-probability bound, let  ψ ( S ) = sup w ∈W  ∥∇ F ( w ;  D )  −∇ F ( w ;  S ) ∥ and let  w ∗ ∈W   achieves the supremum. We can bound the increment between neighbouring datasets  S   and  S ′   as,

| ψ ( S )  − ψ ( S ′ ) | ≤|∥∇ F ( w ∗ ;  D )  −∇ F ( w ∗ ;  S ) ∥−∥∇ F ( w ∗ ;  D )  −∇ F ( w ∗ ;  S ′ ) ∥|

≤∥∇ F ( w ∗ ;  S )  −∇ F ( w ∗ ;  S ′ ) ∥

≤ 2 L 0  ∥X∥

n

Finally, applying McDiarmid’s inequality gives the claimed bound.

32



| 0             | 1                                        | 2                                         | 3            | 4         | 5                                        | 6                                        | 7                    | 8                                        | 9                                        |
|:--------------|:-----------------------------------------|:------------------------------------------|:-------------|:----------|:-----------------------------------------|:-----------------------------------------|:---------------------|:-----------------------------------------|:-----------------------------------------|
| sup           | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                                           | (⟨w, xi⟩) xi |           | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                                          |                      |                                          |                                          |
| ES∼Dn E{σi}   | 1 n                                      |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               | n(cid:88) i                              |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               | σiϕ′                                     |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               |                                          | yi                                        |              |           |                                          |                                          |                      |                                          |                                          |
| w∈W           |                                          |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               | =1                                       |                                           |              |           |                                          |                                          |                      |                                          |                                          |
| = ES∼Dn E{σi} | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | 1 n                                       | σi           | (cid:0)ϕ′ | (⟨w, xi⟩) − ϕ′                           | (⟨w0, xi⟩) + ϕ′                          | (⟨w0, xi⟩)(cid:1) xi | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                                          |
|               | sup                                      | n(cid:88) i                               |              |           |                                          |                                          |                      |                                          |                                          |
|               |                                          |                                           |              | yi        | yi                                       | yi                                       |                      |                                          |                                          |
|               | w∈W,w0∈W0                                |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               |                                          | =1                                        |              |           |                                          |                                          |                      |                                          |                                          |
| ≤ ES∼Dn E{σi} | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | 1 n                                       | σi           | (cid:0)ϕ′ | (⟨w, xi⟩) − ϕ′                           | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | n(cid:88) i          | σiϕ′                                     | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
|               | sup                                      | n(cid:88) i                               |              |           |                                          | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |                      |                                          | (⟨w0, xi⟩) xi                            |
|               |                                          |                                           |              |           |                                          | 1 n                                      |                      |                                          |                                          |
|               |                                          |                                           |              |           |                                          | +                                        |                      |                                          |                                          |
|               |                                          |                                           |              |           |                                          | (⟨w0, xi⟩)(cid:1) xi                     |                      |                                          |                                          |
|               |                                          |                                           |              | yi        | yi                                       |                                          |                      | yi                                       |                                          |
|               | w∈W,w0∈W0                                |                                           |              |           |                                          |                                          |                      |                                          |                                          |
|               |                                          | =1                                        |              |           |                                          |                                          | =1                   |                                          |                                          |
| ≤ ES∼Dn E{σi} | sup                                      | L1 |⟨w, xi⟩ − ⟨w0, xi⟩| ∥X ∥ + ES∼DnE{σi} |              |           |                                          | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) | 1 n                  | σiϕ′                                     | (cid:13)(cid:13)(cid:13)(cid:13)(cid:13) |
|               |                                          |                                           |              |           |                                          | sup                                      | n(cid:88) i          |                                          | (⟨w0, xi⟩) xi                            |
|               |                                          |                                           |              |           |                                          |                                          |                      |                                          | yi                                       |
|               | w∈W,w0∈W0                                |                                           |              |           |                                          | w0∈W0                                    |                      |                                          |                                          |
|               |                                          |                                           |              |           |                                          |                                          |                      | =1                                       |                                          |


Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

Proof of Corollary  6.2 .   The results follow from Theorem  6.1  provided we show that the conditions on the base algorithm in the Theorem statement are satisfied.   The privacy and accuracy claims follow from Theorem  3.2  and  5.1  respectively.   We note that even though we are given population stationarity guarantee for the convex case, the same bound for empirical stationarity guarantee simply follows from the re-sampling argument in ( Bassily et al. ,  2019 ).   The only thing left to show is the high-probability bound on the trajectory of the algorithm.

Non-convex setting with Private Spiderboost: From the update in Algorithm  2 , we have that for any  t

∥∇ t ∥≤

t �

i =1 ∥ ∆ i ∥ +

�����

t �

i =1 g t

����� ≤ 2 tL 0  +

�����

t �

i =1 g t

�����

where   the   last   inequality   follows   from   the   Lipschitzness   assumption. Note   that   g t ∼N (0 , σ 2 t   I )   where   σ t ≤

O  (max ( σ 1 ,  � σ 2 ))   =   O  ( poly ( n, d, L 0 , L 1 )) .   Hence ���� t i =1   g t ��� ≤ � d  log (1 /β ′ ) O  ( poly ( n, d, L 0 , L 1 ))   with   probabil- ity   at   least   1  − β ′ .   Taking   a   union   bound   over   all   t   ∈ T   gives   us   ∥ w t ∥≤ poly ( n, d, L 0 , L 1 ,  log ( poly ( n, d ) /β ))   with probability at least  1  − β .   Substituting  β   = 1 √ n   yields the guarantee of Theorem  6.1 .

Convex setting with Recursive Regularization: Since the iterates are restricted to the constraint set, the final output, with probability one, lies in the set of radius

R T   = 2 T/ 2   ∥ w ∗ ∥ =  O

�� L 1

λ   ∥ w ∗ ∥

�

=  O

� L 1  ∥ w ∗ ∥ 3 / 2   n

L 0

�

which completes the proof.

33



| 0                                                                                                                             | 1                                                                                       |
|:------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
|                                                                                                                               | Faster Rates of Convergence to Stationary Points in Differentially Private Optimization |
| Proof of Corollary 6.2. The results follow from Theorem 6.1 provided we show that the conditions on the base algorithm in     |                                                                                         |
| the Theorem statement are satisfied. The privacy and accuracy claims follow from Theorem 3.2 and 5.1 respectively. We         |                                                                                         |
| note that even though we are given population stationarity guarantee for the convex case, the same bound for empirical        |                                                                                         |
| stationarity guarantee simply follows from the re-sampling argument in (Bassily et al., 2019). The only thing left to show is |                                                                                         |
| the high-probability bound on the trajectory of the algorithm.                                                                |                                                                                         |
| Non-convex setting with Private Spiderboost:                                                                                  | From the update in Algorithm 2, we have that for any t                                  |


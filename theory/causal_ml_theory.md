# Causal Machine Learning - A brief introduction

## Causal modeling and introduction

### Causality & probability
We are used to think of causality as a deterministic process, but we can gain a lot by pursuing a **probabilistic analysis of causality**.

There are two main reasons that justify this approach:
- Causes may sometimes make an event **more probable, but not certain**. With a probabilistic approach we can study the *strenght* of causal connections and not just their presence.
- Even the most assertive causal expression in natural language often contain **exceptions** that are not listed for simplicity: "If I forget the cake in the electric oven it will get burned, *expect* if my mother takes it out, if the lights go out, etc...". Probability theory helps us take into account the uncertainty related to those exceptions. 

#### Why is Association Not Causality?

The term association refers to statistical dependence. Causality is a special kind of association, so causality implies association but not viceversa.

In a graphical model association flows along all unblocked paths. In causal graph, causation flows along directed paths. Since causation is a sub-category of association, both flow along directed paths. We refer to the flow of association along directed paths as *causal association*; an example of a common type of non-causal association is called *confounding association*, when a confounder is involved.

Another important distinction is between interventional and observational data. 
- **Interventional data** is the one that came after an **experiment** in which we perform an action on the system and then measure the results of the action. In this controlled setting it is easy to measure the causal effect, e.g. in physics experiments. 
- **Observational data** is information collected by observing and recording behaviors, events, or phenomena as they naturally occur, without any interference or any actions by the researcher. When dealing with observational data it is more complicated to identify causal relationship because confounding is almost always introduced into the data.

### Potential Outcomes, ITE, ATE

Let $T$ be the random variable for the treatment, $Y$ the r.v. for the oucome of nterest and X to denote covariates. Assume for example a binary treatment, *i.e.*, $t\in{0,1}$ where $t$ is the realization of $T$.

The **potential outcome** $Y(t)$ denotes what the outcome would be after the treatment $T=t$, i.e., all the possible outcomes after each possible treatment.

Problem: not all potential outcomes are observed but all potential outcomes can potentially be observed. The one that is actually observed depends on the value that the treatment $T$ takes on.

Potential outcomes that we don't (and cannot) observe are known as **counterfactuals**, the one that is actually observed is referred to as a **factual**.


In general we consider many individuals in the population of size $n$. We define the **Individual Treatment Effect** (ITE) as:

$$
\tau_i=Y_i(1)-Y_i(0)
$$

The *fundamental problem of causal inference* is that we cannot observe both $Y_i(0)$ and $Y_i(1)$ for the same individual and so neither the ITE. The idea is to use the **Average Treatment Effect** (ATE):

$$
\text{ATE} = \mathbb{E}[Y(1)-Y(0)]
$$

### Ignorability and Conditional Ignorability

In a real dataset, for each individual, we have only one between $Y(1)$ and $Y(0)$, so how do we compute the ATE? We need to make an assumption in order to be able to compute the ATE:

**Ignorability Assumption:** $(Y(1),Y(0))$ statistically independent of $T$. 

Means that we assume that the treatment was randomly assigned to each individual and not accordingly to some individual feature.

Note this seems a bit counterintuitive. The observed outcome $Y$ depends on the treatment $T$, this is exactly what we want to study. The ignorability assumption means that the assigment mechanism of the treatment $T$ is independent from the potential outcomes $Y(0), Y(1)$. The possible outcomes are an intrinsic property of the individual, the treatmen only trigger the one observed.

Under this assumption we can compute the ATE as:

$$
\text{ATE} = \mathbb{E}[Y(1)-Y(0)]=\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)]
=\mathbb{E}[Y(1)|T=1]-\mathbb{E}[Y(0)|T=0]
$$

the ignorability assumption is equivalent to the **exchangeability assumption**, that is:

$$
\mathbb{E}[Y(1)|T=1]=\mathbb{E}[Y(1)|T=0]=\mathbb{E}[Y(1)]
$$
and
$$
\mathbb{E}[Y(0)|T=0]=\mathbb{E}[Y(0)|T=1]=\mathbb{E}[Y(0)]
$$

A causal quantity e.g. $\mathbb{E}[Y(0)]$ is identifiable if we can compute it from a pure statistical quantity e.g. $\mathbb{E}[Y|T=0]$.

In general, a part from randomized tests, this is completely unrealistic because there is likely to be confounding in most data we observe. 
The idea is so to control for relevant variables by conditioning them. Consider the set of covariates $X$. 

We then ask for **conditional exchangangeability/unconfoundedness**:
$(Y(1),Y(0))$ statistically independent of $T$, conditioned on $X$. 

>Note: we need to two more technical assumptions, namely no interference and consistency, but this are very technical and not so interesting.

With this assumption, controlling for $X$ makes the treatment groups comparable. In this way there is no association flows in the path $T\leftarrow X\rightarrow Y$, so there is no longer any non-causal association between $T$ and $Y$. Under this assumption we can compute:

$$
\mathbb{E}[Y(1)-Y(0)|X]=\mathbb{E}[Y(1)|X]-\mathbb{E}[Y(0)|X]=\\
=\mathbb{E}[Y(1)|T=1,X]-\mathbb{E}[Y(0)|T=0,X]=\\
=\mathbb{E}[Y|T=1,X]-\mathbb{E}[Y|T=0,X]
$$

and 

$$
\text{ATE}=\mathbb{E}_X[\mathbb{E}[Y(1)-Y(0)|X]]=\mathbb{E}_X[\mathbb{E}[Y|T=1,X]-\mathbb{E}[Y|T=0,X]]
$$

This last formula is called **adjustment formula**.


#### How do we use it?

For real data we do not know if conditional exchangeability holds since there may be some unobserved confounders that are not part of X. This is where **CEVAE** helps!

Knowing this formula we can in principle compute the ATE just by training a ML model in order to learn $\mathbb{E}[Y|t,x]$ so that we can generate data for the two different value of $t$ ($x$ fixed) and compute the difference.

CEVAE use a similiar strategy: learn z using a VAE and then compute $\mathbb{E}[Y|t,z]$ using the decoder. The idea is that $X$ is only a proxy of an unknown confounder $Z$ in a bigger space.

### The *do*-operator formalism
In probability we have conditioning: $T=t$ means that we restrict the focus to the subset of population to those who received the treatment $t$. In contrast in causal inference, an intervention would be to take the whole population and give everyone treatment $t$, this operation is denoted through the *do*-operator $do(T=t)$.

We can write the distribution of the potential outcome $Y(t)$ as: 
$$
P(Y(t)=y)=P(Y=y|do(T=t))
$$
and the ATE as:
$$
\text{ATE}=\mathbb{E}[Y|do(T=1)]-\mathbb{E}[Y|do(T=0)]
$$

With this operator we are now able to express causal concepts using conditional statements. A note about the jargon: all the distributions with the do operator inside are called *interventional distributions*. An expression containing the do operator such that can be reduced to one without the do operator is called *identifiable*.

Consider a bayesian network such that:
$$
P(y,t,x)=P(x)P(t|x)P(y|t,x)
$$
then if we intervene on the treatment:
$$
P(y,x|do(t))=P(x)P(y|t,x)
$$
so we just remove the factor related to the treatment, i.e., it is equal to one.

In the do-operator formalism the adjustment formula is equivalent to something called **backdoor adjustment**.

### Probabilistic Graphical Models

To represent relations between variables we have seen different Probabilistic Graphical Models (PGM): here we consider specifically **Bayesian Networks** (BN) which are a type of Directed Acyclic Graph (DAG).

In a Bayesian Network the edges represent *association/dependence, not causation*! In fact, given a joint probability distribution we can refactorize it in many different ways and for each factorization we obtain a valid Bayesian Network.

Meanwhile, in **Causal Bayesian Networks** edges represent causality instead of association. This stems from the idea that causal knowledge is superior to associational knowledge. Indeed, causal relations are closer to how we perceive the real world and are more "stable": a graphical model constructed on causal relationships risk less changes than a model based on associational relationships. The main reason is that causal relationships are *ontological* - they describe physical constraints- whereas probabilistic relationships are *epistemic* -they describe what we know or believe.

A PGM based on causality also imply that:
- If I **manually do** something, e.g. set a variable to a specific value, that has normally causes, I have to delete those causal relations. Since I "forced" it, I can't pretend something caused it.
- If I **observe**  something that has causes, I can instead infer something on its natural causes.


> **Historical note**: causal models were introduced first in the form of deterministic, *functional* equations.
The probabilities were introduced as a consequence of the inability to observe all variables needed in the deterministic process. This reflects **Laplace conception** of natural phenomena: *causality is deterministic and randomness stems from our ignorance*.
The **modern**, and opposite, **view** instead states that *causality is inherently probabilistic* and determinism is just an approximation.

### Queries in Causal Models
Given a causal model, there are **three main queries** we can ask:
- **predictions**: if we *observe* this, will that happen?
- **interventions**: if we *do* this, will that happen?
- **conterfactuals**: *given* that we *observed* this and saw that, if we observed not-this would that (or not-that) happen?

### Functional Causal Models
A functional causal model assumes Laplace conception of causality. It consists in a set of equations of the form:
$$x_i = f_i(pa_i,u_i), \ i = 1,...,n$$
where:
- $pa_i$ are the parents (direct causes) of $X_i$
- $U_i$ are the errors due to omitted factors (exogenous noise)

It can be interpreted as the _law_ that specifies which value nature would assign to $X_i$ based on whatever combination of values for $pa_i$ and $u_i$.

Why would we need functional causal models instead of causal graphical models? Because stochastic causal models (like causal bayesian netowrks) are **insufficient** to compute conterfactuals! I would need knowledge of the actual process from cause to effect. What instead **we can compute**  are **interventional quantities**, like the treatment effect or the expected outcome in case we force a certain treatment value. These quantities will be enough for our analysis so we won't touch Functional Causal Models any further. 

### What we need to remeber to do Causal Analysis
Main difference between statical and causal concepts: "behind every causal claim there must lie some causal assumption that is not discernable from the joint distribution and, hence, not testable in observational studies". These causal assumptions are usually provided by human experts and then tested.

We also need new mathematical notation for causal inference: classic probability calculus can't distinguish between base concepts like statistical dependence and causal dependence.

> **Remark**: there is a lot of literature about learning the *existence* of causal relationships from raw data: we will **not** cover this topic since in our case we assume to know what are the relations (by human reasoning) so we focus on inferring the "strenght" of these relations  

## Identification of causal effects in a causal model
Given a causal graph and the set of observed variables, identification of causal effect is the process of determining whether the effect can be estimated using the available variables’ data.

Identifiability ensures that the added assumptions conveyed by a causal model will supply the missing information without explicating the causal model in full detail.

Whenever we try to evaluate the effect of one factor T on another Y, the question arises as to whether we should adjust our measurements for possible variations in some other factors X, also called *confounders*. The problem is that, based on which variables we adjust for, we risk to fall into *Simpson's Paradox*! (A positive relation between two variables becomes negative, or viceversa, based on which variable we adjust for).

What criterion should we use to decide which variables are appropriate for adjustment? 

There exists a simple graphical test, the *back-door criterion*, that can be applied to the causal diagram in order to test if a subset of variables is sufficient to yield an unbiased estimate of the causal effect P(y|do(T=t)).

**Back-door criterion**:
A set of variables X satisfies the criterion relative to an ordered pair of variables $(T,Y)$ in a directed acyclic graph G if:
- no node in X is a descendant of $T$
- X blocks every path from $T$ to $Y$
 

## Estimation of causal effect in a causal model
Once we have identified the expression for the causal effect under the model assumptions, we can finally estimate the causal effect using statistical methods.

For example, if we have obtained a backdoor set we use the **Back-door adjustment**:
If a set of variables X satisfy the back-door criterion relative to (T,Y) then the causal effect of T on Y is identifiable and is given by the formula:
$$p(y|do(t))=\sum_x P(y|t,x)P(x)$$

We can see from this formula that estimating the back-door adjust is thus equivalent to estimate the *conditional probability distribution* $P(y|t,x)$. One of the most common method to do so is to use **linear regression**. 

The linear regression method is useful when the data-generating process for the outcome Y can be approximated as the linear function $Y = \beta_0 + \beta_1 T + \beta_2 X + \varepsilon$. When we fit the linear model we are estimating $\mathbb{\hat E}[Y|T=t,X=x]$.

We then compute the causal effect by marginalizing over all values of x and we use as estimate of the probability of each value $\hat P(x)$ the frequency of that value in the dataset. We obtain:
$$\hat p(y|do(t)) \approx \sum_x \mathbb{\hat E}[Y|T=t,X=x]\hat P(x)$$

## DoWhy library

DoWhy is a useful library to do causal inference and answering different types of causal questions.

In case I don't know the causal graph DoWhy provides some **discovery algorithms** to learn the causal structure from the data, though they do not garantee the validity of a learned graph.

We can also check if a proposal causal graph makes sense: if the dataset does not satisfy any of the Local Markov Conditions (conditional indeprendencies) implied by the graph, then the graph is invalid. If it satisfies them it could still be wrong, so we can use it only to refute a causal graph and not to accept it.

Since we will use synthetic data we assume to have already the correct causal graph. For this reason we will **not** further approach the aforementioned topics: instead we will focus on identifying and estimating causal effects.

### Functions from the DoWhy.casual_model module

To **model** the known causal mechanisms:
```py
model = CausalModel(data, treatment, outcome, common_causes)
```
**Args**:
- data = pandas dataframe containg treatment, outcome and other
- treatment = name of the treatment variable
- outcome = name of the outcome variable
- common_causes = names of common causes of treatment and outcome
- effect_modifiers = names of variables that can modify the treatment effect. If not provided, then the causal graph is used to find effect modifiers. Estimators will return multiple different estimates based on each value of effect_modifiers.
- graph = path to a DOT file containing a DAG or a string containg a DAG in DOT format 

To **identify** the target estimand: 
```py
identified_estimand = model.identify_effect(method)
```
**Args**:
- method_name = type of backdoor adjustment. As default it will do a mix of minimal and maximal adjustments and it will return one of the smallest valid backdoor set.
- proceed_when_unidentifiable = binary flag indicating whether identification should proceed in the presence of (potential) unobserved confounders.
 

To **estimate** causal effect:
```py
estimate = model.estimate_effect()
```
**Args**:
- identified_estimand = a probability expression that represents the effect to be estimated
- method_name = method to estimate the causal effect. Common method: "backdoor.linear_regression". (Others include "backdoor.propensity_score_matching","backdoor.generalized_linear_model" or “iv.instrumental_variable”)
- control_value = value of the treatment in the control group, for effect estimation
- treatment_value = value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list
- test_significance = binary flag on whether to additionally do a statistical signficance test for the estimate

## References
- ["Causality" of J. Pearl](https://bayes.cs.ucla.edu/BOOK-2K/)
- [Online course "Introduction to causal inference" of Brady Neal ](https://www.bradyneal.com/causal-inference-course)

- [DoWhY documentation: Estimating causal effects](https://www.pywhy.org/dowhy/v0.12/user_guide/causal_tasks/estimating_causal_effects/index.html)
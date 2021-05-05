## Project Description

_We want to find out in which sense a "Wasserstein t-SNE" can be useful. Which datasets exist where each datum itself is a probability distribution? How are they visualized/clustered now, and which metrics on probability distributions can improve clustering?_

### Datasets

#### WisconsinBreastCancer:

_get better clustering results than without Wassserstein_

 - assume Gaussian distribution in each of the 10 dimensions
 - Problem: no new information with Wasserstein approach?


#### ElectionResults:

_visualize as clusters of districts_

 - each district is one datum with p.d. over parties
 - Problem: same as using squared distance of percentages?


#### EuropeaValuesStudy:

_Try to reproduce a plot similar to Sinus-Milieus_

 - one participant is one datum (p.d. is approval of topics)
 - Problem: define topics in the questionaire...

 OR

 - one district is one datum (p.d. over answers of participants in district)
 - Problem: few datapoints?


### Implementation

 - opentsne framework
 - flexible with metric in high- and low-dimensional space


## Other Ideas

Survey: What are you scared of?

 - Each participant y describes a probability distribution
 - With what probability would x be the fear of y at a given moment
 - we can then apply t-SNE to visualize clusters
 - How would one do it without probability distribution approach?

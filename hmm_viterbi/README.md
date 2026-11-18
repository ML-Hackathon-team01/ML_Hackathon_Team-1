\# HMM and Viterbi Route Reconstruction



\## Implementation



We used the Gradient Boosting classifier to score candidate road arcs for each GPS observation. Classifier-only prediction selects the highest-scoring candidate independently at each observation. HMM/Viterbi instead selects a high-scoring sequence, using both classifier scores and directed road connectivity.



\## Hidden states



Each hidden state is a candidate directed road arc for one GPS observation. Viterbi selects one candidate per observation to form a route sequence.



\## Emission scores



The emission score is the Gradient Boosting model's predicted probability for a candidate. It is calculated using candidate rank, GPS-to-candidate distance, direction difference, GPS speed, and time gap.



\## Transition scores



The implementation assigns relative weights of 0.85 to staying on the same arc, 0.14 to moving to an arc connected in the correct direction, and 0.01 to a disconnected transition. These weights are normalized across the available previous candidates.



\## Missing observations and GPS gaps



Decoding restarts after a missing observation ID or a GPS time gap greater than 60 seconds. The evaluated student-test input contained candidates for all 19,160 observations.



\## Disconnected roads



Disconnected transitions are allowed with a low fallback weight of 0.01. The decoder prefers same-arc or directly connected alternatives, but it does not insert intermediate roads to repair a disconnected route.



\## Results



| Measure | Classifier only | Classifier + HMM/Viterbi |

|---|---:|---:|

| Provisional route-match rate | 76.20% | 84.62% |

| Route continuity | 77.97% | 97.10% |



In this run, the provisional route-match rate increased by 8.42 percentage points and route continuity increased by 19.12 percentage points.



\## Limitation



The current labels are provisional: a candidate is labelled positive if its arc appears anywhere in the trip's reference route, not necessarily at that specific GPS observation. Therefore, the route-match rates are not verified route accuracy and should be reported as preliminary results. The disconnected-transition fallback also means some disconnected pairs may remain.



\## Reproducibility



Run from the repository root with:



```cmd

python models\\hmm\_viterbi.py




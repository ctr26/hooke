# Rescue Screen

Code in this module is adapted from the following RXRX repositories.

```
platform-analysis-flows:
    Date: 2025-10-21
    Commit hash: 45b4b0aa9960cf27f6a0c6ee050da42419ae03af
    URL: https://github.com/recursionpharma/platform-analysis-flows/

cartographer:
    Date: 2025-10-21
    Commit hash: 5ddcb5e7d8039d2bf8eefcd5e9e855c08a8fb692
    URL: https://github.com/recursionpharma/cartographer/

rxrx19a-metrics:
    Date: 2025-10-24
    Commit hash: d8f8c452476e0ae08ee889c2e9bd8e0fa29db4e4
    URL: https://github.com/recursionpharma/rxrx19a-metrics/
```

> [!NOTE]
> The rest of this description was copied over from the [PR](https://github.com/valence-labs/vcb/pull/30).

## What
- Add a simplified, stand-alone script that replicates the [Rescue Flow](https://github.com/recursionpharma/platform-analysis-flows/blob/trunk/flows/rescue_screen.py).
  - Remove outliers using `IsolationForests`
  - Transform the data using `PCAW`
  - Project into Prometheus Space
  - Sample and aggregate "glyphons" for the plot. 
  - Compute and aggregate hit-scores
- Associated unit test cases

### What not?

> [!NOTE] 
> All of the below optimizations were added to make the pipeline more robust to nosier experiments (e.g. siRNA). They are _nice to have_, not _must have_. Historically, versions of this pipeline without these optimizations were already successfully used at RXRX on much noisier data than we're using.

- **Z-factors**: You can think of z-factors as a quality control metric for the experiment. It's a measure of separability between the disease model cloud and the negative control (i.e. healthy) cloud. We don't need this when computing hit scores. The formula for computing hit scores already takes into account the separability through use of the standard deviations of these clouds.
- **Theil Sen filtering**: We use [Isolation Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#) for outlier detection in the disease and negative control data. The production-ready code uses something called Theil-Sen filtering for outlier detection in the treatment data. You can think of it as a curve-fitting approach, which makes for nicer compound curves. This added a lot of complexity and is _not_ a _must-have_.
- **Train-test split**: Since some of the transformation are fit to the healthy and disease data, whereas the treatment data is _not_, the prod implementation uses a (repeated) train-test split to make healthy, disease, and treatment more comparable. This adds quite a bit of complexity. In discussing this with @drpeterfoster and @jsrv , it was determined that this was _not_ a must have.

## Why
- Rescue screens are an interesting application area for HOOKE. We stopped using rescue screens as our main screening approach at RXRX due to its combinatorial scaling, but the scalability of an in-silico method like our Virtual Cell, with the lab to validate, could make this an interesting approach again.
- The reason for using a stand-alone script, rather than the Prefect flow directly is so that we have more control over it (e.g. to adapt it for Transcriptomics).

## Integration Test

- Integration test:
  - Experiment ID `VHL-Core01-H-C700a`
  - Compounds: `REC-0064744` & `REC-0001788`

> [!NOTE] 
> We don't expect a perfect reproduction. We might even be using a different embedding space here. A perfect reproduction is not needed to demonstrate potential in this application area. To quote @drpeterfoster : "it’s impressive that it looks _that_ similar, considering how sensitive these plots are to the transform operation and choice of the disease vector"

Ground Truth (DART)         |   Reproduced (Ours)
:-------------------------:|:-------------------------:
<img width="687" height="599" alt="image" src="https://github.com/user-attachments/assets/ab438bff-eea1-415c-98a9-d5d534cb45c5" />  | <img width="650" height="650" alt="image" src="https://github.com/user-attachments/assets/e60b91c5-4092-44c0-a730-64991214e41f" />

To reproduce:
```
uv run vcb/metrics/drugscreen/rescue_screen.py 
```
(Note: This will show the curves for 8 compounds, not just the two cherry-picked ones above)

## Further Reading
- [Bridge course for a conceptual understanding](https://recursion.bridgeapp.com/learner/programs/9b142417/enroll)
- [The production-ready flow](https://github.com/recursionpharma/platform-analysis-flows/blob/trunk/flows/rescue_screen.py)
- [Original, now archived implementation](https://github.com/recursionpharma/rxrx-grimoire/blob/trunk/grimoire/flows/drugscreen.py)
- [Minimal implementation in RxRx19a](https://github.com/recursionpharma/rxrx19a-metrics/blob/d8f8c452476e0ae08ee889c2e9bd8e0fa29db4e4/rxrx19a_metrics/benchmark.py#L24)
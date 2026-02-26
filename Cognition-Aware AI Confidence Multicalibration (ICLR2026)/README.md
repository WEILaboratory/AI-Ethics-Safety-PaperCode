## Repository Structure

- `human_ai_interactions_data/`: Contains the dataset and corresponding data processing code. [1]
- `fair_hai.py`: The main code that runs the experiments.
- `output.ipynb`: Jupyter notebook for visualizing and analyzing the results.

## Running the Experiment

To run the experiment with different calibrations, use the following command:

**1. Run with original multicalibration:**

```{r, engine='bash'}
    python3 fair_hai.py
```

**2. Run with group-level multicalibration:**

```{r, engine='bash'}
    python3 fair_hai.py --method fair
```

**Output:**

- The output consists of AI-assisted decision-making outcomes for all tasks, calibrated using the specified method.
- The results will be saved in two files: `before_calibration_results.pkl` and `after_calibration_results.pkl` for the original multicalibration method,
  `before_fair_calibration_results.pkl` and `after_fair_calibration_results.pkl` for the group-level multicalibration.`

[1] Vodrahalli, Kailas, et al. "Do humans trust advice more if it comes from ai? an analysis of human-ai interactions." Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society. 2022.

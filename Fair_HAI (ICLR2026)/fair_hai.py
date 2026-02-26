import argparse
from human_ai_interactions_data import haiid
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import pickle

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.6f}".format
path = "plots/"


class Experiment:
    """
    Experiment class for simulating and evaluating human-AI interactions in decision-making tasks.
    This class is modified on the baseline-framework provided in:
    https://github.com/Networks-Learning/human-aligned-calibration
    """
    def __init__(self, name, h_bins, b_bins, random_state):
        self.task_name = name 
        self.h_bins = h_bins # Number of bins for discretizing human confidence levels
        self.b_bins = b_bins  # Number of bins for discretizing AI confidence levels
        self.h_bin_boundaries = []
        self.min_datapoints = 50 #  Minimum data points required in a cell for statistical significance
        self.task_data = None
        self.df_cell_prob = None # Overall probability of positive labels for all human decision-makers in each (h, b) cell
        self.prob_y1_h_b = None # Group-wise probability of positive labels in each (h, b) cell
        self.mass_y1_h_b = None  # Group-wise count of positive labels in each (h, b) cell
        self.decision_model = None # Decision policy π modeled using an MLP
        self.df_metrics = None # DataFrame containing evaluation results
        
        self.set_task_data()
        self.generate_decision_model(random_state)
        self.compute_cell_prob_matrix()
        self.run_experiment()


    def fair_aware_multicalibration(
        self,  
        alpha,
    ):
        """
        Implements group-level multicalibration using λ-discretization.

        Args:
            alpha (float): Calibration tolerance threshold.
        """
        lambda_param = 1 / self.b_bins
        df = self.task_data
        
        # Perform alpha-multicalibration for the "male" group
        update = True
        all_b_bins = np.sort(df["b_bin"].unique())
        all_h_bins = np.sort(df["h_bin"].unique())
        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())
            df_male = df[df["gender"] == "male"]
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_male_h_bin = df_male[df_male["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_male_h_bin_b_bin = df_male[
                        (df_male["h_bin"] == h_bin) & (df_male["b_bin"] == b_bin)
                    ]
                    if (
                        len(df_male_h_bin_b_bin)
                        <= alpha * lambda_param * len(df_male_h_bin)
                        or len(df_male_h_bin_b_bin) <= self.min_datapoints // 2
                    ):
                        continue
                    b_mean = df_male_h_bin_b_bin["b"].mean()
                    r_value = self.prob_y1_h_b[1].iat[b_index, h_index]
                    if abs(r_value - b_mean) > alpha:
                        update = True
                        df.loc[df_male_h_bin_b_bin.index, "b"] = df_male_h_bin_b_bin[
                            "b"
                        ].apply(lambda x: min(max(x + (r_value - b_mean), 0), 1))

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)
            
            #  Update group-wise probabilities of positive labels (P(Y=1)) for each (h, b) cell
            self.prob_y1_h_b = [
                df[df["gender"] == "female"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
                df[df["gender"] == "male"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            ]

        df_male = df[df["gender"] == "male"]
        for b_index, b_bin in enumerate(b_bin_values):
            df_male_b_bin = df_male[df_male["b_bin"] == b_bin]
            b_mean = df_male_b_bin["b"].mean()
            df.loc[df_male_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        #  Update group-wise probabilities of positive labels (P(Y=1)) for each (h, b) cell
        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
        ]
        # Perform alpha-multicalibration for the "female" group
        update = True
        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())

            df_female = df[df["gender"] == "female"]
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_female_h_bin = df_female[df_female["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_female_h_bin_b_bin = df_female[
                        (df_female["h_bin"] == h_bin) & (df_female["b_bin"] == b_bin)
                    ]
                    if (
                        len(df_female_h_bin_b_bin)
                        <= alpha * lambda_param * len(df_female_h_bin)
                        or len(df_female_h_bin_b_bin) <= self.min_datapoints // 2
                    ):
                        continue
                    b_mean = df_female_h_bin_b_bin["b"].mean()
                    r_value = self.prob_y1_h_b[0].iat[b_index, h_index]

                    if abs(r_value - b_mean) > alpha:
                        update = True
                        df.loc[df_female_h_bin_b_bin.index, "b"] = (
                            df_female_h_bin_b_bin["b"].apply(
                                lambda x: min(max(x + (r_value - b_mean), 0), 1)
                            )
                        )

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)
            #  Update group-wise probabilities of positive labels (P(Y=1)) for each (h, b) cell
            self.prob_y1_h_b = [
                df[df["gender"] == "female"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
                df[df["gender"] == "male"]
                .groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            ]

        df_female = df[df["gender"] == "female"]
        for b_index, b_bin in enumerate(b_bin_values):
            df_female_b_bin = df_female[df_female["b_bin"] == b_bin]
            b_mean = df_female_b_bin["b"].mean()
            df.loc[df_female_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)
        #  Update group-wise probabilities of positive labels (P(Y=1)) for each (h, b) cell
        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0),
        ]

        # update decision model and h+AI column
        df.loc[df["gender"] == "female", "h+AI"] = self.decision_model.predict(
            df[df["gender"] == "female"][["h", "b"]]
        )
        df.loc[df["gender"] == "male", "h+AI"] = self.decision_model.predict(
            df[df["gender"] == "male"][["h", "b"]]
        )
        self.task_data = df

    def multicalibration(self, alpha):
        """
        Implements  multicalibration using λ-discretization.
        Args:
            alpha (float): Calibration tolerance threshold.
        Reference:
            Hébert-Johnson, Ursula, et al. "Multicalibration: Calibration for the (computationally-identifiable) masses." International Conference on Machine Learning. PMLR, 2018.
        """
        lambda_param = 1 / self.b_bins
        df = self.task_data

        update = True
        all_h_bin_values = np.sort(df["h_bin"].unique())
        all_b_bin_values = np.sort(df["b_bin"].unique())

        while update:
            h_bin_values = np.sort(df["h_bin"].unique())
            b_bin_values = np.sort(df["b_bin"].unique())
            update = False
            # Iterate over each combination of h_bin and b_bin
            for h_index, h_bin in enumerate(h_bin_values):
                df_h_bin = df[df["h_bin"] == h_bin]
                for b_index, b_bin in enumerate(b_bin_values):
                    # Filter the DataFrame for the current combination of h_bin and b_bin
                    df_h_bin_b_bin = df[(df["h_bin"] == h_bin) & (df["b_bin"] == b_bin)]
                    if (
                        len(df_h_bin_b_bin) <= alpha * lambda_param * len(df_h_bin)
                        or len(df_h_bin_b_bin) <= self.min_datapoints
                    ):
                        continue
                    b_mean = df_h_bin_b_bin["b"].mean()
                    r_value = self.df_cell_prob.iat[b_index, h_index]
                    if abs(r_value - b_mean) > alpha:
                        update = True
                        df.loc[df_h_bin_b_bin.index, "b"] = df_h_bin_b_bin["b"].apply(
                            lambda x: min(max(x + (r_value - b_mean), 0), 1)
                        )

            df["b_bin"] = (df["b"] // lambda_param) + 1
            df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
            df["b_bin"] *= lambda_param
            df["b_bin"] = df["b_bin"].round(3)

            self.df_cell_prob = (
                df.groupby(by=["b_bin", "h_bin"])["y"]
                .mean()
                .unstack()
                .reindex(index=all_b_bin_values, columns=all_h_bin_values, fill_value=0)
            )

        for b_index, b_bin in enumerate(b_bin_values):
            df_b_bin = df[df["b_bin"] == b_bin]
            b_mean = df_b_bin["b"].mean()
            df.loc[df_b_bin.index, "b"] = b_mean

        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        self.df_cell_prob = (
            df.groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bin_values, columns=all_h_bin_values, fill_value=0)
        )

        # update decision model and h+AI column
        df.loc[df["gender"] == "female", "h+AI"] = self.decision_model.predict(
            df[df["gender"] == "female"][["h", "b"]]
        )
        df.loc[df["gender"] == "male", "h+AI"] = self.decision_model.predict(
            df[df["gender"] == "male"][["h", "b"]]
        )

        self.task_data = df

    def generate_decision_model(self, random_state):
        df = self.task_data
        X = df[["h", "b"]]
        y = df["h+AI"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=random_state
        )

        model = MLPClassifier(
            hidden_layer_sizes=(20),
            learning_rate_init=0.1,
            max_iter=500,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("decision model fitting accuracy:", accuracy)
        self.decision_model = model


   
    def set_task_data(self):
        """
        Loads and preprocesses the dataset for the specified task.

        Returns:
            DataFrame: The preprocessed dataset for the specified task.
        """
        # load all data
        df = haiid.load_dataset("./human_ai_interactions_data")
        # get specified task data
        task_df = haiid.load_task(df, self.task_name)
        if self.task_name == "census":
            self.task_data = self.preprocess_data(task_df, ">=50k")
        elif self.task_name == "sarcasm":
            self.task_data = self.preprocess_data(task_df, "sarcasm")
        else:
            self.task_data = self.preprocess_data(task_df)

        self.discretize_human_estimates()

        return self.task_data

    
    def preprocess_data(self, data, label_1=None):
        # filter data for 'geographic region' and 'perceived accuracy'
        df = data.loc[
            (data["geographic_region"] == "United States")
            & (data["perceived_accuracy"] == 80)
        ]
        #  Filter out rows with missing 'gender' information
        df = df[df["gender"] != "prefer not to say"]
        # select relevant data columns
        df = df[
            [
                "task_instance_id",
                "participant_id",
                "correct_label",
                "advice",
                "response_1",
                "response_2",
                "gender",
            ]
        ]
        # assign event Y=1
        if label_1 == None:
            # assigning event Y=1 to a random label per task instance
            # randomness needed since there are 4 labels accross tasks, but each task has two labels -> assign one of them to event Y=1
            np.random.seed(320)
            task_ids = df["task_instance_id"].unique()
            df_label = pd.DataFrame(task_ids, columns=["task_instance_id"])
            df_label["y"] = np.random.choice(2, len(task_ids)).astype(int)
            df = df.merge(df_label, how="left", on="task_instance_id")
        else:
            # assigning event Y=1 to label specified by label_1
            # The code is creating a new column "y" in the DataFrame `df` where the values are 1 if the
            # value in the column "correct_label" is equal to `label_1`, and 0 otherwise.
            df["y"] = (df["correct_label"] == label_1).astype(int)

        # map outcomes from [-1,1] to [0,1]
        df[["b", "h", "h+AI"]] = (
            df.loc[:, ["advice", "response_1", "response_2"]] + 1
        ) / 2.0
        df.loc[df["y"] == 0, ["b", "h", "h+AI"]] = (
            1 - df.loc[df["y"] == 0, ["b", "h", "h+AI"]]
        )
        # map final decisions to binary values: 0 if < 0.5, otherwise 1
        df["h+AI"] = np.where(df["h+AI"] < 0.5, 0, 1)
        return df[
            ["task_instance_id", "participant_id", "h", "b", "y", "h+AI", "gender"]
        ]
        
    
    def discretize_human_estimates(self):
        """
        Discretizes human confidence estimates into a specified number of bins (n_bins)  with approximately equal mass (number of data points in each bin).
        """
        # sort calibration data by human estimate h ascending
        df = self.task_data.copy()
        df = df.sort_values(by=["h"])

        # split calibration data into uniform sized bins
        # find bin boundaries
        if self.h_bins == 2:
            bin_bounds = [0.5, 1]
        else:
            df_split = np.array_split(df, self.h_bins)
            # find maximum value of h in each bin
            bin_bounds = []
            for df_h in df_split:
                max_h = round(df_h.loc[:, "h"].max(), 2)
                bin_bounds.append(max_h)

        # set value of h in calibration data to maximum value in each bin
        df.loc[
            (df["h"] < bin_bounds[0]) | np.isclose(df["h"], bin_bounds[0]), ["h_bin"]
        ] = bin_bounds[0]
        for i in range(1, len(bin_bounds)):
            df.loc[
                ((df["h"] < bin_bounds[i]) | np.isclose(df["h"], bin_bounds[i]))
                & (df["h"] > bin_bounds[i - 1]),
                ["h_bin"],
            ] = bin_bounds[i]

        # set data and h bin boundaries
        self.task_data = df
        self.h_bin_boundaries = bin_bounds

    
    # Computes the probabilities (P(Y=1)) and density mass for each bin (h, b),considering both the overall data and subgroup-specific distributions (e.g., gender-based groups here).
    def compute_cell_prob_matrix(self):
        df = self.task_data
        lambda_param = 1 / self.b_bins

        # partition model risk estimate space into B bins,
        # set new column in df with the max bin value (indicates in which bin each data point is))
        df["b_bin"] = (df["b"] // lambda_param) + 1
        df.loc[df["b_bin"] == self.b_bins + 1, ["b_bin"]] = self.b_bins
        df["b_bin"] *= lambda_param
        df["b_bin"] = df["b_bin"].round(3)

        # compute probability Y=1 and density mass in each bin
        all_b_bins = np.sort(df["b_bin"].unique())
        all_h_bins = np.sort(df["h_bin"].unique())

        self.df_cell_prob = (
            df.groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0)
        )

        self.prob_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .mean()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
        ]
        self.mass_y1_h_b = [
            df[df["gender"] == "female"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .count()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
            df[df["gender"] == "male"]
            .groupby(by=["b_bin", "h_bin"])["y"]
            .count()
            .unstack()
            .reindex(index=all_b_bins, columns=all_h_bins, fill_value=0)
            .fillna(0),
        ]



    # computes expected and average alignment error
    def check_alignment(self):
        # compute max and average alignment violations
        max_aligment = 0.0
        sum_aligment = 0.0
        max_aligment_s = 0.0
        sum_aligment_s = 0.0
        num_summants = 0.0
        disaligned_cells = set({})
        disaligned_cells_s = set({})

        for h_0_index in range(self.prob_y1_h_b[0].columns.shape[0] - 1):
            for b_0_index in range(self.prob_y1_h_b[0].index.shape[0] - 1):
                for h_prime_0_index in range(
                    h_0_index, self.prob_y1_h_b[0].columns.shape[0]
                ):
                    for b_prime_0_index in range(
                        b_0_index, self.prob_y1_h_b[0].index.shape[0]
                    ):
                        for h_1_index in range(h_0_index, h_0_index + 1):
                            for h_prime_1_index in range(
                                h_prime_0_index, h_prime_0_index + 1
                            ):
                                for b_1_index in range(b_0_index, b_0_index + 1):
                                    for b_prime_1_index in range(
                                        b_prime_0_index, b_prime_0_index + 1
                                    ):
                                        num_summants += 1
                                        # check misalignment of the pair of cells if enough datapoints in each cells
                                        if (
                                            self.mass_y1_h_b[0].iat[
                                                b_prime_0_index, h_prime_0_index
                                            ]
                                            + self.mass_y1_h_b[1].iat[
                                                b_prime_1_index, h_prime_1_index
                                            ]
                                            >= self.min_datapoints
                                            and self.mass_y1_h_b[0].iat[
                                                b_0_index, h_0_index
                                            ]
                                            + self.mass_y1_h_b[1].iat[
                                                b_1_index, h_1_index
                                            ]
                                            >= self.min_datapoints
                                        ):
                                            alignment = max(
                                                0.0,
                                                self.df_cell_prob.iat[
                                                    b_0_index, h_0_index
                                                ]
                                                - self.df_cell_prob.iat[
                                                    b_prime_0_index, h_prime_0_index
                                                ],
                                            )

                                            alignment2 = max(
                                                0.0,
                                                (
                                                    self.prob_y1_h_b[0].iat[
                                                        b_0_index, h_0_index
                                                    ]
                                                    * self.mass_y1_h_b[0].iat[
                                                        b_0_index, h_0_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_0_index, h_0_index
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_0_index, h_0_index
                                                        ]
                                                    )
                                                    + self.prob_y1_h_b[1].iat[
                                                        b_1_index, h_1_index
                                                    ]
                                                    * self.mass_y1_h_b[1].iat[
                                                        b_1_index, h_1_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_1_index, h_1_index
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_1_index, h_1_index
                                                        ]
                                                    )
                                                )
                                                - (
                                                    self.prob_y1_h_b[0].iat[
                                                        b_prime_0_index, h_prime_0_index
                                                    ]
                                                    * self.mass_y1_h_b[0].iat[
                                                        b_prime_0_index, h_prime_0_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_prime_0_index,
                                                            h_prime_0_index,
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_prime_0_index,
                                                            h_prime_0_index,
                                                        ]
                                                    )
                                                    + self.prob_y1_h_b[1].iat[
                                                        b_prime_1_index, h_prime_1_index
                                                    ]
                                                    * self.mass_y1_h_b[1].iat[
                                                        b_prime_1_index, h_prime_1_index
                                                    ]
                                                    / (
                                                        self.mass_y1_h_b[0].iat[
                                                            b_prime_1_index,
                                                            h_prime_1_index,
                                                        ]
                                                        + self.mass_y1_h_b[1].iat[
                                                            b_prime_1_index,
                                                            h_prime_1_index,
                                                        ]
                                                    )
                                                ),
                                            )

                                            max_aligment = max(max_aligment, alignment)
                                            if alignment > 0.0:
                                                sum_aligment += alignment
                                                disaligned_cells |= {
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_1_index
                                                        ],
                                                    ),
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_prime_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_prime_1_index
                                                        ],
                                                    ),
                                                }

                                            # case 1
                                        if (
                                            self.mass_y1_h_b[0].iat[
                                                b_prime_0_index, h_prime_0_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[1].iat[
                                                b_prime_1_index, h_prime_1_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[0].iat[
                                                b_0_index, h_0_index
                                            ]
                                            >= self.min_datapoints // 2
                                            and self.mass_y1_h_b[1].iat[
                                                b_1_index, h_1_index
                                            ]
                                            >= self.min_datapoints // 2
                                        ):
                                            alignment_s = abs(
                                                self.prob_y1_h_b[1].iat[
                                                    b_1_index, h_1_index
                                                ]
                                                - self.prob_y1_h_b[0].iat[
                                                    b_0_index, h_0_index
                                                ]
                                            ) + abs(
                                                -self.prob_y1_h_b[1].iat[
                                                    b_prime_1_index,
                                                    h_prime_1_index,
                                                ]
                                                + self.prob_y1_h_b[0].iat[
                                                    b_prime_0_index,
                                                    h_prime_0_index,
                                                ]
                                            )
                                            max_aligment_s = max(
                                                max_aligment_s, alignment_s
                                            )
                                            if alignment_s > 0:
                                                sum_aligment_s += alignment_s
                                                disaligned_cells_s |= {
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_1_index
                                                        ],
                                                    ),
                                                    (
                                                        self.prob_y1_h_b[0].index[
                                                            b_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[0].columns[
                                                            h_prime_0_index
                                                        ],
                                                        self.prob_y1_h_b[1].index[
                                                            b_prime_1_index
                                                        ],
                                                        self.prob_y1_h_b[1].columns[
                                                            h_prime_1_index
                                                        ],
                                                    ),
                                                }

        if num_summants > 0:
            avg_alignment = sum_aligment / num_summants
            avg_alignment_s = sum_aligment_s / num_summants
        else:
            avg_alignment = 0.0
            avg_alignment_s = 0.0

        self.disaligned_cells = disaligned_cells
        self.disaligned_cells_s = disaligned_cells_s

        mae = max_aligment
        eae = avg_alignment

        mae_s = max_aligment_s
        eae_s = avg_alignment_s

        return (eae, mae, eae_s, mae_s)

    def run_experiment(self):
        # compute expected/maximum alignment error and expected/maximum inter-group alignment error
        eae, mae, eiae, miae = self.check_alignment()

        acc_hAI = accuracy_score(
            self.task_data["y"],
            (self.task_data["h+AI"]).astype(int),
        )

        acc_h = accuracy_score(
            self.task_data["y"], (self.task_data["h"] >= 0.5).astype(int)
        )

        acc_b = accuracy_score(
            self.task_data["y"], (self.task_data["b"] >= 0.5).astype(int)
        )

        acc_h_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["h"] >= 0.5).astype(
                int
            ),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h"] >= 0.5).astype(
                int
            ),
        )

        acc_hAI_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["h+AI"]),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["h+AI"]),
        )

        acc_b_disp = accuracy_score(
            self.task_data[self.task_data["gender"] == "female"]["y"],
            (self.task_data[self.task_data["gender"] == "female"]["b"] >= 0.5).astype(
                int
            ),
        ) - accuracy_score(
            self.task_data[self.task_data["gender"] == "male"]["y"],
            (self.task_data[self.task_data["gender"] == "male"]["b"] >= 0.5).astype(
                int
            ),
        )

        dict_metrics = {
            "EAE": [round(eae, 4)],
            "MAE": [round(mae, 4)],
            "EIAE": [round(eiae, 4)],
            "MIAE": [round(miae, 4)],
            "acc_b": [acc_b],
            "acc_h": [acc_h],
            "acc_h+AI": [acc_hAI],
            "acc_b_disp": [acc_b_disp],
            "acc_h_disp": [acc_h_disp],
            "acc_h+AI_disp": [acc_hAI_disp],
        }
        self.df_metrics = pd.DataFrame(dict_metrics, index=[self.task_name]).round(4)

    def get_task_data(self):
        return self.task_data

    def get_metrics(self):
        return self.df_metrics

def main(args):
    before_calibration_results = []
    after_calibration_results = []
    for k in range(20):
        print(k, "-th iterations:\n")
        exp_art = Experiment("art", h_bins=3, b_bins=args.b_bins, random_state=k)
        exp_sarcasm = Experiment("sarcasm", h_bins=3, b_bins=args.b_bins, random_state=k)
        exp_cities = Experiment("cities", h_bins=3, b_bins=args.b_bins, random_state=k)
        exp_census = Experiment("census", h_bins=3, b_bins=args.b_bins, random_state=k)

        df_results = exp_art.get_metrics()
        df_results = pd.concat([df_results, exp_sarcasm.get_metrics()])
        df_results = pd.concat([df_results, exp_cities.get_metrics()])
        df_results = pd.concat([df_results, exp_census.get_metrics()])

        before_calibration_results.append(df_results)
        print("Before Calibration:\n", df_results)

         # Setting the method to use for calibration
        if args.method == 'fair':
            # Apply group_level_multicalibration
            exp_art.fair_aware_multicalibration(alpha=1.0/args.b_bins)
            exp_sarcasm.fair_aware_multicalibration(alpha=1.0/args.b_bins)
            exp_cities.fair_aware_multicalibration(alpha=1.0/args.b_bins)
            exp_census.fair_aware_multicalibration(alpha=1.0/args.b_bins)
        else:
            # Apply  multicalibration
            exp_art.multicalibration(alpha=1.0/args.b_bins)
            exp_sarcasm.multicalibration(alpha=1.0/args.b_bins)
            exp_cities.multicalibration(alpha=1.0/args.b_bins)
            exp_census.multicalibration(alpha=1.0/args.b_bins)

        # Update the cell probability matrix and run the experiment
        exp_art.compute_cell_prob_matrix()
        exp_art.run_experiment()

        exp_sarcasm.compute_cell_prob_matrix()
        exp_sarcasm.run_experiment()

        exp_cities.compute_cell_prob_matrix()
        exp_cities.run_experiment()

        exp_census.compute_cell_prob_matrix()
        exp_census.run_experiment()

        # Collect the results after calibration
        df_results = exp_art.get_metrics()
        df_results = pd.concat([df_results, exp_sarcasm.get_metrics()])
        df_results = pd.concat([df_results, exp_cities.get_metrics()])
        df_results = pd.concat([df_results, exp_census.get_metrics()])

        after_calibration_results.append(df_results)
        print("After Calibration:\n", df_results)

    # Save results to files with conditional naming
    before_file_name = "before_{}_calibration_results_lambda_{}.pkl".format(args.method,args.b_bins) 
    after_file_name = "after_{}_calibration_results_lambda_{}.pkl".format(args.method,args.b_bins) 

    with open(before_file_name, "wb") as f:
        pickle.dump(before_calibration_results, f)

    with open(after_file_name, "wb") as f:
        pickle.dump(after_calibration_results, f)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with calibration methods")
    parser.add_argument('--method', choices=['fair', 'standard'], default='standards', 
                        help="Specify the calibration method: 'fair' for group-level multicalibration, 'standard' for multicalibration")
    parser.add_argument('--b_bins',  default=8, type=int, help="Number of discretization bins for AI confidence")
    args = parser.parse_args()

    # Run the main function with the selected method
    main(args)

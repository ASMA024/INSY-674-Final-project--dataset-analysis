# Telco Customer Churn: Who Should Get a Retention Offer?

INSY 674 Final Project

All the code is in `Telco_Churn_Project.ipynb`. The notebook follows the same five steps as this README, so each section here points to the matching part of the notebook.

## How to run it

1. Open the notebook in Jupyter or VS Code with a Python 3 environment that has pandas, scikit-learn, matplotlib and seaborn (Anaconda works).
2. Run the first cell once. It installs `kagglehub`, `shap` and `imbalanced-learn`.
3. Run all cells. The data (IBM Telco Customer Churn on Kaggle, `blastchar/telco-customer-churn`) downloads automatically. A full run takes about five minutes on a laptop.

The trained model is saved to `models/churn_model.joblib`, and `score_customers()` at the end of the notebook scores new customers.

## 1. Frame the Problem and Look at the Big Picture

A telecom company loses revenue every time a customer cancels. The retention team can send offers, but each offer costs money, so they need to know who to target. Our model ranks every customer by churn risk, and we then pick the cut-off that gives the lowest total cost.

This is a supervised binary classification problem. The label `Churn` says whether the customer left in the last month, and 26.5% did. The model would run as a monthly batch job on the full customer list, so it does not need to work in real time.

We use two kinds of metric. To compare models we use PR-AUC, because churners are the minority class and PR-AUC measures how well a model ranks them without depending on a threshold. To make the final decision we use expected cost in dollars, because missing a churner costs much more than sending an offer that wasn't needed. Our cost assumptions are $50 per offer, a 30% chance that an offer keeps a churner, and 12 months of `MonthlyCharges` as the revenue lost when someone leaves. These numbers are examples, and the notebook shows how the decision changes if they change.

Our baselines are a dummy model that always predicts "stay" and a simple business rule: month-to-month contract and tenure under 12 months means the customer will churn.

## 2. Explore the Data

We split off 20% of the customers as a test set first and only explored the training set. The dataset has 7,043 customers and 21 columns, mostly categorical. The only real data-quality issue is `TotalCharges`, which is stored as text and has 11 blanks. All 11 are customers with zero tenure who have not been billed yet, so we set them to 0. Values like "No internet service" are not missing data. They repeat information already in `InternetService`, so we merged them into "No".

The main things we learned:

- Contract type is the clearest signal. 42.7% of month-to-month customers churn, compared with 2.9% of two-year customers.
- The first year is the riskiest, and churn drops steadily as tenure grows.
- Fiber optic customers churn at 42.1%, compared with 18.7% for DSL, even though fiber is the premium service.
- Customers paying by electronic check churn at 45.7%, roughly three times the rate of customers on automatic payment.
- Customers with online security or tech support churn much less.
- `TotalCharges` is strongly correlated with `tenure` (r = 0.83), so the two overlap.

Data we don't have but that would help: dated monthly snapshots, contract end dates, customer service calls and complaints, past offers and their results, and competitor prices.

## 3. Prepare the Data

All feature engineering is in a custom scikit-learn transformer inside the pipeline, so during cross-validation every step is fit only on the training folds and nothing leaks. We added four features: the number of add-on services, a price-change measure (`TotalCharges − MonthlyCharges × tenure`), tenure bands, and an automatic-payment flag. Numeric columns are standardized, Yes/No columns become 0/1, `Contract` is ordinal encoded because its order matters, and the other categorical columns are one-hot encoded.

We tested a "lean" version without the redundant columns and a version without our engineered features. PR-AUC moved by 0.003 at most, which is smaller than the fold-to-fold noise. We kept the full set because the engineered features are easy to explain and we use them again in the error analysis.

For the class imbalance, we compared doing nothing, `class_weight="balanced"` and SMOTE (inside an imbalanced-learn pipeline, so it does not leak). All three have almost the same PR-AUC. What changes is recall versus precision at the default 0.5 threshold. Since we choose our own threshold later, we kept `class_weight="balanced"` because it is the simplest.

## 4. Shortlist Promising Models

We compared eight approaches with 5-fold stratified cross-validation on the training set: the dummy model, the rule, logistic regression, naive Bayes, KNN, a decision tree, a random forest and gradient boosting. Logistic regression had the best PR-AUC (0.663), followed by random forest (0.653) and gradient boosting (0.648). The gaps between the three were smaller than one standard deviation. The rule only reached 0.397. The random forest and boosting models scored around 0.86–0.90 on the training folds but only 0.65 on validation, so they were overfitting. A validation curve for the decision tree shows the same thing.

We used PR-AUC and not recall to choose models. Recall at the 0.5 threshold mostly reflects the threshold and the class weights, not how well a model ranks customers. The same logistic regression goes from 54% to 80% recall just by adding class weights, while its PR-AUC stays the same. Recall belongs to the threshold decision, which comes later.

## 5. Fine-Tune the System

We tuned the top three models with `RandomizedSearchCV`, because the tree models have many hyperparameters with wide ranges and random sampling covers them more cheaply than a full grid. Tuning mainly helped the tree ensembles, and after tuning the three models were within 0.003 PR-AUC of each other. A soft-voting ensemble of all three reached 0.667. Since that is within 0.005 of logistic regression (0.664), we chose logistic regression: it performs the same and is much easier to explain.

We then chose the threshold that minimized total cost on out-of-fold training predictions. The best threshold was 0.45, which sends offers to about 45% of customers and catches about 83% of churners. We also show what happens if the business requires a minimum recall. Up to about 85% recall it costs almost nothing extra, but 97% recall means contacting 72% of customers and roughly $5,700 more per 1,000 customers. A sensitivity table shows the best threshold moving from 0.18 to 0.76 depending on the offer cost and success rate, so the business inputs matter as much as the model.

The error analysis was the most interesting part. The churners we catch look like the textbook profile: new, month-to-month, electronic check. The ones we miss look loyal: about 36 months of tenure on average, often on one- or two-year contracts, paying automatically and with more add-ons. Many of them are also high-value fiber customers. Nothing in this data tells the model they are unhappy, and a person couldn't tell either. Contract end dates and complaint history are what would help.

SHAP shows that internet type, tenure and contract type drive most predictions, and demographics, including gender, contribute the least.

**Final test result (used once, at the end).** On the 1,409 held-out customers, PR-AUC is 0.636, ROC-AUC 0.842 and recall 83%, all very close to the cross-validation numbers, so the model generalizes. Under our cost assumptions it saves about $36,800 per 1,000 customers, compared with $19,700 for the rule (87% more), and it captures 65% of the savings a perfect model would get. A 2,000-sample bootstrap puts the model's savings between about $31,000 and $42,400 per 1,000 customers, and the model beat the rule in every resample.

## Limitations

1. The data is a single one-month snapshot without dates, so some "false alarms" may churn later. The fix is monthly snapshots with a time-based split.
2. The model predicts who will churn, not who will stay because of an offer. Answering that needs an A/B test and an uplift model.
3. The dollar amounts are assumptions, and monthly charges are revenue, not profit. The sensitivity and recall-target tables show how the decision changes, and the team should plug in real numbers.
4. The model misses loyal-looking churners because their reasons for leaving aren't in the data. Contract end dates, service calls and complaints would help most.
5. This is an IBM sample dataset from one company and may be simplified. The model would need to be retrained on the company's own recent data.
6. With class weights, the predicted probabilities are higher than the real churn rates (see the calibration plot). This doesn't affect the ranking or the cost-based threshold, but the scores shouldn't be shown to managers as literal percentages without calibrating them first.

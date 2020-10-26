## Feature Selection / Filtering {#fs}




Often, data sets include a large number of features.
The technique of extracting a subset of relevant features is called "feature selection".

The objective of feature selection is to fit the sparse dependent of a model on a subset of available data features in the most suitable manner.
Feature selection can enhance the interpretability of the model, speed up the learning process and improve the learner performance.
Different approaches exist to identify the relevant features.
Two different approaches are emphasized in the literature:
one is called [Filtering](#fs-filtering) and the other approach is often referred to as feature subset selection or [wrapper methods](#fs-wrapper).

What are the differences [@chandrashekar2014]?

* **Filtering**: An external algorithm computes a rank of the variables (e.g. based on the correlation to the response).
  Then, features are subsetted by a certain criteria, e.g. an absolute number or a percentage of the number of variables.
  The selected features will then be used to fit a model (with optional hyperparameters selected by tuning).
  This calculation is usually cheaper than “feature subset selection” in terms of computation time.
* **Wrapper Methods**: Here, no ranking of features is done.
  Features are selected by a (random) subset of the data.
  Then, we fit a model and subsequently assess the performance.
  This is done for a lot of feature combinations in a cross-validation (CV) setting and the best combination is reported.
  This method is very computationally intensive as a lot of models are fitted.
  Also, strictly speaking all these models would need to be tuned before the performance is estimated.
  This would require an additional nested level in a CV setting.
  After undertaken all of these steps, the selected subset of features is again fitted (with optional hyperparameters selected by tuning).

There is also a third approach which can be attributed to the "filter" family:
The embedded feature-selection methods of some [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).
Read more about how to use these in section [embedded feature-selection methods](#fs-embedded).

[Ensemble filters](#fs-ensemble) built upon the idea of stacking single filter methods.
These are not yet implemented.

All functionality that is related to feature selection is implemented via the extension package [mlr3filters](https://mlr3filters.mlr-org.com).

### Filters {#fs-filter}

Filter methods assign an importance value to each feature.
Based on these values the features can be ranked.
Thereafter, we are able to select a feature subset.
There is a list of all implemented filter methods in the [Appendix](#list-filters).

### Calculating filter values {#fs-calc}

Currently, only classification and regression tasks are supported.

The first step it to create a new R object using the class of the desired filter method.
Each object of class `Filter` has a `.$calculate()` method which computes the filter values and ranks them in a descending order.


```r
library("mlr3filters")
filter = FilterJMIM$new()

task = tsk("iris")
filter$calculate(task)

as.data.table(filter)
```

```
##         feature  score
## 1: Sepal.Length 1.0401
## 2:  Petal.Width 0.9894
## 3: Petal.Length 0.9881
## 4:  Sepal.Width 0.8314
```

Some filters support changing specific hyperparameters.
This is similar to setting hyperparameters of a [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) using `.$param_set$values`:


```r
filter_cor = FilterCorrelation$new()
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default value
## 1: everything      
## 2:    pearson
```

```r
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default    value
## 1: everything         
## 2:    pearson spearman
```

Rather than taking the "long" R6 way to create a filter, there is also a built-in shorthand notation for filter creation:


```r
filter = flt("cmim")
filter
```

```
## <FilterCMIM:cmim>
## Task Types: classif, regr
## Task Properties: -
## Packages: praznik
## Feature types: integer, numeric, factor, ordered
```

### Variable Importance Filters {#fs-var-imp-filters}

All [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) with the property "importance" come with integrated feature selection methods.

You can find a list of all learners with this property in the [Appendix](#fs-filter-embedded-list).

For some learners the desired filter method needs to be set during learner creation.
For example, learner `classif.ranger` (in the package [mlr3learners](https://mlr3learners.mlr-org.com)) comes with multiple integrated methods.
See the help page of [`ranger::ranger`](https://www.rdocumentation.org/packages/ranger/topics/ranger).
To use method "impurity", you need to set the filter method during construction.


```r
library("mlr3learners")
lrn = lrn("classif.ranger", importance = "impurity")
```

Now you can use the [`mlr3filters::FilterImportance`](https://mlr3filters.mlr-org.com/reference/FilterImportance.html) class for algorithm-embedded methods to filter a [`Task`](https://mlr3.mlr-org.com/reference/Task.html).


```r
library("mlr3learners")

task = tsk("iris")
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
```

```
##         feature  score
## 1:  Petal.Width 45.687
## 2: Petal.Length 41.988
## 3: Sepal.Length  9.256
```

### Ensemble Methods {#fs-ensemble}

Work in progress.

### Wrapper Methods {#fs-wrapper}

Wrapper feature selection is supported via the [mlr3fselect](https://mlr3fselect.mlr-org.com) extension package.
At the heart of [mlr3fselect](https://mlr3fselect.mlr-org.com) are the R6 classes:

* [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html), [`FSelectInstanceMultiCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceMultiCrit.html): These two classes describe the feature selection problem and store the results.
* [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html): This class is the base class for implementations of feature selection algorithms.

### The `FSelectInstance` Classes {#fs-wrapper-optimization}

The following sub-section examines the feature selection on the [`Pima`](https://mlr3.mlr-org.com/reference/mlr_tasks_sonar.html) data set which is used to predict whether or not a patient has diabetes.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```
We use the classification tree from [rpart](https://cran.r-project.org/package=rpart).


```r
learner = lrn("classif.rpart")
```

Next, we need to specify how to evaluate the performance of the feature subsets.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to choose the available budget for the feature selection.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when feature selection does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html):


```r
library("mlr3fselect")

evals20 = trm("evals", n_evals = 20)

instance = FSelectInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  terminator = evals20
)
instance
```

```
## <FSelectInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveFSelect:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper      levels        default value
## 1:      age ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 2:  glucose ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 3:  insulin ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 4:     mass ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 5: pedigree ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 6: pregnant ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 7: pressure ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 8:  triceps ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveFSelect>
## Null data.table (0 rows and 0 cols)
```

To start the feature selection, we still need to select an algorithm which are defined via the [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) class

### The `FSelector` Class

The following algorithms are currently implemented in [mlr3fselect](https://mlr3fselect.mlr-org.com):

* Random Search ([`FSelectorRandomSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorRandomSearch.html))
* Exhaustive Search ([`FSelectorExhaustiveSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorExhaustiveSearch.html))
* Sequential Search ([`FSelectorSequential`](https://mlr3fselect.mlr-org.com/reference/FSelectorSequential.html))
* Recursive Feature Elimination ([`FSelectorRFE`](https://mlr3fselect.mlr-org.com/reference/FSelectorRFE.html))
* Design Points ([`FSelectorDesignPoints`](https://mlr3fselect.mlr-org.com/reference/FSelectorDesignPoints.html))

In this example, we will use a simple random search.


```r
fselector = fs("random_search")
```

### Triggering the Tuning {#wrapper-selection-triggering}

To start the feature selection, we simply pass the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html). The algorithm proceeds as follows

1. The [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) proposes at least one feature subset and may propose multiple subsets to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each feature subset, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the feature subset with the best observed performance.
5. Store the best feature subset as the result in the instance object.
The best featue subset (`$result_feature_set`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
fselector$optimize(instance)
```

```
## INFO  [13:37:12.364] Starting to optimize 8 parameter(s) with '<FSelectorRandomSearch>' and '<TerminatorEvals>' 
## INFO  [13:37:12.400] Evaluating 10 configuration(s) 
## INFO  [13:37:14.882] Result of batch 1: 
## INFO  [13:37:14.885]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [13:37:14.885]   TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2734 
## INFO  [13:37:14.885]  FALSE   FALSE    TRUE FALSE    FALSE    FALSE     TRUE   FALSE     0.3945 
## INFO  [13:37:14.885]  FALSE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2773 
## INFO  [13:37:14.885]   TRUE    TRUE   FALSE FALSE     TRUE     TRUE    FALSE    TRUE     0.2695 
## INFO  [13:37:14.885]  FALSE    TRUE   FALSE  TRUE    FALSE    FALSE     TRUE    TRUE     0.2773 
## INFO  [13:37:14.885]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2656 
## INFO  [13:37:14.885]  FALSE   FALSE   FALSE FALSE     TRUE     TRUE    FALSE   FALSE     0.3125 
## INFO  [13:37:14.885]   TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3281 
## INFO  [13:37:14.885]   TRUE   FALSE   FALSE FALSE    FALSE     TRUE    FALSE   FALSE     0.3398 
## INFO  [13:37:14.885]  FALSE   FALSE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.3320 
## INFO  [13:37:14.885]                                 uhash 
## INFO  [13:37:14.885]  a0b784c9-8e3b-44a6-942d-0ce086abda35 
## INFO  [13:37:14.885]  634e1667-ebc0-4ab1-bff6-3c2d3ae9d752 
## INFO  [13:37:14.885]  91c11760-a98e-4867-ac7d-585648b3a776 
## INFO  [13:37:14.885]  0ac67fac-76b5-42d9-80fe-f83e70d22dd2 
## INFO  [13:37:14.885]  57e833b6-186f-4289-be26-c778a83b00e5 
## INFO  [13:37:14.885]  32bfd1ee-a405-40ef-9182-7e540106b663 
## INFO  [13:37:14.885]  8f9894d2-544a-414a-8816-9b2bfebb5a59 
## INFO  [13:37:14.885]  f2301010-ca51-4e92-b70d-173c0dab1610 
## INFO  [13:37:14.885]  8a43ff24-9fd3-46f8-9bf7-19a18b003fa4 
## INFO  [13:37:14.885]  cccdfe63-0929-4025-a5eb-d86d590556cf 
## INFO  [13:37:14.889] Evaluating 10 configuration(s) 
## INFO  [13:37:16.908] Result of batch 2: 
## INFO  [13:37:16.912]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [13:37:16.912]  FALSE    TRUE    TRUE FALSE    FALSE     TRUE    FALSE   FALSE     0.2773 
## INFO  [13:37:16.912]   TRUE   FALSE   FALSE  TRUE     TRUE    FALSE    FALSE   FALSE     0.3398 
## INFO  [13:37:16.912]  FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3477 
## INFO  [13:37:16.912]   TRUE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.2578 
## INFO  [13:37:16.912]   TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3281 
## INFO  [13:37:16.912]  FALSE    TRUE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE     0.2383 
## INFO  [13:37:16.912]  FALSE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2656 
## INFO  [13:37:16.912]   TRUE   FALSE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.3398 
## INFO  [13:37:16.912]   TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2695 
## INFO  [13:37:16.912]   TRUE    TRUE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.2617 
## INFO  [13:37:16.912]                                 uhash 
## INFO  [13:37:16.912]  ad0756ff-f338-4d3e-acab-8e7dad514f41 
## INFO  [13:37:16.912]  d552d816-52fd-4403-9eb2-95b253770ccb 
## INFO  [13:37:16.912]  d1976977-3058-4559-9615-1255895e240b 
## INFO  [13:37:16.912]  fa0d13a1-3e58-4995-8c99-ca509f26171d 
## INFO  [13:37:16.912]  ab4182ab-eb68-4cca-b8e6-0fb95c4f95a2 
## INFO  [13:37:16.912]  e5a01b98-3aba-49b4-b995-92ee280aa169 
## INFO  [13:37:16.912]  e278de0c-dd97-4fd5-9b93-710b7bcc7cf0 
## INFO  [13:37:16.912]  2da18bcb-4e93-4301-8616-9272e55362c0 
## INFO  [13:37:16.912]  1182c882-9cbf-4f97-b4d0-7a351a6ab777 
## INFO  [13:37:16.912]  1f493e1c-3693-43c7-8ea0-bf31924f7e77 
## INFO  [13:37:16.919] Finished optimizing after 20 evaluation(s) 
## INFO  [13:37:16.921] Result: 
## INFO  [13:37:16.923]    age glucose insulin  mass pedigree pregnant pressure triceps 
## INFO  [13:37:16.923]  FALSE    TRUE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE 
## INFO  [13:37:16.923]                           features  x_domain classif.ce 
## INFO  [13:37:16.923]  glucose,insulin,pedigree,pregnant <list[8]>     0.2383
```

```
##      age glucose insulin  mass pedigree pregnant pressure triceps
## 1: FALSE    TRUE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE
##                             features  x_domain classif.ce
## 1: glucose,insulin,pedigree,pregnant <list[8]>     0.2383
```

```r
instance$result_feature_set
```

```
## [1] "glucose"  "insulin"  "pedigree" "pregnant"
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2383
```
One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##       age glucose insulin  mass pedigree pregnant pressure triceps classif.ce
##  1:  TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2734
##  2: FALSE   FALSE    TRUE FALSE    FALSE    FALSE     TRUE   FALSE     0.3945
##  3: FALSE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2773
##  4:  TRUE    TRUE   FALSE FALSE     TRUE     TRUE    FALSE    TRUE     0.2695
##  5: FALSE    TRUE   FALSE  TRUE    FALSE    FALSE     TRUE    TRUE     0.2773
##  6:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2656
##  7: FALSE   FALSE   FALSE FALSE     TRUE     TRUE    FALSE   FALSE     0.3125
##  8:  TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3281
##  9:  TRUE   FALSE   FALSE FALSE    FALSE     TRUE    FALSE   FALSE     0.3398
## 10: FALSE   FALSE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.3320
## 11: FALSE    TRUE    TRUE FALSE    FALSE     TRUE    FALSE   FALSE     0.2773
## 12:  TRUE   FALSE   FALSE  TRUE     TRUE    FALSE    FALSE   FALSE     0.3398
## 13: FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3477
## 14:  TRUE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.2578
## 15:  TRUE   FALSE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3281
## 16: FALSE    TRUE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE     0.2383
## 17: FALSE    TRUE    TRUE  TRUE     TRUE     TRUE     TRUE    TRUE     0.2656
## 18:  TRUE   FALSE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.3398
## 19:  TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2695
## 20:  TRUE    TRUE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.2617
##                                    uhash  x_domain           timestamp batch_nr
##  1: a0b784c9-8e3b-44a6-942d-0ce086abda35 <list[8]> 2020-10-26 13:37:14        1
##  2: 634e1667-ebc0-4ab1-bff6-3c2d3ae9d752 <list[8]> 2020-10-26 13:37:14        1
##  3: 91c11760-a98e-4867-ac7d-585648b3a776 <list[8]> 2020-10-26 13:37:14        1
##  4: 0ac67fac-76b5-42d9-80fe-f83e70d22dd2 <list[8]> 2020-10-26 13:37:14        1
##  5: 57e833b6-186f-4289-be26-c778a83b00e5 <list[8]> 2020-10-26 13:37:14        1
##  6: 32bfd1ee-a405-40ef-9182-7e540106b663 <list[8]> 2020-10-26 13:37:14        1
##  7: 8f9894d2-544a-414a-8816-9b2bfebb5a59 <list[8]> 2020-10-26 13:37:14        1
##  8: f2301010-ca51-4e92-b70d-173c0dab1610 <list[8]> 2020-10-26 13:37:14        1
##  9: 8a43ff24-9fd3-46f8-9bf7-19a18b003fa4 <list[8]> 2020-10-26 13:37:14        1
## 10: cccdfe63-0929-4025-a5eb-d86d590556cf <list[8]> 2020-10-26 13:37:14        1
## 11: ad0756ff-f338-4d3e-acab-8e7dad514f41 <list[8]> 2020-10-26 13:37:16        2
## 12: d552d816-52fd-4403-9eb2-95b253770ccb <list[8]> 2020-10-26 13:37:16        2
## 13: d1976977-3058-4559-9615-1255895e240b <list[8]> 2020-10-26 13:37:16        2
## 14: fa0d13a1-3e58-4995-8c99-ca509f26171d <list[8]> 2020-10-26 13:37:16        2
## 15: ab4182ab-eb68-4cca-b8e6-0fb95c4f95a2 <list[8]> 2020-10-26 13:37:16        2
## 16: e5a01b98-3aba-49b4-b995-92ee280aa169 <list[8]> 2020-10-26 13:37:16        2
## 17: e278de0c-dd97-4fd5-9b93-710b7bcc7cf0 <list[8]> 2020-10-26 13:37:16        2
## 18: 2da18bcb-4e93-4301-8616-9272e55362c0 <list[8]> 2020-10-26 13:37:16        2
## 19: 1182c882-9cbf-4f97-b4d0-7a351a6ab777 <list[8]> 2020-10-26 13:37:16        2
## 20: 1f493e1c-3693-43c7-8ea0-bf31924f7e77 <list[8]> 2020-10-26 13:37:16        2
```

The associated resampling iterations can be accessed in the [`BenchmarkResult`](https://mlr3.mlr-org.com/reference/BenchmarkResult.html):


```r
instance$archive$benchmark_result$data
```

```
## <ResultData>
##   Public:
##     as_data_table: function (view = NULL, reassemble_learners = TRUE, convert_predictions = TRUE, 
##     clone: function (deep = FALSE) 
##     combine: function (rdata) 
##     data: list
##     initialize: function (data = NULL) 
##     iterations: function (view = NULL) 
##     learners: function (view = NULL, states = TRUE, reassemble = TRUE) 
##     logs: function (view = NULL, condition) 
##     prediction: function (view = NULL, predict_sets = "test") 
##     predictions: function (view = NULL, predict_sets = "test") 
##     resamplings: function (view = NULL) 
##     sweep: function () 
##     task_type: active binding
##     tasks: function (view = NULL, reassemble = TRUE) 
##     uhashes: function (view = NULL) 
##   Private:
##     deep_clone: function (name, value) 
##     get_view_index: function (view)
```

The `uhash` column links the resampling iterations to the evaluated feature subsets stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized feature subset can be used to subset the task and fit the model on all observations.


```r
task$select(instance$result_feature_set)
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during feature selection and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resampling) is required.

### Automating the Feature Selection {#autofselect}

The [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) wraps a learner and augments it with an automatic feature selection for a given task.
Because the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically starts a feature selection on the given task using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and and uses a simple random search as feature selection algorithm:


```r
library("paradox")
library("mlr3fselect")

learner = lrn("classif.rpart")
terminator = trm("evals", n_evals = 10)
fselector = fs("random_search")

at = AutoFSelector$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = terminator,
  fselector = fselector
)
at
```

```
## <AutoFSelector:classif.rpart.fselector>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the optimized feature subset to the complete feature set.
This way, the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) will do its resampling for feature selection on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during feature selection or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the optimized feature subset with the complete feature set, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlrfselect
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid, store_models = TRUE)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id              learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.fselector            cv     3
## 2:  2 <ResampleResult[21]>    pima           classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.3177          0
## 2:     0.2604          0
```

Note that we do not expect any significant differences since we only evaluated a small fraction of the possible feature subsets.

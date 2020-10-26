## Hyperparameter Tuning {#tuning}

Hyperparameters are second-order parameters of machine learning models that, while often not explicitly optimized during the model estimation process, can have an important impact on the outcome and predictive performance of a model.
Typically, hyperparameters are fixed before training a model.
However, because the output of a model can be sensitive to the specification of hyperparameters, it is often recommended to make an informed decision about which hyperparameter settings may yield better model performance.
In many cases, hyperparameter settings may be chosen _a priori_, but it can be advantageous to try different settings before fitting your model on the training data.
This process is often called model 'tuning'.

Hyperparameter tuning is supported via the [mlr3tuning](https://mlr3tuning.mlr-org.com) extension package.
Below you can find an illustration of the process:

<img src="images/tuning_process.svg" style="display: block; margin: auto;" />

At the heart of [mlr3tuning](https://mlr3tuning.mlr-org.com) are the R6 classes:

* [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html), [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html): These two classes describe the tuning problem and store the results.
* [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html): This class is the base class for implementations of tuning algorithms.

### The `TuningInstance` Classes {#tuning-optimization}

The following sub-section examines the optimization of a simple classification tree on the [`Pima Indian Diabetes`](https://mlr3.mlr-org.com/reference/mlr_tasks_pima.html) data set.


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

We use the classification tree from [rpart](https://cran.r-project.org/package=rpart) and choose a subset of the hyperparameters we want to tune.
This is often referred to as the "tuning space".


```r
learner = lrn("classif.rpart")
learner$param_set
```

```
## <ParamSet>
##                 id    class lower upper      levels        default value
##  1:       minsplit ParamInt     1   Inf                         20      
##  2:      minbucket ParamInt     1   Inf             <NoDefault[3]>      
##  3:             cp ParamDbl     0     1                       0.01      
##  4:     maxcompete ParamInt     0   Inf                          4      
##  5:   maxsurrogate ParamInt     0   Inf                          5      
##  6:       maxdepth ParamInt     1    30                         30      
##  7:   usesurrogate ParamInt     0     2                          2      
##  8: surrogatestyle ParamInt     0     1                          0      
##  9:           xval ParamInt     0   Inf                         10     0
## 10:     keep_model ParamLgl    NA    NA  TRUE,FALSE          FALSE
```

Here, we opt to tune two parameters:

* The complexity `cp`
* The termination criterion `minsplit`

The tuning space needs to be bounded, therefore one has to set lower and upper bounds:


```r
library("paradox")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
tune_ps
```

```
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>
```

Next, we need to specify how to evaluate the performance.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to select the budget available, to solve this tuning instance.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when tuning does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html):


```r
library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = tune_ps,
  terminator = evals20
)
instance
```

```
## <TuningInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveTuning:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveTuning>
## Null data.table (0 rows and 0 cols)
```

To start the tuning, we still need to select how the optimization should take place.
In other words, we need to choose the **optimization algorithm** via the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) class.

### The `Tuner` Class

The following algorithms are currently implemented in [mlr3tuning](https://mlr3tuning.mlr-org.com):

* Grid Search ([`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html))
* Random Search ([`TunerRandomSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_random_search.html)) [@bergstra2012]
* Generalized Simulated Annealing ([`TunerGenSA`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_gensa.html))
* Non-Linear Optimization ([`TunerNLoptr`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_nloptr.html))

In this example, we will use a simple grid search with a grid resolution of 5.


```r
tuner = tnr("grid_search", resolution = 5)
```

Since we have only numeric parameters, [`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html) will create an equidistant grid between the respective upper and lower bounds.
As we have two hyperparameters with a resolution of 5, the two-dimensional grid consists of $5^2 = 25$ configurations.
Each configuration serves as a hyperparameter setting for the previously defined [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and triggers a 3-fold cross validation on the task.
All configurations will be examined by the tuner (in a random order), until either all configurations are evaluated or the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) signals that the budget is exhausted.

### Triggering the Tuning {#tuning-triggering}

To start the tuning, we simply pass the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html).
The tuner proceeds as follows:

1. The [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) proposes at least one hyperparameter configuration (the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) may propose multiple points to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each configuration, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the configuration with the best observed performance.
5. Store the best configurations as result in the instance object.
   The best hyperparameter settings (`$result_learner_param_vals`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
tuner$optimize(instance)
```

```
## INFO  [13:37:02.873] Starting to optimize 2 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [13:37:02.911] Evaluating 1 configuration(s) 
## INFO  [13:37:03.251] Result of batch 1: 
## INFO  [13:37:03.254]   cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.254]  0.1        5       0.25 4139a8a1-99a6-4eba-94a1-30f42e51978e 
## INFO  [13:37:03.256] Evaluating 1 configuration(s) 
## INFO  [13:37:03.478] Result of batch 2: 
## INFO  [13:37:03.481]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.481]  0.02575       10     0.2227 4a6f0d6c-f69a-46c6-b9d9-42f27f5eb18a 
## INFO  [13:37:03.484] Evaluating 1 configuration(s) 
## INFO  [13:37:03.591] Result of batch 3: 
## INFO  [13:37:03.594]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.594]  0.07525        8       0.25 d6c36ce6-b4ce-4854-b19d-1f8f2134474b 
## INFO  [13:37:03.598] Evaluating 1 configuration(s) 
## INFO  [13:37:03.715] Result of batch 4: 
## INFO  [13:37:03.717]   cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.717]  0.1        1       0.25 ca8a2cc9-3492-4f20-82b8-b9340b853b18 
## INFO  [13:37:03.720] Evaluating 1 configuration(s) 
## INFO  [13:37:03.816] Result of batch 5: 
## INFO  [13:37:03.819]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.819]  0.02575        1     0.2227 bd7dc4dc-aa5f-40fd-b307-f2f9f7b7697f 
## INFO  [13:37:03.822] Evaluating 1 configuration(s) 
## INFO  [13:37:03.922] Result of batch 6: 
## INFO  [13:37:03.925]      cp minsplit classif.ce                                uhash 
## INFO  [13:37:03.925]  0.0505        3       0.25 8945cdae-8b74-489f-97a4-4e1ca77ea7a4 
## INFO  [13:37:03.927] Evaluating 1 configuration(s) 
## INFO  [13:37:04.025] Result of batch 7: 
## INFO  [13:37:04.027]     cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.027]  0.001        1      0.293 09fc8f70-bd7e-4bdd-8319-3172ed6add27 
## INFO  [13:37:04.030] Evaluating 1 configuration(s) 
## INFO  [13:37:04.132] Result of batch 8: 
## INFO  [13:37:04.135]     cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.135]  0.001        5     0.2773 3518e50f-3c90-4e4c-904b-2042b46bbd36 
## INFO  [13:37:04.137] Evaluating 1 configuration(s) 
## INFO  [13:37:04.234] Result of batch 9: 
## INFO  [13:37:04.236]      cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.236]  0.0505        8       0.25 1f43561b-58c2-437d-9a8a-70d64be594a2 
## INFO  [13:37:04.239] Evaluating 1 configuration(s) 
## INFO  [13:37:04.340] Result of batch 10: 
## INFO  [13:37:04.343]   cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.343]  0.1        3       0.25 da469625-63c6-49ce-ac9d-b00ea9350607 
## INFO  [13:37:04.345] Evaluating 1 configuration(s) 
## INFO  [13:37:04.445] Result of batch 11: 
## INFO  [13:37:04.448]     cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.448]  0.001        8     0.2852 8c98e821-8940-4e23-911d-3fcc5db8e952 
## INFO  [13:37:04.451] Evaluating 1 configuration(s) 
## INFO  [13:37:04.557] Result of batch 12: 
## INFO  [13:37:04.560]     cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.560]  0.001        3      0.293 45da92fe-8849-44de-9329-0ce006d1034c 
## INFO  [13:37:04.563] Evaluating 1 configuration(s) 
## INFO  [13:37:04.661] Result of batch 13: 
## INFO  [13:37:04.664]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.664]  0.07525        1       0.25 8800797d-b751-4ddb-9ba5-04cb51eeb692 
## INFO  [13:37:04.667] Evaluating 1 configuration(s) 
## INFO  [13:37:04.767] Result of batch 14: 
## INFO  [13:37:04.770]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.770]  0.07525       10       0.25 b8c1187b-913c-4374-92ec-089e0be46757 
## INFO  [13:37:04.773] Evaluating 1 configuration(s) 
## INFO  [13:37:04.871] Result of batch 15: 
## INFO  [13:37:04.873]     cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.873]  0.001       10     0.2852 36799f3b-9775-4274-8c5b-97b7a4247f9a 
## INFO  [13:37:04.876] Evaluating 1 configuration(s) 
## INFO  [13:37:04.982] Result of batch 16: 
## INFO  [13:37:04.985]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:04.985]  0.07525        3       0.25 e8b60b51-5e64-4509-a516-8187b924c639 
## INFO  [13:37:04.988] Evaluating 1 configuration(s) 
## INFO  [13:37:05.085] Result of batch 17: 
## INFO  [13:37:05.087]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:05.087]  0.02575        5     0.2227 b44c8060-c330-4710-b5d1-6ec497739774 
## INFO  [13:37:05.090] Evaluating 1 configuration(s) 
## INFO  [13:37:05.194] Result of batch 18: 
## INFO  [13:37:05.196]      cp minsplit classif.ce                                uhash 
## INFO  [13:37:05.196]  0.0505        1       0.25 3abf0b56-3337-405e-b000-026455a5b016 
## INFO  [13:37:05.199] Evaluating 1 configuration(s) 
## INFO  [13:37:05.296] Result of batch 19: 
## INFO  [13:37:05.298]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:05.298]  0.02575        8     0.2227 93120580-347a-4dd9-9e91-635f478b2cf4 
## INFO  [13:37:05.301] Evaluating 1 configuration(s) 
## INFO  [13:37:05.403] Result of batch 20: 
## INFO  [13:37:05.405]       cp minsplit classif.ce                                uhash 
## INFO  [13:37:05.405]  0.07525        5       0.25 6e241d72-2726-4ae2-95d2-5be35c9bb030 
## INFO  [13:37:05.413] Finished optimizing after 20 evaluation(s) 
## INFO  [13:37:05.415] Result: 
## INFO  [13:37:05.417]       cp minsplit learner_param_vals  x_domain classif.ce 
## INFO  [13:37:05.417]  0.02575       10          <list[3]> <list[2]>     0.2227
```

```
##         cp minsplit learner_param_vals  x_domain classif.ce
## 1: 0.02575       10          <list[3]> <list[2]>     0.2227
```

```r
instance$result_learner_param_vals
```

```
## $xval
## [1] 0
## 
## $cp
## [1] 0.02575
## 
## $minsplit
## [1] 10
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2227
```

One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##          cp minsplit classif.ce                                uhash  x_domain
##  1: 0.10000        5     0.2500 4139a8a1-99a6-4eba-94a1-30f42e51978e <list[2]>
##  2: 0.02575       10     0.2227 4a6f0d6c-f69a-46c6-b9d9-42f27f5eb18a <list[2]>
##  3: 0.07525        8     0.2500 d6c36ce6-b4ce-4854-b19d-1f8f2134474b <list[2]>
##  4: 0.10000        1     0.2500 ca8a2cc9-3492-4f20-82b8-b9340b853b18 <list[2]>
##  5: 0.02575        1     0.2227 bd7dc4dc-aa5f-40fd-b307-f2f9f7b7697f <list[2]>
##  6: 0.05050        3     0.2500 8945cdae-8b74-489f-97a4-4e1ca77ea7a4 <list[2]>
##  7: 0.00100        1     0.2930 09fc8f70-bd7e-4bdd-8319-3172ed6add27 <list[2]>
##  8: 0.00100        5     0.2773 3518e50f-3c90-4e4c-904b-2042b46bbd36 <list[2]>
##  9: 0.05050        8     0.2500 1f43561b-58c2-437d-9a8a-70d64be594a2 <list[2]>
## 10: 0.10000        3     0.2500 da469625-63c6-49ce-ac9d-b00ea9350607 <list[2]>
## 11: 0.00100        8     0.2852 8c98e821-8940-4e23-911d-3fcc5db8e952 <list[2]>
## 12: 0.00100        3     0.2930 45da92fe-8849-44de-9329-0ce006d1034c <list[2]>
## 13: 0.07525        1     0.2500 8800797d-b751-4ddb-9ba5-04cb51eeb692 <list[2]>
## 14: 0.07525       10     0.2500 b8c1187b-913c-4374-92ec-089e0be46757 <list[2]>
## 15: 0.00100       10     0.2852 36799f3b-9775-4274-8c5b-97b7a4247f9a <list[2]>
## 16: 0.07525        3     0.2500 e8b60b51-5e64-4509-a516-8187b924c639 <list[2]>
## 17: 0.02575        5     0.2227 b44c8060-c330-4710-b5d1-6ec497739774 <list[2]>
## 18: 0.05050        1     0.2500 3abf0b56-3337-405e-b000-026455a5b016 <list[2]>
## 19: 0.02575        8     0.2227 93120580-347a-4dd9-9e91-635f478b2cf4 <list[2]>
## 20: 0.07525        5     0.2500 6e241d72-2726-4ae2-95d2-5be35c9bb030 <list[2]>
##               timestamp batch_nr
##  1: 2020-10-26 13:37:03        1
##  2: 2020-10-26 13:37:03        2
##  3: 2020-10-26 13:37:03        3
##  4: 2020-10-26 13:37:03        4
##  5: 2020-10-26 13:37:03        5
##  6: 2020-10-26 13:37:03        6
##  7: 2020-10-26 13:37:04        7
##  8: 2020-10-26 13:37:04        8
##  9: 2020-10-26 13:37:04        9
## 10: 2020-10-26 13:37:04       10
## 11: 2020-10-26 13:37:04       11
## 12: 2020-10-26 13:37:04       12
## 13: 2020-10-26 13:37:04       13
## 14: 2020-10-26 13:37:04       14
## 15: 2020-10-26 13:37:04       15
## 16: 2020-10-26 13:37:04       16
## 17: 2020-10-26 13:37:05       17
## 18: 2020-10-26 13:37:05       18
## 19: 2020-10-26 13:37:05       19
## 20: 2020-10-26 13:37:05       20
```

In sum, the grid search evaluated 20/25 different configurations of the grid in a random order before the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) stopped the tuning.

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
The `uhash` column links the resampling iterations to the evaluated configurations stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized hyperparameters can take the previously created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), set the returned hyperparameters and train it on the full dataset.


```r
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during tuning and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resamling) is required.

### Automating the Tuning {#autotuner}

The [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) wraps a learner and augments it with an automatic tuning for a given set of hyperparameters.
Because the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically tunes the parameters `cp` and `minsplit` using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and use a simple random search as tuning algorithm:


```r
library("paradox")
library("mlr3tuning")

learner = lrn("classif.rpart")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)
at
```

```
## <AutoTuner:classif.rpart.tuned>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the tuner to a classification tree without tuning.
This way, the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) will do its resampling for tuning on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during tuning or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the tuned learner with the learner that uses default values, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlr3tuning
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id          learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.tuned            cv     3
## 2:  2 <ResampleResult[21]>    pima       classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2461          0
## 2:     0.2604          0
```

Note that we do not expect any differences compared to the non-tuned approach for multiple reasons:

* the task is too easy
* the task is rather small, and thus prone to overfitting
* the tuning budget (10 evaluations) is small
* [rpart](https://cran.r-project.org/package=rpart) does not benefit that much from tuning


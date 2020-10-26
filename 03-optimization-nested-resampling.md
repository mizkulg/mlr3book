## Nested Resampling {#nested-resampling}

In order to obtain unbiased performance estimates for learners, all parts of the model building (preprocessing and model selection steps) should be included in the resampling, i.e., repeated for every pair of training/test data.
For steps that themselves require resampling like hyperparameter tuning or feature-selection (via the wrapper approach) this results in two nested resampling loops.

<img src="images/nested_resampling.png" width="98%" style="display: block; margin: auto;" />

The graphic above illustrates nested resampling for parameter tuning with 3-fold cross-validation in the outer and 4-fold cross-validation in the inner loop.

In the outer resampling loop, we have three pairs of training/test sets.
On each of these outer training sets parameter tuning is done, thereby executing the inner resampling loop.
This way, we get one set of selected hyperparameters for each outer training set.
Then the learner is fitted on each outer training set using the corresponding selected hyperparameters.
Subsequently, we can evaluate the performance of the learner on the outer test sets.

In [mlr3](https://mlr3.mlr-org.com), you can run nested resampling for free without programming any loops by using the [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class.
This works as follows:

1. Generate a wrapped Learner via class [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) or `mlr3filters::AutoSelect` (not yet implemented).
2. Specify all required settings - see section ["Automating the Tuning"](#autotuner) for help.
3. Call function [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) with the created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

You can freely combine different inner and outer resampling strategies.

A common setup is prediction and performance evaluation on a fixed outer test set.
This can be achieved by passing the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) strategy (`rsmp("holdout")`) as the outer resampling instance to either [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html).

The inner resampling strategy could be a cross-validation one (`rsmp("cv")`) as the sizes of the outer training sets might differ.
Per default, the inner resample description is instantiated once for every outer training set.

Note that nested resampling is computationally expensive.
For this reason we use relatively small search spaces and a low number of resampling iterations in the examples shown below.
In practice, you normally have to increase both.
As this is computationally intensive you might want to have a look at the section on [Parallelization](#parallelization).

### Execution {#nested-resamp-exec}

To optimize hyperparameters or conduct feature selection in a nested resampling you need to create learners using either:

* the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class, or
* the `mlr3filters::AutoSelect` class (not yet implemented)

We use the example from section ["Automating the Tuning"](#autotuner) and pipe the resulting learner into a [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call.


```r
library("mlr3tuning")
task = tsk("iris")
learner = lrn("classif.rpart")
resampling = rsmp("holdout")
measure = msr("classif.ce")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamDbl$new("cp", lower = 0.001, upper = 0.1)))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
  param_set, terminator, tuner = tuner)
```

Now construct the [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call:


```r
resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
```

```
## INFO  [13:37:25.761] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [13:37:25.799] Evaluating 1 configuration(s) 
## INFO  [13:37:25.971] Result of batch 1: 
## INFO  [13:37:25.975]     cp classif.ce                                uhash 
## INFO  [13:37:25.975]  0.012     0.0303 ca806d8e-d9ca-4768-87e6-54e82ff57b7f 
## INFO  [13:37:25.978] Evaluating 1 configuration(s) 
## INFO  [13:37:26.169] Result of batch 2: 
## INFO  [13:37:26.172]     cp classif.ce                                uhash 
## INFO  [13:37:26.172]  0.078     0.0303 ec27e0e3-99dc-4640-abde-980557d3bbe2 
## INFO  [13:37:26.176] Evaluating 1 configuration(s) 
## INFO  [13:37:26.292] Result of batch 3: 
## INFO  [13:37:26.295]     cp classif.ce                                uhash 
## INFO  [13:37:26.295]  0.045     0.0303 9d52544b-e9f6-47e7-a9a5-fdda1d498b4b 
## INFO  [13:37:26.299] Evaluating 1 configuration(s) 
## INFO  [13:37:26.407] Result of batch 4: 
## INFO  [13:37:26.410]   cp classif.ce                                uhash 
## INFO  [13:37:26.410]  0.1     0.0303 b431ac42-72dc-48f8-a3f1-1b2097dd83d5 
## INFO  [13:37:26.412] Evaluating 1 configuration(s) 
## INFO  [13:37:26.511] Result of batch 5: 
## INFO  [13:37:26.513]     cp classif.ce                                uhash 
## INFO  [13:37:26.513]  0.034     0.0303 f9dbcd88-763f-4e40-a7ee-e75a9a20c740 
## INFO  [13:37:26.522] Finished optimizing after 5 evaluation(s) 
## INFO  [13:37:26.523] Result: 
## INFO  [13:37:26.525]     cp learner_param_vals  x_domain classif.ce 
## INFO  [13:37:26.525]  0.012          <list[2]> <list[1]>     0.0303 
## INFO  [13:37:26.584] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [13:37:26.589] Evaluating 1 configuration(s) 
## INFO  [13:37:26.680] Result of batch 1: 
## INFO  [13:37:26.682]     cp classif.ce                                uhash 
## INFO  [13:37:26.682]  0.023          0 7171aff1-a7e6-47b0-abc6-59cbf8f6eccd 
## INFO  [13:37:26.685] Evaluating 1 configuration(s) 
## INFO  [13:37:26.787] Result of batch 2: 
## INFO  [13:37:26.790]   cp classif.ce                                uhash 
## INFO  [13:37:26.790]  0.1          0 deef137b-9f01-4dbb-8bc5-766b2c453da7 
## INFO  [13:37:26.792] Evaluating 1 configuration(s) 
## INFO  [13:37:26.889] Result of batch 3: 
## INFO  [13:37:26.892]     cp classif.ce                                uhash 
## INFO  [13:37:26.892]  0.067          0 7ffea7cd-6a6a-46b7-b3d5-327e6ea9e6bf 
## INFO  [13:37:26.895] Evaluating 1 configuration(s) 
## INFO  [13:37:26.997] Result of batch 4: 
## INFO  [13:37:26.999]     cp classif.ce                                uhash 
## INFO  [13:37:26.999]  0.045          0 776335ed-5e99-4ad6-a2c4-e69b90af7fee 
## INFO  [13:37:27.002] Evaluating 1 configuration(s) 
## INFO  [13:37:27.099] Result of batch 5: 
## INFO  [13:37:27.101]     cp classif.ce                                uhash 
## INFO  [13:37:27.101]  0.034          0 cb40f8f8-15f9-4397-98e9-f1f852cbba7e 
## INFO  [13:37:27.109] Finished optimizing after 5 evaluation(s) 
## INFO  [13:37:27.110] Result: 
## INFO  [13:37:27.112]     cp learner_param_vals  x_domain classif.ce 
## INFO  [13:37:27.112]  0.023          <list[2]> <list[1]>          0 
## INFO  [13:37:27.165] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [13:37:27.170] Evaluating 1 configuration(s) 
## INFO  [13:37:27.265] Result of batch 1: 
## INFO  [13:37:27.268]     cp classif.ce                                uhash 
## INFO  [13:37:27.268]  0.012     0.1212 fb6d3023-1b1e-46f0-8d4f-ae4c99cbf537 
## INFO  [13:37:27.271] Evaluating 1 configuration(s) 
## INFO  [13:37:27.369] Result of batch 2: 
## INFO  [13:37:27.371]     cp classif.ce                                uhash 
## INFO  [13:37:27.371]  0.078     0.1212 fbc0cd26-c37b-4f4c-9467-768cd24c668a 
## INFO  [13:37:27.374] Evaluating 1 configuration(s) 
## INFO  [13:37:27.474] Result of batch 3: 
## INFO  [13:37:27.476]     cp classif.ce                                uhash 
## INFO  [13:37:27.476]  0.023     0.1212 cd3cf6d5-bfb6-43fc-98fe-e7a014fe1d86 
## INFO  [13:37:27.479] Evaluating 1 configuration(s) 
## INFO  [13:37:27.577] Result of batch 4: 
## INFO  [13:37:27.580]     cp classif.ce                                uhash 
## INFO  [13:37:27.580]  0.045     0.1212 14efa17f-bd7e-4dbd-86a6-ea1326b0e80b 
## INFO  [13:37:27.583] Evaluating 1 configuration(s) 
## INFO  [13:37:27.683] Result of batch 5: 
## INFO  [13:37:27.686]     cp classif.ce                                uhash 
## INFO  [13:37:27.686]  0.089     0.1212 69280796-137a-417b-88bb-012c66afaa6d 
## INFO  [13:37:27.693] Finished optimizing after 5 evaluation(s) 
## INFO  [13:37:27.695] Result: 
## INFO  [13:37:27.697]     cp learner_param_vals  x_domain classif.ce 
## INFO  [13:37:27.697]  0.012          <list[2]> <list[1]>     0.1212
```

### Evaluation {#nested-resamp-eval}

With the created [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) we can now inspect the executed resampling iterations more closely.
See the section on [Resampling](#resampling) for more detailed information about [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) objects.

For example, we can query the aggregated performance result:


```r
rr$aggregate()
```

```
## classif.ce 
##       0.06
```

Check for any errors in the folds during execution (if there is not output, warnings or errors recorded, this is an empty `data.table()`:


```r
rr$errors
```

```
## Empty data.table (0 rows and 2 cols): iteration,msg
```

Or take a look at the confusion matrix of the joined predictions:


```r
rr$prediction()$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         46         5
##   virginica       0          4        45
```

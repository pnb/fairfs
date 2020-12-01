# Performs Mann-Whitney U-tests
# Results: input data. Name: name to attach to csv. 
# Shift: The step between unfairness weights being compared.
# Column: which feature is being tested
doUTest <- function(results, name, shift, column) {
  # create cutoff for loop based on shift. 9000 comes from data points per metric
  cutoff = 9000-(100*(shift))
  
  # create empty data frame to store results
  all_results <- data.frame("model"="",
                            "unfairness_metric"="",
                            "unfairness_weight_one"="",
                            "unfairness_weight_two"="",
                            "statistic"="",
                            "p-value"=""
                            )
  datalist = list()
  
  for(n in seq(1,cutoff,100)){
    # define indices
    i_one = n
    i_two = n+99
    i_three = n+100 * shift
    i_four = n + 100 * shift + 99
    
    model_first = results[i_one, "model"]
    metric_first = results[i_one, "unfairness_metric"]
    model_second = results[i_three, "model"]
    metric_second = results[i_three, "unfairness_metric"]
    # only do comparison if the model and metric match up
    if(model_first == model_second && metric_first == metric_second){
      weight_first = results[i_one, "unfairness_weight"]
      weight_second = results[i_three, "unfairness_weight"]
      first_column = results[i_one:i_two, column]
      second_column = results[i_three:i_four, column]
      test_results <- wilcox.test(first_column, second_column)
      temp_df = data.frame("model"=model_first,
                   "unfairness_metric"=metric_first,
                   "unfairness_weight_one"=weight_first,
                   "unfairness_weight_two"=weight_second,
                   "statistic"=test_results$statistic,
                   "p-value"=test_results$p.value
                   )
      datalist[[n]] <- temp_df
    }
  }
  all_results = do.call(rbind, datalist)
  write.csv(all_results,paste("/Users/clarabelitz/Documents/git/illinois/fairfs/utest_", shift,"_protected_feature_", name, ".csv", sep=""), row.names = FALSE)
}


# read data. Sorry this is hardcoded... couldn't figure out R's directory situation
sim_data = read.csv('/Users/clarabelitz/Documents/git/illinois/fairfs/fairfs_results-simulated_data.csv')
sp_math_data = read.csv('/Users/clarabelitz/Documents/git/illinois/fairfs/fairfs_results-student-performance-math-updated.csv')
sp_port_data = read.csv("/Users/clarabelitz/Documents/git/illinois/fairfs/fairfs_results-student-performance-port-updated.csv")
sa_data = read.csv('/Users/clarabelitz/Documents/git/illinois/fairfs/fairfs_results-student-academics-updated.csv')

# perform U tests for all steps between 1 and 4, inclusive. Edit to adjust which data is used
for(n in 1:4) {
  doUTest(sp_port_data, "sp_port_data", n, "protected_column_selected_prop")
}


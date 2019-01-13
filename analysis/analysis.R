{
  library(tidyverse)
  library(pROC)
  library(plotROC)
  library(jsonlite)
  library(rjson)
  library(DTComPair)
  library(zeallot)
  devtools::load_all("analysis")
  r <- list() # results list
  rm_these_vars <- c("before_prob", "after_prob", "a_email")
}

# expensive operation, run only if necessary
if (!exists("unfiltered")) {
  unfiltered <- 
    #p_gsub(c("__"), c(""), readLines("data/export20180923.json")) %>% 
    p_gsub(c("__"), c(""), readLines("data/export20181028.json")) %>% 
    map(~ fromJSON(.x, simplify = TRUE)) %>% 
    map_dfr(~ unnest_ratings(.x, rm_these_vars)) %>% 
    group_by(a_identity) %>% 
    nest()
}  

expert_robots <- 
  unfiltered %>% 
  filter(have_enough_ratings(data)) %>%
  filter(contains_pgy(data) | contains_bot(data) | contains_attending(data)
         ) %>% 
  unnest()

experts <- 
  filter(expert_robots, grepl("pgy|ttending", tolower(a_training_level))) 

experts <- spread_ratings(experts)
names(experts) <- make.names(names(experts))

experts[["consensus"]] <- 
  select(experts, starts_with("X")) %>% 
  get_consensus()

robot <- 
  expert_robots %>% 
  filter(a_identity == "RhinoNet") %>% 
  filter(i_filepath %in% experts[["i_filepath"]]) %>% 
  select(i_filepath, RhinoNet = y_pred)

consensus <- left_join(experts, robot)

r[["kappas"]] <- 
  imap_dfr(
    list(consensus    = c("truth", "consensus"), 
         rhinonet     = c("truth", "RhinoNet"),
         consensus_v_rhinonet = c("consensus", "RhinoNet")
         ), 
      function(v_nms, nm) mutate(gauge(consensus[,v_nms]), test = nm)
    ) %>% 
  select(test, everything())


# best case: human-augmented predictions
#            see RhinoNet/consensus accordance for potential benefit?

con <- modify(consensus[,c("truth", "consensus", "RhinoNet")], as.numeric)
r[["t"]] <- 
  tab.paired(
    d = con$truth, 
    y1 = con$consensus, 
    y2 = con$RhinoNet, 
    testnames = c("consensus", "rhinonet")
    )

# try a few confidence interval methods, preference for CI
measurements <- 
  list(acc.paired, dlr.regtest, sesp.diff.ci, 
       pv.gs, sesp.mcnemar, sesp.exactbinom)

tidiers <- 
  list(tidy_acc, tidy_dlr, tidy_ses, 
       tidy_xpv, tidy_ses, tidy_ses)

r[c("perf", "pdlr", "sens_diff", "ppv_diff", "perf_mcn", "perf_bin")] %<-%
  (map2(tidiers, measurements, compose) %>% map(~ .x(r[["t"]])))
  
# map(c("wald", "agresti-min", "bonett-price", "tango"), 
#     ~ sesp.diff.ci(r$t, ci.method = .x))

performance <- 
  left_join(r$perf, r$kappas) %>% 
    t() %>% 
    (function(z) {
      df <- as.data.frame(z[-1,])
      names(df) <- z[1,]
      df
    }) %>% 
  rownames_to_column(var = "measure") 

# measures of classification performance
msrs <- list()

msrs[["s"]] <-  
  r$sens_diff %>% 
  modify_at(c("diff.lcl", "diff.ucl"), ~ sprintf("%.2f", .x)) %>% 
  modify_at("diff", ~ sprintf("%.3f", .x)) %>% 
  mutate(difference = paste(diff, " (", diff.lcl, "-", diff.ucl, ")", sep = "")
         ) %>% 
  select(measure, difference) 

msrs[["p"]] <- 
  r$ppv_diff %>% 
    modify_at("p.value", ~ sprintf("%.4f", .x)) %>% 
    modify_at("diff", ~ sprintf("%.3f", .x)) %>% 
    mutate(difference = paste(diff, " (p: ", p.value, ")", sep = "")
    ) %>% 
    select(measure, difference)

msrs[["pd"]] <- 
  data.frame(
    measure = "pdlr", 
    difference = 
       paste(r$pdlr[["ratio"]],
             " (",
             r$pdlr[["lcl"]], 
             ", ",
             r$pdlr[["ucl"]], 
             ")", sep = "")
  )

r[["performance"]] <- left_join(performance, reduce(msrs, bind_rows))

rm(msrs)

write_csv(r$performance, "analysis/results/performance.csv")

# Plot materials
everyone <- 
  unfiltered %>% 
  filter(have_enough_ratings(data)) %>%
  unnest()

everyone[["a_training_level"]] <- 
  ifelse(everyone$a_identity == "RhinoNet", "Model", everyone[["a_training_level"]])

everyone <- group_by(everyone, a_identity) %>% nest()

perf <- 
  map_dfr(everyone[["data"]], ~ gauge(.x[,c("y_pred", "y_actual")])) %>% 
  mutate(training = map_chr(everyone[["data"]], get_training), 
         id = everyone$a_identity, 
         viewcount = map_int(everyone[["data"]], nrow)) %>% 
  separate(kappa, c("v", "ci"), " \\(") %>% 
  modify_at("ci", ~ gsub("\\)", "", .x)) %>% 
  separate(ci, c("lci", "uci"), "-") %>% 
  modify_at(c("v", "lci", "uci"), as.numeric) %>% 
  filter(training != "Unlisted") %>%
  modify_at("training", 
            ~ factor(.x, levels = c("Model", "Attending", "PGY-5", 
                                    "PGY-4", "PGY-3", "PGY-2", "None")))

perf <- 
  group_by(perf, training) %>% 
  summarise(vc = sum(viewcount)) %>% 
  right_join(perf)

r[["plot"]] <- 
  ggplot(perf, aes(x = training, y = v)) + #, color = id
    geom_errorbar(aes(ymin = lci, ymax = uci, width = 0.25)) + 
    geom_point(size = 2) + 
    theme(legend.position = "none") + 
    geom_text(aes(label = vc, y = 0.85)) + 
    labs(x = "Training", y = "Kappa (95% CI)")

ggsave("analysis/results/kappa-figure.jpeg", 
       r$plot, 
       device = "jpeg", 
       width = 6, 
       height = 4, 
       units = "in", 
       dpi = 300
)

# Number of photos and number of raters

r[["training_levels"]] <- 
  data.frame(level = map_chr(unfiltered$data, get_training), 
             count = map_int(unfiltered$data, nrow))

r[["viewcounts"]] <- 
  r$training_levels %>% 
    group_by(level) %>% 
    summarise(rated = sum(count, na.rm = T)) 
  
r[["clinician_viewcounts"]] <- 
  r$viewcounts %>% filter(!(level %in% c("None", "Unlisted")))

r[["clinician_total_viewcount"]] <- sum(r$clinician_viewcounts$rated)




  
  
  









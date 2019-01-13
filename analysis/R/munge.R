



training_levels <- function(rating) {
  tl <- "a_training_level"
  if (is.null(rating[[tl]])) return("NA")
  else rating[[tl]]
}

unnest_ratings <- function(json_rtngs, rm_these = NULL) {
  nstd_nms <- c(a = "author", i = "image", k = "key")
  # for each of the above nested columns, extract its contents and prepend an identifier
  mk_new_names <- function(a, b) paste(a, names(json_rtngs[[b]]), sep = "_")
  for (i in seq_along(nstd_nms)) {
    new_name <- mk_new_names(names(nstd_nms)[i], nstd_nms[i])
    json_rtngs[new_name] <- json_rtngs[[nstd_nms[i]]] 
  }
  # remove nested columns
  json_rtngs[map_lgl(json_rtngs, is.list)] <- NULL
  
  # remove other columns 
  if (!missing(rm_these)) {
    rm_these <- c("before_prob", "after_prob", rm_these)#, "a_training_level")
    rm_these_cols <- grep(paste(rm_these, collapse = "|"), names(json_rtngs), invert = T)
    json_rtngs <- json_rtngs[rm_these_cols]
  }
  
  # add training levels (i.e. PGY, attending, etc)
  json_rtngs[["a_training_lvl"]] <- training_levels(json_rtngs)
  
  data.frame(json_rtngs, stringsAsFactors = FALSE)
}

p_gsub <- function(ps, rs, vs) {
  reduce2(ps, rs, function(a, x, y) gsub(x, y, a), .init = vs) 
}

check_the <- function(pattern, this_col) {
  function(the_df) 
    any(grepl(pattern, unique(tolower(the_df[[this_col]]))))
}

have_enough_ratings <- function(the_dfs) {
  map_lgl(the_dfs, ~ nrow(.x) > 100)
}

contains_pgy <- function(the_dfs) {
  map_lgl(the_dfs, check_the("pgy", "a_training_level"))
}

contains_bot <- function(the_dfs) {
  map_lgl(the_dfs, check_the("rhinonet", "a_display_name"))
}

contains_attending <- function(the_dfs) {
  map_lgl(the_dfs, check_the("attending", "a_training_level"))
}


which_everyone_saw <- function(ratings_df) {
  keep_these_pics <- 
    group_by(ratings_df, a_identity) %>% 
    nest() %>% 
    (function(z) z[which.min(map_int(z[["data"]], nrow)),]) %>% 
    unnest() %>%
    select(i_filepath) %>% 
    unlist()
  filter(ratings_df, i_filepath %in% keep_these_pics)
}

get_consensus <- function(wide_df) {
  apply(wide_df, 1, 
        function(z) names(sort(table(unlist(z)), decreasing = T)[1]))
}

spread_ratings <- function(ratings_df) {
  ratings_df %>% 
    select(i_filepath, ppl = a_identity, truth = y_actual, rated = y_pred) %>% 
    # we only want people's ratings for each photo, looked like user 108719145923577102678 saw bb1331ec2275273056582817cee06a8e5ac9904567b558882e6a1e8ec0932576 twice
    distinct() %>% 
    #(function(z) browser()) %>% 
    spread(ppl, rated)
}


gauge <- function(ratings_df) {
  stopifnot(ncol(ratings_df) == 2)
  k <- psych::cohen.kappa(as.matrix(ratings_df))[c("kappa", "confid")]
  r <- 
    data.frame(
      k_lower_conf = k[["confid"]][1,1],
      kappa        = k[["kappa"]],
      k_upper_conf = k[["confid"]][1,3]
    )
  modify_at(r, "kappa", ~ sprintf("%.3f", .x)) %>% 
    modify_at(c("k_lower_conf", "k_upper_conf"), 
              ~ sprintf("%.2f", .x)) %>% 
    mutate(kappa1 = paste(kappa, " (", k_lower_conf, "-", k_upper_conf, ")", sep = "")) %>% 
    select(kappa = kappa1)
}


tidy_acc_part <- function(acc) {
  nm <- gsub("con\\$", "", acc[["testname"]])
  acc[c("tab", "alpha", "testname", "ndlr")] <- NULL
  process_acc_r <- function(z) {
    names(z) <- c("e", "s", "l", "u")
    th <- function(a) sprintf("%.3f", a)
    tw <- function(a) sprintf("%.2f", a)
    paste(th(z[["e"]]), 
          " (", 
          tw(z[["l"]]), 
          "-", 
          tw(z[["u"]]), 
          ")", 
          sep = "")
  }
  bind_cols(tibble(test = nm), map_dfc(acc, process_acc_r))
}

tidy_acc <- function(b) map_dfr(b, tidy_acc_part)

tidy_dlr <- function(dlr) {
  pdlr <- dlr[["pdlr"]]
  pdlr[c("test1", "test2", "se.log", "test.statistic")] <- NULL
  r <- data.frame(pdlr)
  two_dgt_vars <- c("ratio", "lcl", "ucl")
  r[two_dgt_vars] <- lapply(r[,two_dgt_vars], function(z) sprintf("%.1f", z))
  r[["p.value"]] <- sprintf("%.4f", r[["p.value"]])
  select(r, ratio, lcl, ucl, everything())
}
  
tidy_gen <- function(a, b) {
  function(ss_ci) {
    names(ss_ci[[a]]) <- gsub(a, "", names(ss_ci[[a]]))
    names(ss_ci[[b]]) <- gsub(b, "", names(ss_ci[[b]]))
    ss_ci <- bind_rows(ss_ci[[a]], ss_ci[[b]])
    ss_ci[["measure"]] <- c(a, b)
    names(ss_ci)[1:2] <- c("consensus", "rhinonet")
    #ss_ci[["p.value"]] <- c(ss_ci[[a]][["p.value"]], ss_ci[[b]][["p.value"]])
    select(ss_ci, measure, everything())
  }
}

tidy_ses <- tidy_gen("sensitivity", "specificity")
tidy_xpv <- tidy_gen("ppv", "npv")

get_training <- function(a_df) {
  a <- unique(a_df[["a_training_level"]])
  a <- a[!is.na(a)]
  a <- grep("PGY|ttending|one|odel", a, value = T)
  if (length(a) == 0) return("Unlisted")
  else a
}



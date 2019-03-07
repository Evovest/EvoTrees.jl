library(data.table)
library(fst)
library(xgboost)

# dt <- read_fst("./data/performance_tot_v2_perc.fst", as.data.table = T)
dt <- fread("./data/performance_tot_v2_perc.csv")
# dt <- dt[TYPE == "TRAIN"]
# dt[, TYPE := NULL]
# dt[, ID_CPTE := NULL]

# fwrite(dt, file = "./data/performance_tot_v2_perc.csv")
#
# iris <- datasets::iris
# fwrite(iris, file = "./data/iris.csv")

data <- dt[, 1:53]
label <- dt$Default

params <- list(subsample = 0.5,
               max_depth = 5,
               eta = 0.1,
               colsample_bytree = 0.5,
               min_child_weight = 1,
               lambda = 0,
               alpha = 0,
               gamma = 0,
               objective = "reg:linear",
               eval_metric = "rmse",
               nthread = 1)


train_id <- sample(nrow(data), size = as.integer(0.8 * nrow(data)), replace = F)
x_train <- data[train_id]
x_eval <- data[-train_id]

y_train <- label[train_id]
y_eval <- label[-train_id]

xgb_train <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
xgb_eval <- xgb.DMatrix(data = as.matrix(x_eval), label = y_eval)

watchlist <- list(train = xgb_train, eval = xgb_eval)
system.time(model <- xgb.train(data = xgb_train, watchlist = watchlist,
                             params = params, nrounds = 100, verbose = 1, print_every_n = 10L,
                             early_stopping_rounds = NULL))

# xgboost::xgb.plot.tree(model = model, trees = 1)
pred <- predict(model, as.matrix(data))
mean((label - pred) ^ 2)

# sort(table(pred))
summary(pred)

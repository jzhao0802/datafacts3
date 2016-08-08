'''
This function could run standard LR  with Spark-submit in Spark ML module.
RegParam is regularization parameter (>= 0).elasticNetParam should be 1.
'''

# Import the packages
import sys
import os
import time
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors
import numpy as np
#import pandas as pd
import random
import csv
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import monotonicallyIncreasingId

#from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
#from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

app_name = sys.argv[1]
s3_path = "s3://emr-rwes-pa-spark-dev-datastore"
seed = 42
par = 5

######## !!!!!!!Some variables, change before running!!!!!!############
# Path variables
data_path = s3_path + "/BI_IPF_2016/01_data/"
s3_outpath = s3_path + "/BI_IPF_2016/02_result/"
master_path = "/home/hjin/BI_IPF_2016/03_result/"
master_data_path = "/home/hjin/BI_IPF_2016/01_data/"

# data file
pos_file = "Ipf_cohort_replace_extremes.csv"
        #"/Ipf_md_dt_cap_ext.csv"
neg_file ="non_Ipf_cohort_replace_extremes.csv"
    #"/nonipf_md_dt_cap_ext.csv"
ss_file = '/score_sample_model_data_cap_ext.csv'
exc_file = 'vars_to_exclude.csv'

# Number of simulation
num_sim = 5

#########Setting End##########################################################

# Don't need to be setup
# seed
random.seed(seed)
seed_seq = [random.randint(10, 100) for i in range(num_sim)]

# S3 output folder
start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
resultDir_s3 = s3_outpath + app_name + "_" + st + "/"

# Master node output folder
resultDir_master = master_path + app_name + '_' + st + "/"
if not os.path.exists(resultDir_master):
    os.makedirs(resultDir_master)

os.chmod(resultDir_master, 0o777)

#function 1: get the predicted probability in Vector
def parsePoint(line):
    return LabeledPoint(line.label, line.features)

#function 2: to add simulation or iteration ID
def addID(dataset, number, npar, name):
    nPoses = dataset.count()
    npFoldIDsPos = np.array(list(range(number)) * np.ceil(float(nPoses) / number))
    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]
    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)
    rddFoldIDs = sc.parallelize(npFoldIDs, npar).map(int)
    dfDataWithIndex = dataset.rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select('orgData.matched_positive_id', 'key') \
        .withColumnRenamed('key', name) \
        .coalesce(npar)
    return dfJoined


#function 3: PR curve calculation
def pr_curve(data, resultDir_s3, output):
    #get label and prob from the dataframe
    labelsAndProbs = data.withColumn('prob_1', round(data.prob_1,3))\
        .select(["label", "prob_1"])
    #get distinct threshold values
    thresholds_interim = labelsAndProbs\
        .select(col('prob_1').alias("threshold"))\
        .distinct()
    #num_partitions = thresholds_interim.rdd.getNumPartitions()
    thresholds = thresholds_interim.coalesce(10)
    #cache dataframes
    labelsAndProbs.cache()
    thresholds.cache()
    PRs = thresholds.join(labelsAndProbs)\
    	.withColumn("pred", when(col("prob_1") > col("threshold"),1.0).otherwise(0.0))\
    	.withColumn("bTP", when((col("label") == col("pred")) & (col("pred") == 1),1.0).otherwise(0.0))\
    	.withColumn("bFP", when((col("label") != col("pred")) & (col("pred") == 1),1.0).otherwise(0.0))\
    	.withColumn("bTN", when((col("label") == col("pred")) & (col("pred") == 0),1.0).otherwise(0.0))\
    	.withColumn("bFN", when((col("label") != col("pred")) & (col("pred") == 0),1.0).otherwise(0.0))\
        .select(col("threshold"),col("bTP"),col("bFP"),col("bTN"),col("bFN"))\
        .groupBy("threshold")\
        .agg(sum(col("bTP")).alias("nTPs"),
                         sum(col("bFP")).alias("nFPs"),
                         sum(col("bTN")).alias("nTNs"),
                         sum(col("bFN")).alias("nFNs"))\
        .select(round(col("nTPs") / (col("nTPs") + col("nFPs") + 1e-9),3).alias("precision"),
			round(col("nTPs") / (col("nTPs") + col("nFNs") + 1e-9),3).alias("recall"),
			col("threshold"))\
        .coalesce(1)\
        .save((resultDir_s3+output),"com.databricks.spark.csv",header="true")

#function 4: register template table
def regit(table, isim, i,name):
    return table[isim][i].registerTempTable((name + str(isim)))

#function 5: create SQL query
def sqlqu(nsim):
    iquery = 'SELECT * FROM sim_0 UNION ALL '
    for i in range(1, nsim-1):
        iquery = iquery + 'SELECT * FROM sim_' + str(i) + ' UNION ALL '
    query = iquery + 'SELECT * FROM sim_' + str(nsim-1)
    return query

def sqlqu2(num_part):
    iquery = 'SELECT * FROM pred_0 UNION ALL '
    for i in range(1, num_part-1):
        iquery = iquery + 'SELECT * FROM pred_' + str(i) + ' UNION ALL '
    query = iquery + 'SELECT * FROM pred_' + str(num_part-1)
    return query


#function 6: generate excluded variable list
def exc_list(d_path, file):
    data = np.loadtxt(d_path + file ,dtype=np.str,delimiter=',',skiprows=0)
    var_ls = data[1:, 0].tolist()
    var_flag_ls = [x + '_FLAG' for x in var_ls]
    var_avg_ls = [x + '_AVG_RXDX' for x in var_ls]
    var_exc_ls = var_flag_ls + var_avg_ls
    return var_exc_ls

#function 7: run cross-validation
def sim_function(isim, patsim, dataset, ss_ori):

    #select patients in each simulation from patsim
    valsimid = patsim.filter(patsim.simid == isim)

    sssim = ss_ori\
        .join(valsimid,valsimid.matched_positive_id==ss_ori.matched_positive_id,'inner')\
        .select(ss_ori.matched_positive_id, ss_ori.label, ss_ori.patid,ss_ori.features)


    #select corresponding trainning and test set
    valsim = dataset\
        .join(valsimid, valsimid.matched_positive_id==dataset.matched_positive_id,
              'inner')\
        .select(dataset.matched_positive_id, dataset.label, dataset.patid,dataset.features)

    trsim = dataset.subtract(valsim)

    #get LabeledandPoint rdd data
    trsimrdd = trsim.map(parsePoint)
    valsimrdd = valsim.map(parsePoint)
    sssimrdd = sssim.map(parsePoint)

    # Build the model
    sim_model = LogisticRegressionWithLBFGS.train(trsimrdd, intercept=True,
                                            regType=None)
    #clear the threshold
    sim_model.clearThreshold()

    #output model
    sim_model.save(sc, resultDir_s3+"model_sim"+str(isim))
    #load model
    #model = LogisticRegressionModel.load(sc, resultDir_s3+"model_sim"+str(isim))

    #predict on test data
    scoreAndLabels_val = valsimrdd.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label))
    scoreAndLabels_ss = sssimrdd.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label))

    #Identify the probility of response
    pred_score_val = scoreAndLabels_val.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')

    pred_score_ss = scoreAndLabels_ss.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')

    return [pred_score_val, pred_score_ss]

#define the main function
def main(sc, data_path=data_path, pos_file=pos_file, ss_file=ss_file,
         neg_file= neg_file, num_sim=num_sim, seed=seed,
         seed_seq=seed_seq, par=par, resultDir_s3=resultDir_s3,
         resultDir_master=resultDir_master):

    #reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')
    ss = sqlContext.read.load((data_path + ss_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')
    ss = ss.withColumnRenamed("patid", "matched_positive_id")\
             .withColumnRenamed("nonipf_patid", "patid")

    pos = pos.drop("nonipf_patid")
    pos = pos.withColumn("matched_positive_id", pos["patid"])
    neg = neg.withColumnRenamed("patid", "matched_positive_id")\
             .withColumnRenamed("nonipf_patid", "patid")

    #reading in excluded variable list from master node
    exc_var_list = exc_list(master_data_path, exc_file)

    #see the column names
    pos_col = pos.columns

    #include variable list
    common_list = ['matched_positive_id', 'label', 'patid']
    inc_var = [x for x in pos_col if x not in exc_var_list+common_list]

    #combine features
    assembler = VectorAssembler(inputCols=inc_var,outputCol="features")
    #assembler_neg = VectorAssembler(inputCols=inc_var,outputCol="features")

    #get the input positive and negative dataframe
    pos_asmbl = assembler.transform(pos)\
        .select('matched_positive_id', 'label', 'patid', 'features')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double'))

    neg_asmbl = assembler.transform(neg)\
        .select('matched_positive_id', 'label', 'patid', 'features')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double'))

    ss_asmbl = assembler.transform(ss)\
        .select('matched_positive_id', 'label', 'patid', 'features')
    ss_ori = ss_asmbl.withColumn('label', ss_asmbl['label'].cast('double'))

    #union All positive and negative data as dataset
    dataset = pos_ori.unionAll(neg_ori)

    #create a dataframe which has 2 column, 1 is patient ID, other one is simid
    patid_pos = pos_ori.select('matched_positive_id')
    patsim = addID(patid_pos, num_sim, par, 'simid')

    sim_result_ls = [sim_function(isim, patsim=patsim, dataset=dataset, ss_ori=ss_ori)
                 for isim in range(num_sim)]

    #register each result into temp table
    for d in range(num_sim):
        regit(sim_result_ls, d, 0,'sim_')

    #create the query statment
    sqlquery = sqlqu(num_sim)

    #combine results from simulations
    dataset_pred = sqlContext.sql(sqlquery)

    #register each result into temp table
    for d in range(num_sim):
        regit(sim_result_ls, d, 1,'pred_')

    #create the query statment
    sqlquery2 = sqlqu2(num_sim)

    #combine results from simulations
    ss_pred = sqlContext.sql(sqlquery2)

    #output the predicted score on test data
    all_ss_pred = ss_pred.unionAll(dataset_pred).repartition(1000)

    #train the model on validation and prediction on holdout
    datasetrdd = dataset.map(parsePoint)

    # Build the model
    model = LogisticRegressionWithLBFGS.train(datasetrdd, intercept=True,
                                              regType=None)
    #clear the threshold
    model.clearThreshold()

    #output the parameters to results simulation
    intercept = model.intercept
    coef = model.weights
    coef_file = open(resultDir_master + 'Coeff.txt', "w")
    coef_file.writelines("Intercept, %f" %intercept)
    coef_file.writelines("\n")
    for id in range(len(coef)):
        coef_file.writelines("%s , %f" %(inc_var[id] ,coef[id]))
        coef_file.writelines("\n")
    coef_file.close()
    os.chmod(resultDir_master + 'Coeff.txt', 0o777)

    #predict on dataset
    scoreAndLabels_data = dataset_pred.rdd

    #calculate the metrics
    metrics_tr = BinaryClassificationMetrics(scoreAndLabels_data)
    #metrics_ts = BinaryClassificationMetrics(scoreAndLabels_ts)

    #AUC & AUPR
    AUC_tr = metrics_tr.areaUnderROC
    #AUC_ts = metrics_ts.areaUnderROC
    AUPR_tr = metrics_tr.areaUnderPR
    #AUPR_ts = metrics_ts.areaUnderPR

    #print out AUC results
    auc = "Training data AUC = %s " % AUC_tr + "\n"
          #+ "Test data AUC = %s " % AUC_ts + "\n"
    aupr = "Training data AUPR = %s " % AUPR_tr + "\n"
           #+ \ "Test data AUPR = %s " % AUPR_ts
    auc_file = open(resultDir_master + 'AUC_AUPR.txt', "w")
    auc_file.writelines(auc)
    auc_file.writelines(aupr)
    auc_file.close()
    os.chmod(resultDir_master + 'AUC_AUPR.txt', 0o777)


    #output the predicted score on test data
    dataset_pred.save((resultDir_s3+'pred_score_alldata'),
                                 "com.databricks.spark.csv",header="true")
    all_ss_pred.save((resultDir_s3+'pred_score_allss'),
                                 "com.databricks.spark.csv",header="true")
    #output PR curve for dataset through simulation
    pr_curve(dataset_pred, resultDir_s3=resultDir_s3, output='PR_cohort_data/')
    pr_curve(all_ss_pred, resultDir_s3=resultDir_s3, output='PR_ss_data/')

if __name__ == "__main__":
    
    sc = SparkContext(appName = app_name)
    sqlContext = SQLContext(sc)


    #call main function
    main(sc)


    sc.stop()
    




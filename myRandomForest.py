# -*- coding: utf-8 -*-
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.sql.types import StructField, StructType, FloatType, StringType, DoubleType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import ChiSqSelector
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.functions import rand
from pyspark.mllib.classification import SVMWithSGD
from pyspark.sql import SparkSession,SQLContext #2.1
from pyspark import SparkConf, SparkContext
import math
import numpy as np

import dataProcess
import module

def myRandomForest(training, test):
    # 设置默认分类器
    rf = RandomForestClassifier(featuresCol='features', labelCol='label', cacheNodeIds=True, featureSubsetStrategy='all')

    # 构造参数网格
    paramGrid_rf = ParamGridBuilder().addGrid(rf.maxDepth,[12]).addGrid(rf.numTrees,[100]).build()

    # modeling类调用方式
    bestModel_rf, best_epm_rf, best_sampling_rf = module.modeling()._fit(training, rf, paramGrid_rf, [0.01,0.005],2)

    # 预测
    all_list = training.columns
    all_list.remove('"Class"')
    assembler = VectorAssembler().setInputCols(all_list).setOutputCol("features_vector")
    test = assembler.transform(test)
    predictions_rf = bestModel_rf.transform(test)

    # 混淆矩阵
    predictions_rf.groupBy('label','prediction').count().show()
    return predictions_rf
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

def myGBT(training, test, labelColumnName):
    # 设置默认分类器
    gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", cacheNodeIds=True)

    # 构造参数网格
    paramGrid_gbt = ParamGridBuilder().addGrid(gbt.maxDepth,[5,8,10]).addGrid(gbt.minInfoGain,[0.0,0.001]).addGrid(gbt.minInstancesPerNode,[1,3]).addGrid(gbt.maxIter, [100,150,200]).addGrid(gbt.stepSize, [0.01,0.1]).build()

    # modeling类调用方式
    bestModel_gbt, best_epm_gbt, best_sampling_gbt = module.modeling()._fit(training, gbt, paramGrid_gbt, [0.2,0.5,0.8], 3)

    # 预测
    all_list = training.columns
    all_list.remove(labelColumnName)
    assembler = VectorAssembler().setInputCols(all_list).setOutputCol("features_vector")
    test = assembler.transform(test)
    predictions_gbt = bestModel_gbt.transform(test)

    # 混淆矩阵
    predictions_gbt.groupBy('label','prediction').count().show()
    return predictions_gbt
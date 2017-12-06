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
import myRandomForest

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName("modeling").getOrCreate()

path = "/home/truenyl/aws/data/creditcard.csv"
training, test = dataProcess.dataProcess(sc, spark, path)

prediction_rf = myRandomForest.myRandomForest(training, test)

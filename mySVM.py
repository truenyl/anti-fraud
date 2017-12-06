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
from pyspark.sql import SparkSession, SQLContext #2.1
import math
import numpy as np


import dataProcess
import module

def mySVM(training, test):
	# SVM
	training_svc = training.map(lambda x:LabeledPoint(x[29],x[1:28]))
	sv = SVMWithSGD.train(training_svc,iterations = 100,step = 0.1,regParam = 0.01)
	test_svc = test.map(lambda x:LabeledPoint(x[29],x[1:28]))
	predictions = test_svc.map(lambda x:(x.TARGET,float(sv.predict(x.features))))
	return predictions
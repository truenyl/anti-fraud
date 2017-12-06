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
from pyspark.sql import SparkSession, SQLContext  # 2.1
from pyspark import SparkConf, SparkContext
import math
import numpy as np


# 转LabeledPoint格式
def ToFL(lines):
    values = [float(x) for x in lines]
    return LabeledPoint(values[0:-1], values[-1])


# 数据类型转float
def ToF(lines):
    values = [float(x) for x in lines]
    return values


# fill null
def isnull(x):
    if (x == 'null'):
        return 0
    else:
        return x


def fillnanull(x):
    x = map(isnull, x)
    return x


# "0" >>> 0
def is_0(x):
    if (x == '"0"'):
        return 0
    else:
        return x


def fill_0(x):
    x = map(is_0, x)
    return x


# “1” >>> 1
def is_1(x):
    if (x == '"1"'):
        return 1
    else:
        return x


def fill_1(x):
    x = map(is_1, x)
    return x


# 离散化
def ToDiscrete(lines):
    values = [math.floor(float(x)) for x in lines]
    return values


# 读csv为RDD
def dataProcess(sc, spark, path):
    # type: (object, object, object) -> object
    data = sc.textFile(path).map(lambda x: x.split(',')).map(fill_1).map(fill_0).map(fillnanull)
    head = data.take(1)[0]
# 去掉列名行
    data = data.filter(lambda lines: lines[0] != '"Time"')
    field_string = [StructField(field_name, StringType(), True) for field_name in head]
    schema_string = StructType(field_string)
    data_string = spark.createDataFrame(data, schema_string)
    head = data_string.columns
    data_string = data_string.rdd.map(ToF)
    '''
    #databricks包读csv为DataFrame
    data = sc.load(source = 'com.databricks.spark.csv',header = 'true',path = '/home/truenyl/aws/data/creditcard.csv')
    data = data.drop('Time')
    head = data.columns
    data = data.map(fillnanull).map(ToF)
    '''

# RDD转为DataFrame
    field = [StructField(field_name, DoubleType(), True) for field_name in head]
    schema = StructType(field)
    data_new = spark.createDataFrame(data_string, schema)
    data_new = data_new.drop('"Time"')
    training, test = data_new.randomSplit([0.6, 0.4], seed=24)
    training.cache()
    return training, test
